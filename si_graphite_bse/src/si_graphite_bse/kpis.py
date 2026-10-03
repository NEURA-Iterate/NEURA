"""Per-image KPIs computed from the label map (pore / graphite / Si / CBD / gap) and detector images."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from skimage.filters import sato
from skimage.measure import regionprops_table
from skimage.morphology import remove_small_objects, skeletonize
from skimage.segmentation import watershed

from .config import KPIConfig
from .pipeline import SegmentationResult
from .segment import CBD, GAP, GRAPHITE, PORE, SI

PARTICLE_PROPS = (
    "label",
    "area",
    "centroid",
    "equivalent_diameter_area",
    "axis_major_length",
    "axis_minor_length",
    "orientation",
)
GRAPHITE_DOWNSAMPLE = 2


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cum, q / 100 * cum[-1])])


def si_particles(si: np.ndarray, scale: float) -> pd.DataFrame:
    labels, _ = ndi.label(si)
    df = pd.DataFrame(regionprops_table(labels, properties=PARTICLE_PROPS))
    if df.empty:
        return df
    h, w = si.shape
    bbox = ndi.find_objects(labels)
    df["touches_border"] = [
        s[0].start == 0 or s[1].start == 0 or s[0].stop == h or s[1].stop == w for s in bbox
    ]
    df["ecd"] = df["equivalent_diameter_area"] * scale
    df["aspect_ratio"] = df["axis_major_length"] / df["axis_minor_length"].clip(lower=1e-6)
    return df.drop(columns=["equivalent_diameter_area"])


def graphite_particles(graphite: np.ndarray, min_area_px: int) -> pd.DataFrame:
    """Split touching graphite flakes with a distance-transform watershed (on a 2x downsampled mask)."""
    g = graphite[::GRAPHITE_DOWNSAMPLE, ::GRAPHITE_DOWNSAMPLE]
    dist = ndi.distance_transform_edt(g)
    comp, _ = ndi.label(g)
    peaks = peak_local_max(dist, min_distance=10, labels=comp, exclude_border=False)
    markers = np.zeros(g.shape, dtype=np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    ws = watershed(-dist, markers, mask=g)
    df = pd.DataFrame(regionprops_table(ws, properties=PARTICLE_PROPS))
    df = df[df["area"] >= min_area_px / GRAPHITE_DOWNSAMPLE**2].copy()
    df["aspect_ratio"] = df["axis_major_length"] / df["axis_minor_length"].clip(lower=1e-6)
    df["ecd_px"] = df["equivalent_diameter_area"] * GRAPHITE_DOWNSAMPLE
    return df


def homogeneity_cv(si: np.ndarray, solids: np.ndarray, window: int, min_solid: float) -> float:
    """Coefficient of variation of the Si fraction of solids over non-overlapping windows."""
    h, w = (si.shape[0] // window) * window, (si.shape[1] // window) * window
    if h == 0 or w == 0:
        return float("nan")

    def tile_sums(m: np.ndarray) -> np.ndarray:
        return m[:h, :w].reshape(h // window, window, w // window, window).sum(axis=(1, 3)).ravel()

    si_t, sol_t = tile_sums(si), tile_sums(solids)
    ok = sol_t >= min_solid * window * window
    if ok.sum() < 2:
        return float("nan")
    frac = si_t[ok] / sol_t[ok]
    return float(frac.std() / frac.mean()) if frac.mean() > 0 else float("nan")


def through_thickness_profile(si: np.ndarray, solids: np.ndarray, bands: int) -> np.ndarray:
    rows_si = si.sum(axis=1)
    rows_sol = solids.sum(axis=1)
    return np.array(
        [
            s.sum() / max(t.sum(), 1)
            for s, t in zip(np.array_split(rows_si, bands), np.array_split(rows_sol, bands), strict=False)
        ]
    )


def dispersion_index(
    centroids: np.ndarray, areas: np.ndarray, shape: tuple[int, int], window: int, n_shuffles: int, rng
) -> tuple[float, float]:
    """Window CV of Si area divided by the mean CV with the same particles placed uniformly at random.

    Each particle's area goes to the window containing its centroid in both cases, so the ratio removes
    the effect of particle count and size: 1 = random placement, > 1 = clumped. Also returns the
    spread (sd) of the ratio under random placement, so values within ~2 sd of 1 read as random.
    """
    ny, nx = shape[0] // window, shape[1] // window
    if ny * nx < 2 or len(areas) < 2:
        return float("nan"), float("nan")

    def cv(c: np.ndarray) -> float:
        iy, ix = (c[:, 0] // window).astype(int), (c[:, 1] // window).astype(int)
        ok = (iy < ny) & (ix < nx)
        tot = np.bincount(iy[ok] * nx + ix[ok], weights=areas[ok], minlength=ny * nx)
        return float(tot.std() / tot.mean()) if tot.mean() > 0 else float("nan")

    extent = np.array([ny * window, nx * window])
    rand = np.array([cv(rng.uniform(0, extent, size=(len(areas), 2))) for _ in range(n_shuffles)])
    mean = np.nanmean(rand)
    if not mean > 0:
        return float("nan"), float("nan")
    return cv(centroids) / mean, float(np.nanstd(rand / mean))


def clustering_index(centroids: np.ndarray, shape: tuple[int, int], n_shuffles: int, rng) -> float:
    """Mean nearest-neighbour distance under random placement / observed (Clark-Evans style).

    1 = random, > 1 = clustered, < 1 = more regular than random.
    """
    if len(centroids) < 3:
        return float("nan")

    def mean_nn(pts: np.ndarray) -> float:
        d, _ = cKDTree(pts).query(pts, k=2)
        return float(d[:, 1].mean())

    rand = np.mean([mean_nn(rng.uniform(0, shape, size=centroids.shape)) for _ in range(n_shuffles)])
    obs = mean_nn(centroids)
    return rand / obs if obs > 0 else float("nan")


def graphite_alignment(graphite: np.ndarray, sigma: float) -> tuple[float, float, float]:
    """Orientation of graphite boundaries from the summed structure tensor of the smoothed graphite mask.

    Returns (order, coherence, mean angle in degrees). order = gradient-energy weighted <cos 2theta> of
    boundary tangents: +1 all horizontal, 0 random, -1 all vertical. coherence = strength of the dominant
    direction (0..1). Angle is counter-clockwise from horizontal as displayed. No flake splitting needed.
    """
    g = ndi.gaussian_filter(graphite[::2, ::2].astype(np.float32), max(sigma / 2, 0.5))
    gy, gx = ndi.sobel(g, 0), ndi.sobel(g, 1)
    axx, ayy, axy = float((gx * gx).sum()), float((gy * gy).sum()), float((gx * gy).sum())
    tot = axx + ayy
    if tot <= 0:
        return float("nan"), float("nan"), float("nan")
    c, s = (ayy - axx) / tot, 2 * axy / tot
    return c, float(np.hypot(c, s)), float(np.degrees(0.5 * np.arctan2(s, c)))


def crack_mask(etd_n: np.ndarray, phases: np.ndarray, k: float, ds: int = 2) -> np.ndarray:
    """Thin dark ridges (cracks) inside solid particles in ETD, at 1/ds resolution.

    Vertical polishing streaks are suppressed first by removing the column-wise low-frequency pattern.
    """
    e = etd_n[::ds, ::ds].astype(np.float32)
    e = e - (ndi.gaussian_filter1d(e, 20, axis=0) - ndi.gaussian_filter(e, 20))
    ridge = sato(e, sigmas=[1], black_ridges=True)
    interior = ndi.distance_transform_edt(phases[::ds, ::ds]) > 2
    if interior.sum() < 100:
        return np.zeros_like(interior)
    med = float(np.median(ridge[interior]))
    mad = float(np.median(np.abs(ridge[interior] - med))) * 1.4826
    return skeletonize(remove_small_objects((ridge > med + k * max(mad, 1e-6)) & interior, min_size=8))


def _rel_slope(profile: np.ndarray) -> float:
    if not np.isfinite(profile).all() or profile.mean() <= 0:
        return float("nan")
    return float(np.polyfit(np.linspace(0, 1, len(profile)), profile, 1)[0] / profile.mean())


def kpis_from_labels(
    labels: np.ndarray,
    n: np.ndarray,
    cfg: KPIConfig,
    etd_n: np.ndarray | None = None,
    crack_k: float = 6.0,
    seed: int = 0,
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    """All label-based KPIs. Also run on perturbed label maps for the pixel-ambiguity intervals."""
    rng = np.random.default_rng(seed)
    si, graphite, pore = labels == SI, labels == GRAPHITE, labels == PORE
    cbd, gap = labels == CBD, labels == GAP
    voids = pore | gap
    solids = ~voids
    total, n_solids = labels.size, max(int(solids.sum()), 1)
    scale = cfg.pixel_size_um or 1.0
    unit = "um" if cfg.pixel_size_um else "px"
    a_si, a_c = si.sum() / total, (graphite.sum() + cbd.sum()) / total

    k: dict = {
        "frac_si": a_si,
        "frac_graphite": graphite.sum() / total,
        "frac_cbd": cbd.sum() / total,
        "frac_pore": voids.sum() / total,
        "frac_gap": gap.sum() / total,
        "si_fraction_of_solids": si.sum() / n_solids,
        "cbd_fraction_of_solids": cbd.sum() / n_solids,
        "si_wt_pct_estimate": 100
        * a_si
        * cfg.density_si
        / max(a_si * cfg.density_si + a_c * cfg.density_graphite, 1e-12),
        "size_unit": unit,
    }

    parts = si_particles(si, scale)
    area_unit = (total * scale**2 / 1e6) if cfg.pixel_size_um else total / 1e6
    k["si_count"] = len(parts)
    k["si_count_per_mm2" if cfg.pixel_size_um else "si_count_per_mpx"] = len(parts) / area_unit
    if len(parts):
        ecd = parts["ecd"].to_numpy()
        k["si_ecd_d10"], k["si_ecd_d50"], k["si_ecd_d90"] = np.percentile(ecd, [10, 50, 90])
        k["si_ecd_d50_area_weighted"] = _weighted_percentile(ecd, parts["area"].to_numpy(float), 50)
        k["si_aspect_ratio_median"] = float(parts["aspect_ratio"].median())
    if len(parts) >= 2:
        pts = parts[["centroid-0", "centroid-1"]].to_numpy()
        d, _ = cKDTree(pts).query(pts, k=2)
        k["si_nn_distance_median"] = float(np.median(d[:, 1]) * scale)
        sep = ndi.distance_transform_edt(~si) <= cfg.cluster_gap_px / 2
        clusters, _ = ndi.label(sep)
        particle_labels, _ = ndi.label(si)
        cluster_of = ndi.maximum(clusters, particle_labels, index=parts["label"].to_numpy())
        sizes = pd.Series(cluster_of).value_counts()
        k["si_fraction_particles_clustered"] = float(sizes[sizes > 1].sum() / len(parts))
        k["si_clustering_index"] = clustering_index(pts, labels.shape, cfg.n_shuffles, rng)
        for w in cfg.window_sizes:
            k[f"si_dispersion_index_w{w}"], k[f"si_dispersion_random_sd_w{w}"] = dispersion_index(
                pts, parts["area"].to_numpy(float), labels.shape, w, cfg.n_shuffles, rng
            )

    # Sample neighbours a few px away: the blurred transition right at the Si edge is never pore-dark.
    dist_out = ndi.distance_transform_edt(~si)
    ring = (dist_out > cfg.contact_ring_px[0]) & (dist_out <= cfg.contact_ring_px[1]) & ~si
    if ring.any():
        for name, m in (("graphite", graphite), ("cbd", cbd), ("pore", pore), ("gap", gap)):
            k[f"si_contact_{name}"] = float(m[ring].mean())
    near = (dist_out > 0) & (dist_out <= 3)
    if near.any():
        k["si_debond_fraction"] = float(gap[near].mean())

    for w in cfg.window_sizes:
        k[f"si_cv_w{w}"] = homogeneity_cv(si, solids, w, cfg.min_window_solid_fraction)

    ones = np.ones_like(voids)
    profile = through_thickness_profile(si, solids, cfg.profile_bands)
    k["si_profile_rel_slope"] = _rel_slope(profile)
    k["porosity_profile_rel_slope"] = _rel_slope(through_thickness_profile(voids, ones, cfg.profile_bands))
    k["cbd_profile_rel_slope"] = _rel_slope(through_thickness_profile(cbd, solids, cfg.profile_bands))

    if si.any():
        core = ndi.binary_erosion(si, iterations=2)
        vals = n[core] if core.sum() >= 50 else n[si]
        q25, q50, q75 = np.percentile(vals, [25, 50, 75])
        k["si_grey_median_n"], k["si_grey_iqr_n"] = float(q50), float(q75 - q25)

    gparts = graphite_particles(graphite, cfg.graphite_min_area_px)
    if len(gparts):
        k["graphite_ecd_d50"] = float(gparts["ecd_px"].median() * scale)
        k["graphite_aspect_ratio_median"] = float(gparts["aspect_ratio"].median())
    (k["graphite_alignment"], k["graphite_alignment_coherence"], k["graphite_alignment_angle_deg"]) = (
        graphite_alignment(graphite, cfg.alignment_sigma_px)
    )

    if etd_n is not None:
        ds = 2
        cracks = crack_mask(etd_n, si | graphite, crack_k, ds)
        for name, m in (("si", si), ("graphite", graphite)):
            area = m.sum()
            k[f"{name}_crack_density"] = (
                float(cracks[m[::ds, ::ds]].sum() * ds / area * 1e4) if area else float("nan")
            )
    return k, parts, profile


def compute_kpis(
    res: SegmentationResult, cfg: KPIConfig, crack_k: float = 6.0
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    mm = res.mm
    k, parts, profile = kpis_from_labels(
        res.labels, res.n, cfg, etd_n=mm.etd_n if mm is not None else None, crack_k=crack_k
    )
    solids = ~np.isin(res.labels, (PORE, GAP))
    t, nm = res.thresholds, res.norm
    k.update(
        {
            "si_fraction_of_solids_unfiltered": float((res.si_raw & solids).sum() / max(solids.sum(), 1)),
            "multimodal": mm is not None,
            "qc_floor": nm.floor,
            "qc_graphite_mode": nm.graphite_mode,
            "qc_graphite_sigma_n": nm.graphite_sigma_n,
            "qc_threshold_method": t.method,
            "qc_pore_threshold_n": t.pore,
            "qc_si_low_n": t.si_low,
            "qc_si_high_n": t.si_high,
            "qc_si_peak_n": t.si_peak if t.si_peak is not None else np.nan,
            "qc_bright_unassigned_fraction": float(((res.n > t.si_low) & ~res.si).sum() / res.labels.size),
        }
    )
    if mm is not None:
        k.update(mm.qc)
    return k, parts, profile
