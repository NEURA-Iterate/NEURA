"""Per-image KPIs computed from the pore / graphite / Si label map."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from skimage.measure import regionprops_table
from skimage.segmentation import watershed

from .config import KPIConfig
from .pipeline import SegmentationResult
from .segment import GRAPHITE, PORE, SI

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


def _orientation_stats(orientation: np.ndarray) -> tuple[float, float]:
    """Order parameter (+1 all horizontal, -1 all vertical, 0 random) and mean axis angle from horizontal.

    skimage's ``orientation`` is measured from the row axis, so the doubled
    angle from horizontal is ``2*orientation - pi``.
    """
    c, s = -np.cos(2 * orientation).mean(), -np.sin(2 * orientation).mean()
    return float(c), float(np.degrees(0.5 * np.arctan2(s, c)))


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


def compute_kpis(res: SegmentationResult, cfg: KPIConfig) -> tuple[dict, pd.DataFrame, np.ndarray]:
    labels, n = res.labels, res.n
    si, graphite, pore = labels == SI, labels == GRAPHITE, labels == PORE
    solids = ~pore
    total, n_solids = labels.size, max(int(solids.sum()), 1)
    scale = cfg.pixel_size_um or 1.0
    unit = "um" if cfg.pixel_size_um else "px"
    a_si, a_gr = si.sum() / total, graphite.sum() / total

    k: dict = {
        "frac_si": a_si,
        "frac_graphite": a_gr,
        "frac_pore": pore.sum() / total,
        "si_fraction_of_solids": si.sum() / n_solids,
        "si_fraction_of_solids_unfiltered": (res.si_raw & solids).sum() / n_solids,
        "si_wt_pct_estimate": 100
        * a_si
        * cfg.density_si
        / max(a_si * cfg.density_si + a_gr * cfg.density_graphite, 1e-12),
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
        k["si_particles_per_cluster_mean"] = float(sizes.mean())
        k["si_fraction_particles_clustered"] = float(sizes[sizes > 1].sum() / len(parts))

    # Sample neighbours a few px away: the blurred transition right at the Si edge is never pore-dark.
    dist_out = ndi.distance_transform_edt(~si)
    ring = (dist_out > cfg.contact_ring_px[0]) & (dist_out <= cfg.contact_ring_px[1]) & ~si
    if ring.any():
        k["si_contact_graphite"] = float(graphite[ring].mean())
        k["si_contact_pore"] = float(pore[ring].mean())

    for w in cfg.window_sizes:
        k[f"si_cv_w{w}"] = homogeneity_cv(si, solids, w, cfg.min_window_solid_fraction)

    profile = through_thickness_profile(si, solids, cfg.profile_bands)
    if profile.mean() > 0:
        k["si_profile_rel_slope"] = float(
            np.polyfit(np.linspace(0, 1, len(profile)), profile, 1)[0] / profile.mean()
        )

    if si.any():
        q25, q50, q75 = np.percentile(n[si], [25, 50, 75])
        k["si_grey_median_n"], k["si_grey_iqr_n"] = float(q50), float(q75 - q25)

    gparts = graphite_particles(graphite, cfg.graphite_min_area_px)
    if len(gparts):
        k["graphite_count"] = len(gparts)
        k["graphite_ecd_d50"] = float(gparts["ecd_px"].median() * scale)
        k["graphite_aspect_ratio_median"] = float(gparts["aspect_ratio"].median())
        k["graphite_orientation_order"], k["graphite_orientation_mean_deg"] = _orientation_stats(
            gparts["orientation"].to_numpy()
        )

    t, nm = res.thresholds, res.norm
    k.update(
        {
            "qc_floor": nm.floor,
            "qc_graphite_mode": nm.graphite_mode,
            "qc_graphite_sigma_n": nm.graphite_sigma_n,
            "qc_threshold_method": t.method,
            "qc_pore_threshold_n": t.pore,
            "qc_si_low_n": t.si_low,
            "qc_si_high_n": t.si_high,
            "qc_si_peak_n": t.si_peak if t.si_peak is not None else np.nan,
            "qc_bright_unassigned_fraction": float(((n > t.si_low) & ~si).sum() / total),
        }
    )
    return k, parts, profile
