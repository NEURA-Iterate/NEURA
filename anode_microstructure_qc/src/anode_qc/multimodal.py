"""ETD / Inlens refinement of the BSE segmentation.

The three detectors image the same field on the same pixel grid, so they are combined per pixel:

- ETD (topography): open pores are near-black and thin gaps/cracks are sharp, with grey levels consistent
  between images. ETD is normalised to the BSE graphite pixels (graphite = 1).
- Inlens (surface-sensitive): grey levels vary strongly between images (charging / settings), so only its
  edge texture is used: the binder / carbon-black domain (CBD) is a dense mesh of edges, unlike smooth
  particle interiors or isolated particle boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import binary_opening, disk, remove_small_objects

from .config import MultiModalConfig
from .preprocess import smoothed_histogram

ETD_HIST_RANGE = (-0.5, 3.0)


@dataclass
class MultiModal:
    etd_n: np.ndarray
    inlens_z: np.ndarray | None
    etd_dark: np.ndarray
    pore: np.ndarray
    gap: np.ndarray
    cbd: np.ndarray
    si_vetoed: np.ndarray
    qc: dict = field(default_factory=dict)


def robust_z(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    vals = img[ref]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826
    return (img - med) / max(mad, 1e-6)


def normalise_to_reference(img: np.ndarray, ref: np.ndarray, sigma: float, floor_pct: float) -> np.ndarray:
    """Map the darkest pixels (floor percentile) to 0 and the median of ``ref`` (graphite) to 1."""
    den = ndi.gaussian_filter(img.astype(np.float32), sigma)
    floor = float(np.percentile(den.ravel()[::7], floor_pct))
    g = float(np.median(den[ref]))
    return (den - floor) / max(g - floor, 1e-6)


def etd_pore_threshold(etd_n: np.ndarray, cfg: MultiModalConfig) -> tuple[float, str]:
    """Valley between the ETD pore peak (near 0) and the graphite peak (1)."""
    centers, hist = smoothed_histogram(etd_n.ravel()[::5], *ETD_HIST_RANGE, 0.01, 2.0)
    pore_rng = (centers > -0.2) & (centers < 0.5)
    gr_rng = (centers > 0.6) & (centers < 1.6)
    if hist[pore_rng].max() <= 0:
        return cfg.etd_pore_fallback + cfg.etd_pore_offset, "fallback"
    p = np.flatnonzero(pore_rng)[np.argmax(hist[pore_rng])]
    g = np.flatnonzero(gr_rng)[np.argmax(hist[gr_rng])]
    valley = p + int(np.argmin(hist[p : g + 1]))
    thr = float(np.clip(centers[valley], 0.2, 0.7))
    return thr + cfg.etd_pore_offset, "valley"


def detect_gaps(
    etd_fine: np.ndarray, etd_dark: np.ndarray, cfg: MultiModalConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Split ETD-dark structures into wide pores and thin gaps (<= ``gap_max_width_px``).

    Thin gaps are found as deep black top-hat features on lightly smoothed ETD (the standard smoothing
    would wash out 1-2 px gaps).
    """
    r = max(1, cfg.gap_max_width_px // 2)
    fp = disk(r)
    black_tophat = ndi.grey_closing(etd_fine, footprint=fp) - etd_fine
    wide = binary_opening(etd_dark, fp)
    gap = (((black_tophat > cfg.gap_min_depth) & (etd_fine < cfg.gap_max_level)) | etd_dark) & ~wide
    gap = remove_small_objects(gap, min_size=cfg.gap_min_area_px)
    return wide, gap


def veto_porous_si(si: np.ndarray, etd_dark: np.ndarray, max_frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Drop Si components whose filled outline contains many ETD-black pixels (porous binder mesh, not Si)."""
    filled = ndi.binary_fill_holes(si)
    lab, count = ndi.label(filled)
    if count == 0:
        return si, np.zeros_like(si)
    frac = np.asarray(ndi.mean(etd_dark, lab, index=np.arange(1, count + 1)))
    bad = np.isin(lab, np.flatnonzero(frac > max_frac) + 1)
    return si & ~bad, si & bad


def local_std(img: np.ndarray, window: int) -> np.ndarray:
    m = ndi.uniform_filter(img, window)
    return np.sqrt(np.maximum(ndi.uniform_filter(img * img, window) - m * m, 0))


def texture_ratio(img: np.ndarray, interior: np.ndarray, window: int) -> np.ndarray:
    """Local standard deviation relative to its median inside graphite interiors."""
    t = local_std(img.astype(np.float32), window)
    return t / max(float(np.median(t[interior])), 1e-6)


def detect_cbd(
    images: list[np.ndarray],
    bse_n: np.ndarray,
    interior: np.ndarray,
    rims: np.ndarray,
    si: np.ndarray,
    cfg: MultiModalConfig,
) -> np.ndarray:
    """Binder / carbon-black domain (experimental): areas of fine porous-mesh texture.

    Mesh is ~4-6x rougher than graphite interiors in every detector. Two confounders are suppressed:
    rough rims along large pores, long gaps/cracks and Si edges (a band of ``cbd_edge_band_px`` around them is ignored before
    closing), and topographic relief of graphite surfaces that only ETD/Inlens see (BSE texture must also
    be high).
    """
    w = cfg.cbd_texture_window_px
    bse_t = texture_ratio(bse_n, interior, w)
    logs = [np.log(bse_t + 1e-6)] + [np.log(texture_ratio(im, interior, w) + 1e-6) for im in images]
    score = np.exp(np.mean(logs, axis=0))
    band = ndi.distance_transform_edt(~(rims | si)) <= cfg.cbd_edge_band_px
    candidate = (score > cfg.cbd_texture_factor) & (bse_t > cfg.cbd_bse_texture_factor)
    cbd = ndi.binary_closing(candidate & ~band, disk(5))
    if cfg.cbd_open_radius > 0:
        cbd = binary_opening(cbd, disk(cfg.cbd_open_radius))
    cbd = remove_small_objects(cbd, min_size=cfg.cbd_min_area_px)
    # Mesh found outside the band may extend up to the Si / pore edge; rims without a mesh core stay out.
    cbd |= ndi.binary_dilation(cbd, disk(int(np.ceil(cfg.cbd_edge_band_px)))) & candidate
    return cbd & ~si


def streak_severity(etd_n: np.ndarray, interior: np.ndarray) -> float:
    """(Ex - Ey) / (Ex + Ey) of ETD gradients inside graphite: > 0 means vertical polishing streaks."""
    ex = float(np.abs(ndi.sobel(etd_n, 1))[interior].mean())
    ey = float(np.abs(ndi.sobel(etd_n, 0))[interior].mean())
    return (ex - ey) / max(ex + ey, 1e-12)


def refine(
    si: np.ndarray,
    graphite: np.ndarray,
    bse_pore: np.ndarray,
    etd_raw: np.ndarray,
    inlens_raw: np.ndarray | None,
    bse_raw: np.ndarray,
    sigma: float,
    floor_pct: float,
    cfg: MultiModalConfig,
    bse_n: np.ndarray | None = None,
) -> MultiModal:
    interior = ndi.distance_transform_edt(graphite) > cfg.graphite_interior_px
    if interior.sum() < 1000:
        interior = graphite
    etd_n = normalise_to_reference(etd_raw, interior, sigma, floor_pct)
    thr, method = etd_pore_threshold(etd_n, cfg)
    etd_dark = etd_n < thr
    si_kept, si_vetoed = veto_porous_si(si, etd_dark, cfg.si_max_internal_porosity)
    etd_fine = normalise_to_reference(etd_raw, interior, cfg.gap_sigma_px, floor_pct)
    pore, gap = detect_gaps(etd_fine, etd_dark, cfg)
    pore &= ~si_kept
    gap &= ~si_kept & ~pore

    inlens_z = None
    if inlens_raw is not None:
        inlens_z = robust_z(ndi.gaussian_filter(inlens_raw.astype(np.float32), 1.0), interior)
    plab, _ = ndi.label(pore)
    big = np.bincount(plab.ravel()) >= cfg.cbd_big_pore_px
    big[0] = False
    glab, _ = ndi.label(gap)
    long_gap = np.bincount(glab.ravel()) >= cfg.cbd_long_gap_px
    long_gap[0] = False
    if bse_n is None:
        bse_n = normalise_to_reference(bse_raw, interior, sigma, floor_pct)
    cbd = detect_cbd(
        [x for x in (etd_n, inlens_z) if x is not None],
        bse_n,
        interior,
        big[plab] | long_gap[glab],
        si_kept,
        cfg,
    )
    cbd &= ~pore

    union = bse_pore | etd_dark
    sat = [float((bse_raw >= cfg.saturation_level).mean())]
    if inlens_raw is not None:
        sat.append(float((inlens_raw >= cfg.saturation_level).mean()))
    si_inter = (si & si_kept).sum()
    qc = {
        "qc_etd_pore_threshold_n": thr,
        "qc_etd_pore_method": method,
        "qc_si_vetoed_fraction": float(si_vetoed.sum() / max(si.sum(), 1)),
        "qc_modal_agreement": float(2 * si_inter / max(si.sum() + si_kept.sum(), 1)),
        "porosity_detector_disagreement": float((bse_pore ^ etd_dark).sum() / max(union.sum(), 1)),
        "qc_streak_severity": streak_severity(etd_n, interior),
        "qc_charging_index": max(sat),
    }
    return MultiModal(etd_n, inlens_z, etd_dark, pore, gap, cbd, si_vetoed, qc)
