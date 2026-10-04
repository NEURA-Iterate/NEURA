"""Pixel-ambiguity uncertainty: KPI ranges from per-pixel class probabilities and measured edge blur.

Root cause addressed: noise and blurred boundaries. The KPI is recomputed on four perturbed label maps
(confident pixels only, inclusive, objects shrunk by the blur width, objects grown by it) and the range
over these maps plus the nominal map is reported as ``<kpi>_pix_lo`` / ``<kpi>_pix_hi``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import disk, remove_small_objects

from .config import UncertaintyConfig
from .pipeline import SegmentationResult
from .segment import GRAPHITE, PORE, SI

KEY_KPIS: dict[int, tuple[str, list[str]]] = {
    1: ("Si fraction of solids", ["si_fraction_of_solids", "si_wt_pct_estimate"]),
    2: ("Porosity + gradient", ["frac_pore", "porosity_profile_rel_slope"]),
    3: ("Si coarse tail D90 (D50)", ["si_ecd_d90", "si_ecd_d50"]),
    4: ("Si agglomeration", ["si_dispersion_index_w512", "si_clustering_index"]),
    5: ("Cracks + Si debonding", ["si_crack_density", "graphite_crack_density", "si_debond_fraction"]),
    6: ("Binder/carbon-black + gradient", ["cbd_fraction_of_solids", "cbd_profile_rel_slope"]),
    7: ("Si material fingerprint", ["si_grey_median_n"]),
    8: ("Si top-to-bottom gradient", ["si_profile_rel_slope"]),
    9: ("Graphite alignment", ["graphite_alignment"]),
    10: ("Si contact", ["si_contact_graphite", "si_contact_cbd", "si_contact_pore", "si_contact_gap"]),
}
KEY_COLUMNS = [c for _, cols in KEY_KPIS.values() for c in cols]


def _robust(vals: np.ndarray) -> tuple[float, float]:
    med = float(np.median(vals))
    return med, max(float(np.median(np.abs(vals - med))) * 1.4826, 1e-6)


def _posterior(x: np.ndarray, a: tuple[float, float], b: tuple[float, float], prior_a: float) -> np.ndarray:
    """P(class a | x) for two Gaussian classes (mean, sd)."""
    la = -0.5 * ((x - a[0]) / a[1]) ** 2 - np.log(a[1]) + np.log(max(prior_a, 1e-6))
    lb = -0.5 * ((x - b[0]) / b[1]) ** 2 - np.log(b[1]) + np.log(max(1 - prior_a, 1e-6))
    return 1 / (1 + np.exp(np.clip(lb - la, -50, 50)))


def edge_blur_px(n: np.ndarray, si: np.ndarray, background: np.ndarray, max_d: int = 8) -> float:
    """10-90 % width of the mean brightness profile across Si edges (Si -> graphite), in px."""
    d = ndi.distance_transform_edt(~si) - ndi.distance_transform_edt(si)
    sel = (np.abs(d) <= max_d) & (si | background)
    if sel.sum() < 500:
        return float("nan")
    bins = np.round(d[sel]).astype(int)
    med = np.array(
        [np.median(n[sel][bins == b]) if (bins == b).any() else np.nan for b in range(-max_d, max_d + 1)]
    )
    inner, outer = np.nanmedian(med[:3]), np.nanmedian(med[-3:])
    if not inner > outer:
        return float("nan")
    f = (med - outer) / (inner - outer)
    xs = np.arange(-max_d, max_d + 1)
    ok = np.isfinite(f)
    # f falls from 1 (inside) to 0 (outside); interpolate the crossings on the reversed (rising) curve.
    x90 = np.interp(0.9, f[ok][::-1], xs[ok][::-1])
    x10 = np.interp(0.1, f[ok][::-1], xs[ok][::-1])
    return float(abs(x10 - x90))


def class_probabilities(res: SegmentationResult) -> tuple[np.ndarray, np.ndarray]:
    """P(Si) from BSE (Si vs graphite) and P(pore) from ETD if available, else BSE (pore vs graphite)."""
    labels, n = res.labels, res.n
    si, pore, graphite = labels == SI, labels == PORE, labels == GRAPHITE
    g_core = ndi.binary_erosion(graphite, iterations=5)
    si_core = ndi.binary_erosion(si, iterations=3)
    p_si = np.zeros(n.shape, np.float32)
    if si_core.sum() >= 50:
        p_si = _posterior(n, _robust(n[si_core]), (1.0, res.norm.graphite_sigma_n), si.mean()).astype(
            np.float32
        )
    img = res.mm.etd_n if res.mm is not None else n
    p_core = ndi.binary_erosion(pore, iterations=2)
    p_pore = np.zeros(n.shape, np.float32)
    if p_core.sum() >= 50 and g_core.sum() >= 50:
        p_pore = _posterior(img, _robust(img[p_core]), _robust(img[g_core]), pore.mean()).astype(np.float32)
    return p_si, p_pore


def perturbed_labels(
    res: SegmentationResult, delta: int, cfg: UncertaintyConfig, si_min_area_px: int = 30
) -> dict[str, np.ndarray]:
    """Label maps for the pixel-ambiguity range. The Si minimum-area rule is reapplied so that erosion
    fragments are not counted as new small particles."""
    labels = res.labels
    si, pore = labels == SI, labels == PORE
    p_si, p_pore = class_probabilities(res)
    near_si = ndi.distance_transform_edt(~si) <= delta + 2
    near_pore = ndi.distance_transform_edt(~pore) <= delta + 2
    fp = disk(max(delta, 1))
    si_d = ndi.binary_dilation(si, fp)
    variants = {
        "confident": (si & (p_si > cfg.p_confident), pore & (p_pore > cfg.p_confident)),
        "inclusive": (
            si | (near_si & (p_si > cfg.p_inclusive)),
            pore | (near_pore & (p_pore > cfg.p_inclusive)),
        ),
        "shrunk": (ndi.binary_erosion(si, fp), ndi.binary_erosion(pore, fp)),
        "grown": (si_d, ndi.binary_dilation(pore, fp) & ~si_d),
    }
    out = {}
    for name, (s, p) in variants.items():
        s = remove_small_objects(ndi.binary_fill_holes(s) & (s | si), min_size=si_min_area_px)
        lab = labels.copy()
        lab[(si & ~s) | (pore & ~p)] = GRAPHITE
        lab[p & ~s] = PORE
        lab[s] = SI
        out[name] = lab
    return out


def pixel_intervals(
    res: SegmentationResult,
    nominal: dict,
    kpi_fn: Callable[[np.ndarray], dict],
    cfg: UncertaintyConfig,
    si_min_area_px: int = 30,
    map_fn: Callable = map,
) -> dict:
    """``<kpi>_pix_lo`` / ``_pix_hi`` for the key KPIs, plus the measured edge blur.

    ``map_fn`` (e.g. ``ThreadPoolExecutor.map``) lets the perturbed KPI evaluations run concurrently."""
    labels = res.labels
    blur = edge_blur_px(res.n, labels == SI, labels == GRAPHITE, cfg.edge_profile_px)
    delta = int(np.clip(np.round(blur / 2), 1, 4)) if np.isfinite(blur) else 1
    values = {c: [nominal.get(c, np.nan)] for c in KEY_COLUMNS}
    for k in map_fn(kpi_fn, perturbed_labels(res, delta, cfg, si_min_area_px).values()):
        for c in KEY_COLUMNS:
            values[c].append(k.get(c, np.nan))
    out = {"qc_edge_blur_px": blur, "qc_pixel_delta_px": delta}
    for c, v in values.items():
        v = np.asarray(v, float)
        ok = v[np.isfinite(v)]
        out[f"{c}_pix_lo"], out[f"{c}_pix_hi"] = (
            (float(ok.min()), float(ok.max())) if ok.size else (np.nan, np.nan)
        )
    return out
