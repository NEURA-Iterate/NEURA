"""Interpretable, unit-bearing KPIs for one three-view SEM sample.

Columns follow ``kpi__<view>__<name>__<unit>`` (material KPIs) and
``qa__<view>__<name>__<unit>`` (imaging-QA descriptors that should *not*
differ between batches if acquisition was consistent). Units are px because
no pixel size survives in the TIFF metadata.

Phase segmentation on BSE: 3-class multi-Otsu on a smoothed image gives
pore (dark), graphite (mid-grey) and bright-phase (light) masks.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage import filters, measure, morphology

VIEWS = ["BSE", "SE", "InLens"]
DOWNSAMPLE = 2
MIN_OBJECT_PX = 16  # at downsampled resolution


def load_gray(path: str, downsample: int = DOWNSAMPLE) -> np.ndarray:
    a = tifffile.imread(path)
    if a.ndim == 3:
        a = a[..., 0]
    a = a.astype(np.float32)
    if downsample > 1:
        h, w = (a.shape[0] // downsample) * downsample, (a.shape[1] // downsample) * downsample
        a = a[:h, :w].reshape(h // downsample, downsample, w // downsample, downsample).mean(axis=(1, 3))
    return a


def robust_normalise(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [1, 99.5])
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)


def qa_features(img: np.ndarray, view: str, scale: int) -> dict:
    noise = np.median(np.abs(img - ndi.median_filter(img, size=3))) * 1.4826
    return {
        f"qa__{view}__height__px": img.shape[0] * scale,
        f"qa__{view}__width__px": img.shape[1] * scale,
        f"qa__{view}__mean_grey__8bit": float(img.mean()),
        f"qa__{view}__std_grey__8bit": float(img.std()),
        f"qa__{view}__noise_sigma__8bit": float(noise),
        f"qa__{view}__p99_grey__8bit": float(np.percentile(img, 99)),
    }


def size_stats(mask: np.ndarray, scale: int) -> tuple[float, float, float, float]:
    """(number-median eq. diameter, area-weighted d50 eq. diameter, count per Mpx, median elongation)."""
    lab = measure.label(mask, connectivity=1)
    props = measure.regionprops_table(lab, properties=("area", "major_axis_length", "minor_axis_length"))
    area = props["area"].astype(float)
    keep = area >= MIN_OBJECT_PX
    if keep.sum() == 0:
        return np.nan, np.nan, 0.0, np.nan
    area = area[keep]
    eq_d = np.sqrt(4 * area / np.pi) * scale
    order = np.argsort(eq_d)
    cum = np.cumsum(area[order]) / area.sum()
    d50_area = float(eq_d[order][np.searchsorted(cum, 0.5)])
    minor = np.maximum(props["minor_axis_length"][keep], 1e-6)
    elong = props["major_axis_length"][keep] / minor
    mpx = mask.size * scale * scale / 1e6
    return float(np.median(eq_d)), d50_area, float(len(area) / mpx), float(np.median(elong))


def chord_lengths(mask: np.ndarray, axis: int, scale: int) -> float:
    """Median run length of True pixels along ``axis`` (0 = vertical chords, 1 = horizontal)."""
    m = mask if axis == 1 else mask.T
    padded = np.pad(m, ((0, 0), (1, 1)))
    d = np.diff(padded.astype(np.int8), axis=1)
    starts = np.argwhere(d == 1)
    ends = np.argwhere(d == -1)
    runs = ends[:, 1] - starts[:, 1]
    runs = runs[runs >= 2]
    return float(np.median(runs) * scale) if len(runs) else np.nan


def orientation_anisotropy(img: np.ndarray, sigma: float = 4.0) -> tuple[float, float]:
    """Structure-tensor coherence (0 isotropic .. 1 aligned) and dominant angle in degrees from horizontal."""
    gy, gx = np.gradient(filters.gaussian(img, 1.0))
    jxx = filters.gaussian(gx * gx, sigma).mean()
    jyy = filters.gaussian(gy * gy, sigma).mean()
    jxy = filters.gaussian(gx * gy, sigma).mean()
    tr = jxx + jyy
    disc = np.sqrt((jxx - jyy) ** 2 + 4 * jxy**2)
    coherence = disc / max(tr, 1e-12)
    angle = 0.5 * np.degrees(np.arctan2(2 * jxy, jxx - jyy))
    return float(coherence), float(abs(angle))


def bse_kpis(img: np.ndarray, scale: int) -> dict:
    v = "BSE"
    smooth = filters.gaussian(img, 2.0, preserve_range=True)
    t_lo, t_hi = filters.threshold_multiotsu(smooth, classes=3)
    pore = smooth < t_lo
    bright = smooth > t_hi
    solid = ~pore
    pore = morphology.remove_small_objects(pore, MIN_OBJECT_PX)
    bright = morphology.binary_opening(bright, morphology.disk(3))  # drop thin edge halos
    bright = morphology.remove_small_objects(bright, MIN_OBJECT_PX)
    p_med, p_d50, p_cnt, p_elong = size_stats(pore, scale)
    b_med, b_d50, b_cnt, b_elong = size_stats(bright, scale)
    coh, ang = orientation_anisotropy(smooth)
    h_chord = chord_lengths(solid, 1, scale)
    v_chord = chord_lengths(solid, 0, scale)
    return {
        f"kpi__{v}__pore_fraction__frac": float(pore.mean()),
        f"kpi__{v}__bright_phase_fraction__frac": float(bright.mean()),
        f"kpi__{v}__pore_eq_diameter_d50__px": p_d50,
        f"kpi__{v}__pore_count__per_Mpx": p_cnt,
        f"kpi__{v}__pore_elongation__ratio": p_elong,
        f"kpi__{v}__bright_particle_eq_diameter_d50__px": b_d50,
        f"kpi__{v}__bright_particle_count__per_Mpx": b_cnt,
        f"kpi__{v}__solid_chord_horizontal__px": h_chord,
        f"kpi__{v}__solid_chord_vertical__px": v_chord,
        f"kpi__{v}__solid_chord_anisotropy__ratio": h_chord / v_chord if v_chord else np.nan,
        f"kpi__{v}__orientation_coherence__frac": coh,
        f"kpi__{v}__dominant_angle__deg": ang,
        f"kpi__{v}__phase_contrast_bright_vs_graphite__8bit": float(t_hi - t_lo),
    }


def se_kpis(img: np.ndarray, scale: int) -> dict:
    v = "SE"
    n = robust_normalise(filters.gaussian(img, 1.0, preserve_range=True))
    t = filters.threshold_otsu(n)
    dark = n < t
    dark = morphology.remove_small_objects(dark, MIN_OBJECT_PX)
    d_med, d_d50, d_cnt, d_elong = size_stats(dark, scale)
    edges = filters.sobel(n)
    edge_t = filters.threshold_otsu(edges)
    local_var = ndi.uniform_filter(n**2, 15) - ndi.uniform_filter(n, 15) ** 2
    coh, ang = orientation_anisotropy(n)
    return {
        f"kpi__{v}__dark_fraction__frac": float(dark.mean()),
        f"kpi__{v}__dark_region_eq_diameter_d50__px": d_d50,
        f"kpi__{v}__dark_region_count__per_Mpx": d_cnt,
        f"kpi__{v}__edge_pixel_fraction__frac": float((edges > edge_t).mean()),
        f"kpi__{v}__edge_density__per_px": float(edges.mean() / scale),
        f"kpi__{v}__local_roughness__frac": float(np.sqrt(np.clip(local_var, 0, None)).mean()),
        f"kpi__{v}__orientation_coherence__frac": coh,
    }


def inlens_kpis(img: np.ndarray, scale: int) -> dict:
    v = "InLens"
    n = robust_normalise(filters.gaussian(img, 1.0, preserve_range=True))
    t_lo, t_hi = filters.threshold_multiotsu(n, classes=3)
    bright = morphology.remove_small_objects(n > t_hi, MIN_OBJECT_PX)
    dark = morphology.remove_small_objects(n < t_lo, MIN_OBJECT_PX)
    b_med, b_d50, b_cnt, b_elong = size_stats(bright, scale)
    skel = morphology.skeletonize(bright)
    edges = filters.sobel(n)
    coh, ang = orientation_anisotropy(n)
    return {
        f"kpi__{v}__bright_edge_fraction__frac": float(bright.mean()),
        f"kpi__{v}__bright_edge_skeleton_length__px_per_Mpx": float(skel.sum() * scale / (n.size * scale * scale / 1e6)),
        f"kpi__{v}__bright_feature_elongation__ratio": b_elong,
        f"kpi__{v}__dark_fraction__frac": float(dark.mean()),
        f"kpi__{v}__edge_density__per_px": float(edges.mean() / scale),
        f"kpi__{v}__orientation_coherence__frac": coh,
    }


VIEW_FUNCS = {"BSE": bse_kpis, "SE": se_kpis, "InLens": inlens_kpis}


def extract_sample(row: dict, downsample: int = DOWNSAMPLE) -> dict:
    out = {"sample_id": row["sample_id"], "batch_id": row.get("batch_id", "")}
    for view in VIEWS:
        path = row.get(f"view_{view}_path")
        if not isinstance(path, str) or not path:
            continue
        img = load_gray(path, downsample)
        out.update(qa_features(img, view, downsample))
        out.update(VIEW_FUNCS[view](img, downsample))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", default="features_real.csv")
    p.add_argument("--downsample", type=int, default=DOWNSAMPLE)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    manifest = pd.read_csv(args.manifest, dtype={"batch_id": str}).fillna({"batch_id": ""})
    rows = manifest.to_dict("records")
    with ProcessPoolExecutor(args.workers) as ex:
        feats = list(ex.map(extract_sample, rows, [args.downsample] * len(rows)))
    df = pd.DataFrame(feats)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} samples x {len(df.columns) - 2} features to {args.out}")


if __name__ == "__main__":
    main()
