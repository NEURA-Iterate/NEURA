import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from normalize import black_level, graphite_mask, graphite_surface
from prepare_dataset import tile_positions


def _read_channel_zero(path):
    with Image.open(path) as image:
        array = np.asarray(image)
    return array[..., 0] if array.ndim == 3 else array


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _has_qualifying_surface_blocks(mask, block=96, min_frac=0.3):
    height, width = mask.shape
    return any(
        mask[y : y + block, x : x + block].mean() >= min_frac
        for y in range(0, height - block + 1, block)
        for x in range(0, width - block + 1, block)
    )


def fit_agnostic_scale(calibration_raw_paths: dict[str, str | Path]) -> float:
    expected = {"BSE", "ETD", "Inlens"}
    if set(calibration_raw_paths) != expected:
        raise ValueError(f"Calibration requires paths for {sorted(expected)}")
    raw_images = {
        detector: _read_channel_zero(path) for detector, path in calibration_raw_paths.items()
    }
    shape = raw_images["BSE"].shape
    if any(image.shape != shape for image in raw_images.values()):
        raise ValueError("Calibration detector images must have matching dimensions")
    mask = graphite_mask(raw_images["BSE"].astype(np.float32))
    if not mask.any():
        raise ValueError("No graphite-mask pixels found for calibration set")
    pooled = []
    for detector in ("BSE", "ETD", "Inlens"):
        raw_int = raw_images[detector]
        raw = raw_int.astype(np.float32)
        b = black_level(raw)
        surface = graphite_surface(raw - b, mask)
        anchored = (raw - b) / surface
        anchored /= np.median(anchored[mask])
        max_value = np.iinfo(raw_int.dtype).max if np.issubdtype(raw_int.dtype, np.integer) else np.inf
        stuck = raw_int >= max_value
        sample = anchored[::4, ::4]
        sample_stuck = stuck[::4, ::4]
        values = sample[~sample_stuck]
        if values.size == 0:
            raise ValueError(f"No non-stuck calibration pixels for {detector}")
        pooled.append(values.astype(np.float32, copy=False))
    percentile = float(np.percentile(np.concatenate(pooled), 99.9))
    if not np.isfinite(percentile) or percentile <= 0:
        raise ValueError(f"Invalid pooled calibration percentile: {percentile}")
    return float(1.0 / percentile)


def normalize_single_image(img_uint8_2d, scale: float, return_info: bool = False):
    raw_int = np.asarray(img_uint8_2d)
    if raw_int.ndim != 2:
        raise ValueError(f"normalize_single_image expects a 2D grayscale image, got {raw_int.shape}")
    if not np.issubdtype(raw_int.dtype, np.integer):
        raise TypeError(f"Expected an integer raw microscopy image, got {raw_int.dtype}")
    raw = raw_int.astype(np.float32)
    mask = graphite_mask(raw)
    b = black_level(raw)
    corrected = raw - b
    fallback_used = False
    if not mask.any():
        fallback_used = True
        flat_values = corrected
    elif not _has_qualifying_surface_blocks(mask):
        fallback_used = True
        flat_values = corrected[mask]
    else:
        surface = graphite_surface(corrected, mask)
        if not np.isfinite(surface).all() or np.any(surface <= 0):
            fallback_used = True
            flat_values = corrected[mask]
    if fallback_used:
        flat_surface = float(np.median(flat_values))
        if not np.isfinite(flat_surface):
            raise ValueError("Could not compute a finite detector-agnostic flat surface")
        surface = np.full(
            raw.shape,
            max(flat_surface, 1e-3),
            dtype=np.float32,
        )
    anchored = (raw - b) / surface
    reference = anchored[mask] if mask.any() else anchored
    reference_median = float(np.median(reference))
    if not np.isfinite(reference_median) or abs(reference_median) < 1e-6:
        reference_median = 1e-3
    anchored /= reference_median
    normalized = anchored * float(scale)
    stuck = raw_int >= np.iinfo(raw_int.dtype).max
    normalized[stuck] = 1.0
    normalized = np.clip(normalized, 0, 1).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("Detector-agnostic normalization produced non-finite pixels")
    info = {"fallback_used": bool(fallback_used)}
    return (normalized, info) if return_info else normalized


def tile_whole_image(norm, tile=512):
    norm = np.asarray(norm, dtype=np.float32)
    positions = tile_positions(norm.shape[0], norm.shape[1], tile)
    crops = [
        np.round(norm[y : y + tile, x : x + tile], 3).astype(np.float32)
        for y, x in positions
    ]
    return crops, positions


def _pixel_comparison(actual, expected):
    difference = np.abs(np.asarray(actual, dtype=np.float32) - np.asarray(expected, dtype=np.float32))
    return {
        "mae": float(difference.mean()),
        "max_abs_diff": float(difference.max(initial=0.0)),
        "fraction_within_0_01": float((difference <= 0.01).mean()),
        "pixels": int(difference.size),
    }


def _load_agnostic_profile(run_dir):
    path = Path(run_dir) / "profile_agnostic.json"
    return json.loads(path.read_text(encoding="utf-8"))


def parity_report(
    run_id: str,
    split: str = "test",
    prepared_root: Path | None = None,
    raw_root: Path = Path("/data/raw"),
    run_dir: Path | None = None,
    tag: str = "",
):
    run_dir = Path(run_dir or Path("/runs") / run_id)
    prepared_root = Path(prepared_root or Path("/data/prepared") / run_id)
    crop_manifest = pd.read_csv(prepared_root / split / "crop_manifest.csv")
    scale = float(_load_agnostic_profile(run_dir)["scale"])
    records = []
    for source_image_id, rows in crop_manifest.groupby("source_image_id", sort=True):
        first = rows.iloc[0]
        detector = first["detector"]
        raw_path = Path(raw_root) / Path(first["original_relative_path"])
        raw = _read_channel_zero(raw_path)
        normalized, whole_info = normalize_single_image(raw, scale, return_info=True)
        whole_crops, whole_positions = tile_whole_image(normalized, 512)
        whole_by_position = dict(zip(whole_positions, whole_crops))
        for _, row in rows.iterrows():
            position = (int(row["y"]), int(row["x"]))
            prepared = np.loadtxt(row["csv_path"], delimiter=",", dtype=np.float32)
            if position not in whole_by_position:
                raise ValueError(f"Prepared crop position {position} not found for {source_image_id}")
            whole_stats = _pixel_comparison(prepared, whole_by_position[position])
            cutout, cutout_info = normalize_single_image(
                raw[position[0] : position[0] + 512, position[1] : position[1] + 512],
                scale,
                return_info=True,
            )
            cutout = np.round(cutout, 3).astype(np.float32)
            cutout_stats = _pixel_comparison(prepared, cutout)
            records.append(
                {
                    "source_image_id": source_image_id,
                    "group_id": row["group_id"],
                    "detector": detector,
                    "y": position[0],
                    "x": position[1],
                    "whole_fallback_used": whole_info["fallback_used"],
                    "cutout_fallback_used": cutout_info["fallback_used"],
                    **{f"whole_{key}": value for key, value in whole_stats.items()},
                    **{f"cutout_{key}": value for key, value in cutout_stats.items()},
                }
            )

    frame = pd.DataFrame(records)
    by_detector = {}
    for detector, group in frame.groupby("detector", sort=True):
        by_detector[detector] = {
            method: {
                "mae": float(group[f"{method}_mae"].mean()),
                "max_abs_diff": float(group[f"{method}_max_abs_diff"].max()),
                "fraction_within_0_01": float(
                    group[f"{method}_fraction_within_0_01"].mean()
                ),
                "fallback_crop_count": int(group[f"{method}_fallback_used"].sum()),
                "fallback_crop_fraction": float(group[f"{method}_fallback_used"].mean()),
            }
            for method in ("whole", "cutout")
        }
    result = {
        "run_id": run_id,
        "split": split,
        "tag": tag,
        "variant": "DA-v1",
        "calibration_scale": scale,
        "n_crops": int(len(frame)),
        "per_detector": by_detector,
    }
    suffix = f"_{tag}" if tag else ""
    frame.to_csv(run_dir / f"raw_parity{suffix}.csv", index=False)
    (run_dir / f"raw_parity{suffix}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result
