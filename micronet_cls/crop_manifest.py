import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


CROP_PATTERN = re.compile(r"^(Batch [123])-(.+)_y(\d+)_x(\d+)-(BSE|ETD|Inlens)\.csv$")
EXTERNAL_CROP_PATTERN = re.compile(r"^(.+?)-(.+)_y(\d+)_x(\d+)-(BSE|ETD|Inlens)\.csv$")
TILE_SIZE = 512
CROP_COLUMNS = [
    "csv_path",
    "batch_name",
    "cross_section_id",
    "detector",
    "group_id",
    "source_image_id",
    "class_id",
    "source_sha256",
    "original_relative_path",
    "split",
    "y",
    "x",
    "tile",
    "csv_sha256",
    "prep_version",
    "profile_sha256",
    "csv_min",
    "csv_max",
    "csv_mean",
    "csv_std",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _staging_entry(staging_map, batch_name, image_name, detector):
    key = f"{batch_name}/{image_name}_{detector}.tif"
    try:
        return staging_map[key]
    except KeyError as exc:
        raise ValueError(f"Missing staging-map entry for {key}") from exc


def _check_sample_precision(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        for value in line.split(","):
            if "." in value and len(value.rsplit(".", 1)[1]) > 3:
                raise ValueError(f"More than three decimal places in {path}")


def build_crop_manifest(
    prepared_split_dir: Path,
    split_manifest: pd.DataFrame,
    staging_map: dict | Path,
    profile_sha: str,
    split: str,
    external: bool = False,
) -> pd.DataFrame:
    prepared_split_dir = Path(prepared_split_dir)
    if isinstance(staging_map, (str, Path)):
        staging_map = json.loads(Path(staging_map).read_text(encoding="utf-8"))
    source_rows = split_manifest[
        split_manifest["complete_set"].astype(bool)
        & split_manifest["detector"].isin(("BSE", "ETD", "Inlens"))
    ]
    if not external:
        source_rows = source_rows[source_rows["split"] == split]
        source_index = {
            (row["batch_name"], row["cross_section_id"], row["detector"]): row
            for _, row in source_rows.iterrows()
        }
    else:
        source_index = {
            (row["cross_section_id"], row["detector"]): row
            for _, row in source_rows.iterrows()
        }
    prep_script = Path(__file__).resolve().parent.parent / "prepare_dataset.py"
    prep_version = _sha256(prep_script)
    records = []
    precision_checked = False

    for csv_path in sorted(prepared_split_dir.glob("*.csv")):
        pattern = EXTERNAL_CROP_PATTERN if external else CROP_PATTERN
        match = pattern.fullmatch(csv_path.name)
        if not match:
            raise ValueError(f"Unexpected crop filename: {csv_path.name}")
        folder_name, image_name, y_text, x_text, detector = match.groups()
        batch_name = None if external else folder_name
        key = (
            (image_name, detector)
            if external
            else (batch_name, image_name, detector)
        )
        if key not in source_index:
            raise ValueError(f"Crop does not match a source image in split {split}: {csv_path.name}")
        row = source_index[key]
        staged = _staging_entry(staging_map, folder_name, image_name, detector)
        if staged["sha256"] != row["sha256"]:
            raise ValueError(f"Staging-map SHA256 mismatch for {csv_path.name}")

        array = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
        if array.shape != (TILE_SIZE, TILE_SIZE):
            raise ValueError(f"Unexpected tile shape {array.shape} in {csv_path}")
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite values in {csv_path}")
        if array.min() < 0 or array.max() > 1:
            raise ValueError(f"Values outside [0, 1] in {csv_path}")
        if not precision_checked:
            _check_sample_precision(csv_path)
            precision_checked = True

        records.append(
            {
                "csv_path": str(csv_path),
                "batch_name": batch_name,
                "cross_section_id": image_name,
                "detector": detector,
                "group_id": row["group_id"],
                "source_image_id": row["source_image_id"],
                "class_id": None if external else int(row["class_id"]),
                "source_sha256": row["sha256"],
                "original_relative_path": staged["original_relative_path"],
                "split": "external" if external else split,
                "y": int(y_text),
                "x": int(x_text),
                "tile": TILE_SIZE,
                "csv_sha256": _sha256(csv_path),
                "prep_version": prep_version,
                "profile_sha256": profile_sha,
                "csv_min": float(array.min()),
                "csv_max": float(array.max()),
                "csv_mean": float(array.mean()),
                "csv_std": float(array.std()),
            }
        )
    return pd.DataFrame(records, columns=CROP_COLUMNS)


def expected_external_crop_counts(source_manifest: pd.DataFrame) -> dict:
    selected = source_manifest[
        source_manifest["complete_set"].astype(bool)
        & source_manifest["detector"].isin(("BSE", "ETD", "Inlens"))
    ]
    per_group_detector = {}
    zero_crop_groups = []
    for group_id, group in selected.groupby("group_id", sort=True):
        bse = group[group["detector"] == "BSE"]
        if bse.empty:
            raise ValueError(f"Complete external group {group_id} has no BSE image")
        height = int(bse.iloc[0]["height"])
        width = int(bse.iloc[0]["width"])
        count = (height // TILE_SIZE) * (width // TILE_SIZE)
        if count:
            per_group_detector[group_id] = {
                detector: count for detector in ("BSE", "ETD", "Inlens")
            }
        else:
            zero_crop_groups.append(group_id)
    return {
        "per_group_detector": per_group_detector,
        "zero_crop_groups": zero_crop_groups,
        "total": int(
            sum(sum(detector_counts.values()) for detector_counts in per_group_detector.values())
        ),
    }


def expected_crop_counts(split_manifest: pd.DataFrame) -> dict:
    selected = split_manifest[
        (split_manifest["split"] != "excluded")
        & split_manifest["complete_set"].astype(bool)
        & split_manifest["detector"].isin(("BSE", "ETD", "Inlens"))
    ]
    per_group_detector = {}
    for group_id, group in selected.groupby("group_id", sort=True):
        bse = group[group["detector"] == "BSE"]
        if bse.empty:
            raise ValueError(f"Complete group {group_id} has no BSE image")
        height = int(bse.iloc[0]["height"])
        width = int(bse.iloc[0]["width"])
        tiles = (height // TILE_SIZE) * (width // TILE_SIZE)
        per_group_detector[group_id] = {detector: tiles for detector in ("BSE", "ETD", "Inlens")}
    return {
        "per_group_detector": per_group_detector,
        "per_group": {
            group_id: int(sum(detectors.values()))
            for group_id, detectors in per_group_detector.items()
        },
        "total": int(sum(sum(detectors.values()) for detectors in per_group_detector.values())),
    }


def verify_crop_counts(crop_manifest: pd.DataFrame, expected: dict):
    actual = {}
    if not crop_manifest.empty:
        for (group_id, detector), count in crop_manifest.groupby(["group_id", "detector"]).size().items():
            actual.setdefault(group_id, {})[detector] = int(count)
    if actual != expected["per_group_detector"]:
        mismatches = {
            group_id: {
                "expected": expected["per_group_detector"].get(group_id, {}),
                "actual": actual.get(group_id, {}),
            }
            for group_id in sorted(set(actual) | set(expected["per_group_detector"]))
            if actual.get(group_id, {}) != expected["per_group_detector"].get(group_id, {})
        }
        raise ValueError(f"Crop counts do not match expected floor-based counts: {mismatches}")
    return {
        "actual": sum(sum(detectors.values()) for detectors in actual.values()),
        "expected": expected["total"],
    }


def pixel_area_summary(split_manifest: pd.DataFrame, split: str | None = None) -> dict:
    selected = split_manifest[
        split_manifest["complete_set"].astype(bool)
        & split_manifest["detector"].isin(("BSE", "ETD", "Inlens"))
    ]
    if split is not None:
        selected = selected[selected["split"] == split]
    records = []
    for source_image_id, group in selected.groupby("source_image_id", sort=True):
        row = group.iloc[0]
        total = int(row["height"]) * int(row["width"])
        retained = (
            int(row["height"]) // TILE_SIZE
            * TILE_SIZE
            * (int(row["width"]) // TILE_SIZE)
            * TILE_SIZE
        )
        records.append(
            {
                "source_image_id": source_image_id,
                "group_id": row["group_id"],
                "batch_name": row["batch_name"],
                "detector": row["detector"],
                "split": row["split"],
                "retained_pixels": retained,
                "discarded_pixels": total - retained,
                "total_pixels": total,
            }
        )
    return {
        "images": records,
        "retained_pixels": sum(record["retained_pixels"] for record in records),
        "discarded_pixels": sum(record["discarded_pixels"] for record in records),
        "total_pixels": sum(record["total_pixels"] for record in records),
    }


def cache_crops_npy(crop_manifest: pd.DataFrame, out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mmap = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(crop_manifest), TILE_SIZE, TILE_SIZE),
    )
    for index, row in enumerate(crop_manifest.itertuples(index=False)):
        mmap[index] = np.loadtxt(row.csv_path, delimiter=",", dtype=np.float32)
    mmap.flush()
    del mmap

    (out_path.parent / "crop_manifest.csv").write_text(
        crop_manifest.to_csv(index=False), encoding="utf-8"
    )
    train_rows = crop_manifest[crop_manifest["split"] == "train"]
    if not train_rows.empty:
        preview_dir = out_path.parent / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        for (batch_name, detector), group in train_rows.groupby(
            ["batch_name", "detector"], sort=True
        ):
            for preview_index, row in enumerate(group.head(2).itertuples(index=False), start=1):
                values = np.loadtxt(row.csv_path, delimiter=",", dtype=np.float32)
                image = np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)
                Image.fromarray(image).save(
                    preview_dir / f"{batch_name.replace(' ', '-')}-{detector}-{preview_index}.png"
                )
    return out_path
