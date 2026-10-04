import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


Image.MAX_IMAGE_PIXELS = None

BATCH_PATTERN = re.compile(r"^Batch[_ ]?([123])$")
SE_PATTERN = re.compile(r"^(.*)_SE\.tiff?$", re.IGNORECASE)
EXPECTED_DETECTORS = ("BSE", "ETD", "Inlens")
MANIFEST_COLUMNS = [
    "relative_path",
    "batch_name",
    "batch_folder",
    "class_id",
    "cross_section_id",
    "group_id",
    "detector",
    "source_image_id",
    "sha256",
    "height",
    "width",
    "n_channels",
    "dtype",
    "channels_identical",
    "n_mismatched_pixels",
    "complete_set",
    "missing_detectors",
    "tiff_tags_json",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return value.item()
    except AttributeError:
        return str(value)


def _image_details(path: Path):
    with Image.open(path) as image:
        width, height = image.size
        array = np.asarray(image)
        tags = image.tag_v2
        tiff_tags = {
            str(tag): _json_value(tags[tag])
            for tag in (270, 305, 306)
            if tag in tags
        }

    if array.ndim == 2:
        n_channels = 1
        mismatch_count = 0
    elif array.ndim == 3:
        n_channels = int(array.shape[2])
        mismatch_count = int(np.any(array != array[..., :1], axis=2).sum())
    else:
        n_channels = int(array.shape[2]) if array.ndim > 2 else 1
        mismatch_count = 0

    return {
        "height": int(height),
        "width": int(width),
        "n_channels": n_channels,
        "dtype": str(array.dtype),
        "channels_identical": mismatch_count == 0,
        "n_mismatched_pixels": mismatch_count,
        "tiff_tags_json": json.dumps(tiff_tags, sort_keys=True),
    }


def _batch_directories(data_root: Path):
    for first_level in sorted(data_root.iterdir()):
        if not first_level.is_dir():
            continue
        if BATCH_PATTERN.fullmatch(first_level.name):
            yield first_level
        for second_level in sorted(first_level.iterdir()):
            if second_level.is_dir() and BATCH_PATTERN.fullmatch(second_level.name):
                yield second_level


def build_source_manifest(data_root: Path) -> pd.DataFrame:
    """Inventory detector TIFFs in batch folders at depths one and two."""
    from prepare_dataset import FILE_PATTERN

    data_root = Path(data_root).resolve()
    records = []
    for batch_dir in _batch_directories(data_root):
        batch_number = int(BATCH_PATTERN.fullmatch(batch_dir.name).group(1))
        batch_name = f"Batch {batch_number}"
        batch_folder = f"Batch_{batch_number}"
        files_by_image = {}
        candidates = []
        for path in sorted(batch_dir.iterdir()):
            if not path.is_file():
                continue
            match = FILE_PATTERN.match(path.name)
            if match:
                detector = {
                    "bse": "BSE",
                    "etd": "ETD",
                    "inlens": "Inlens",
                }[match.group(2).lower().replace("-", "")]
                image_name = match.group(1)
            else:
                match = SE_PATTERN.match(path.name)
                if not match:
                    continue
                detector = "SE"
                image_name = match.group(1)
            files_by_image.setdefault(image_name, set()).add(detector)
            candidates.append((path, image_name, detector))

        group_status = {}
        for image_name, detectors in files_by_image.items():
            missing = [detector for detector in EXPECTED_DETECTORS if detector not in detectors]
            group_status[image_name] = (not missing, ",".join(missing))

        for path, image_name, detector in candidates:
            details = _image_details(path)
            group_id = f"{batch_folder}/{image_name}"
            records.append(
                {
                    "relative_path": path.relative_to(data_root).as_posix(),
                    "batch_name": batch_name,
                    "batch_folder": batch_folder,
                    "class_id": batch_number - 1,
                    "cross_section_id": image_name,
                    "group_id": group_id,
                    "detector": detector,
                    "source_image_id": f"{group_id}/{detector}",
                    "sha256": _sha256(path),
                    **details,
                    "complete_set": group_status[image_name][0],
                    "missing_detectors": group_status[image_name][1],
                }
            )

    return pd.DataFrame(records, columns=MANIFEST_COLUMNS)


def build_external_source_manifest(data_root: Path) -> pd.DataFrame:
    """Inventory detector TIFFs in one unlabeled external folder."""
    from prepare_dataset import FILE_PATTERN

    data_root = Path(data_root).resolve()
    files_by_image = {}
    candidates = []
    for path in sorted(data_root.iterdir()):
        if not path.is_file():
            continue
        match = FILE_PATTERN.match(path.name)
        if match:
            detector = {
                "bse": "BSE",
                "etd": "ETD",
                "inlens": "Inlens",
            }[match.group(2).lower().replace("-", "")]
            image_name = match.group(1)
        else:
            match = SE_PATTERN.match(path.name)
            if not match:
                continue
            detector = "SE"
            image_name = match.group(1)
        files_by_image.setdefault(image_name, set()).add(detector)
        candidates.append((path, image_name, detector))

    group_status = {}
    for image_name, detectors in files_by_image.items():
        missing = [detector for detector in EXPECTED_DETECTORS if detector not in detectors]
        group_status[image_name] = (not missing, ",".join(missing))

    records = []
    for path, image_name, detector in candidates:
        group_id = f"External/{image_name}"
        records.append(
            {
                "relative_path": path.relative_to(data_root).as_posix(),
                "batch_name": None,
                "batch_folder": None,
                "class_id": None,
                "cross_section_id": image_name,
                "group_id": group_id,
                "detector": detector,
                "source_image_id": f"{group_id}/{detector}",
                "sha256": _sha256(path),
                **_image_details(path),
                "complete_set": group_status[image_name][0],
                "missing_detectors": group_status[image_name][1],
            }
        )
    return pd.DataFrame(records, columns=MANIFEST_COLUMNS)


def inventory_summary(df: pd.DataFrame) -> dict:
    """Summarize batch/group coverage, exclusions, and identical files at distinct paths."""
    batch_counts = {}
    if not df.empty:
        for batch_name in sorted(df["batch_name"].unique()):
            batch_df = df[df["batch_name"] == batch_name]
            groups = batch_df.drop_duplicates("group_id")
            batch_counts[batch_name] = {
                "groups": int(groups["group_id"].nunique()),
                "complete_groups": int(groups.loc[groups["complete_set"], "group_id"].nunique()),
                "images": int(batch_df["source_image_id"].nunique()),
            }

    excluded = []
    if not df.empty:
        for group_id, group in df.groupby("group_id", sort=True):
            if not bool(group["complete_set"].iloc[0]):
                excluded.append(
                    {
                        "group_id": group_id,
                        "batch_name": group["batch_name"].iloc[0],
                        "missing_detectors": group["missing_detectors"].iloc[0],
                    }
                )

    duplicates = []
    if not df.empty:
        for sha256, rows in df.groupby("sha256", sort=True):
            paths = sorted(rows["relative_path"].unique())
            if len(paths) > 1:
                duplicates.append({"sha256": sha256, "relative_paths": paths})

    return {
        "batches": batch_counts,
        "excluded_groups": excluded,
        "duplicate_hashes": duplicates,
    }
