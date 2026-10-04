import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from micronet_cls.inventory import EXPECTED_DETECTORS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_split(manifest: pd.DataFrame, data_root: Path, staged_root: Path, split: str):
    data_root = Path(data_root)
    split_root = Path(staged_root) / split
    split_root.mkdir(parents=True, exist_ok=True)
    selected = manifest[
        (manifest["split"] == split)
        & manifest["complete_set"].astype(bool)
        & manifest["detector"].isin(EXPECTED_DETECTORS)
    ]

    targets = {}
    mapping = {}
    for _, row in selected.iterrows():
        batch_name = row["batch_name"]
        detector = row["detector"]
        image_name = row["cross_section_id"]
        filename = f"{image_name}_{detector}.tif"
        source = data_root / Path(row["relative_path"])
        target = split_root / batch_name / filename
        target_key = target.relative_to(split_root).as_posix()
        previous = targets.get(target_key)
        if previous is not None and previous != row["relative_path"]:
            raise ValueError(
                f"Staging collision for {target_key}: {previous} and {row['relative_path']}"
            )
        targets[target_key] = row["relative_path"]
        if not source.is_file():
            raise FileNotFoundError(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        source_sha = row["sha256"]
        if _sha256(target) != source_sha:
            raise ValueError(f"SHA256 mismatch after staging {source} -> {target}")
        mapping[target_key] = {
            "original_relative_path": row["relative_path"],
            "sha256": source_sha,
        }

    map_path = split_root / "staging_map.json"
    temporary = map_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(map_path)
    return mapping


def calibration_set_for(manifest):
    """Use prepare_dataset's sorted file scan to choose the first staged train calibration set."""
    if isinstance(manifest, pd.DataFrame):
        directory = manifest.attrs.get("staged_train_batch3_dir")
        if directory is None:
            raise ValueError("DataFrame must define attrs['staged_train_batch3_dir']")
    else:
        directory = manifest
    from prepare_dataset import find_image_sets

    image_sets = find_image_sets(os.fspath(directory))
    if not image_sets:
        raise ValueError(f"No complete image sets found in {directory}")
    return next(iter(image_sets))
