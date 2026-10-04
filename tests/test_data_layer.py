import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from micronet_cls.crop_manifest import (
    build_crop_manifest,
    cache_crops_npy,
    expected_crop_counts,
    pixel_area_summary,
    verify_crop_counts,
)
from micronet_cls.inventory import build_source_manifest, inventory_summary
from micronet_cls.prepare_wrapper import prepare_heldout, prepare_train
from micronet_cls.splits import assign_group_splits, check_split_integrity
from micronet_cls.staging import stage_split
from normalize import normalize_images
from prepare_dataset import find_image_sets


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_image(path: Path, seed: int, detector_index: int = 0):
    rng = np.random.default_rng(seed)
    y = np.linspace(-8, 8, 1100, dtype=np.float32)[:, None]
    x = np.linspace(-4, 4, 600, dtype=np.float32)[None, :]
    values = np.clip(
        128 + y + x + rng.normal(0, 20, size=(1100, 600)) + detector_index * 7,
        1,
        240,
    ).astype(np.uint8)
    Image.fromarray(np.repeat(values[..., None], 3, axis=2)).save(path)
    return values


def _make_dataset(root: Path, groups_per_batch: int = 4):
    arrays = {}
    for batch_num in (1, 2, 3):
        directory = (
            root / "Batch_1-archive" / "Batch_1"
            if batch_num == 1
            else root / f"Batch_{batch_num}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(groups_per_batch):
            image_name = f"img_b{batch_num}_{index:02d}"
            arrays[(batch_num, image_name)] = {}
            for detector_index, detector in enumerate(("BSE", "ETD", "Inlens")):
                path = directory / f"{image_name}_{detector}.tif"
                arrays[(batch_num, image_name)][detector] = _write_image(
                    path, batch_num * 100 + index * 10 + detector_index, detector_index
                )
    return build_source_manifest(root), arrays


def _split(manifest):
    return assign_group_splits(
        manifest,
        seed=42,
        counts={"Batch 1": (2, 1, 1), "Batch 2": (2, 1, 1), "Batch 3": (2, 1, 1)},
    )


def test_inventory_nested_labels_and_se_exclusion(tmp_path):
    root = tmp_path / "raw"
    directory = root / "Batch_1-2026" / "Batch_1"
    directory.mkdir(parents=True)
    for detector in ("BSE", "ETD", "Inlens"):
        _write_image(directory / f"img_complete_{detector}.tif", 1)
    _write_image(directory / "img_missing_BSE.tif", 2)
    _write_image(directory / "img_missing_Inlens.tif", 3)
    _write_image(directory / "img_missing_SE.tif", 4)

    manifest = build_source_manifest(root)
    complete = manifest[manifest["cross_section_id"] == "img_complete"]
    assert set(complete["batch_name"]) == {"Batch 1"}
    assert set(complete["batch_folder"]) == {"Batch_1"}
    assert set(complete["class_id"]) == {0}
    assert all(row.startswith("Batch_1-2026/Batch_1/") for row in complete["relative_path"])
    incomplete = manifest[manifest["cross_section_id"] == "img_missing"]
    assert set(incomplete["detector"]) == {"BSE", "Inlens", "SE"}
    assert not incomplete["complete_set"].any()
    assert set(incomplete["missing_detectors"]) == {"ETD"}
    assert inventory_summary(manifest)["excluded_groups"] == [
        {
            "group_id": "Batch_1/img_missing",
            "batch_name": "Batch 1",
            "missing_detectors": "ETD",
        }
    ]


def test_split_determinism_integrity_and_count_failure(tmp_path):
    manifest, _ = _make_dataset(tmp_path / "raw")
    first = _split(manifest)
    second = _split(manifest)
    pd.testing.assert_series_equal(first["split"], second["split"])
    assert check_split_integrity(first)
    for split_name in ("train", "val", "test"):
        assert set(first.loc[first["split"] == split_name, "class_id"]) == {0, 1, 2}

    with pytest.raises(ValueError, match="complete groups"):
        assign_group_splits(
            manifest,
            counts={"Batch 1": (1, 1, 1), "Batch 2": (2, 1, 1), "Batch 3": (2, 1, 1)},
        )


def test_staging_layout_hashes_and_collision(tmp_path):
    raw = tmp_path / "raw"
    manifest, _ = _make_dataset(raw)
    split_manifest = _split(manifest)
    staged = tmp_path / "staged"
    mapping = stage_split(split_manifest, raw, staged, "train")
    assert (staged / "train" / "Batch 1").is_dir()
    assert any(
        path.name.endswith("_Inlens.tif")
        for path in (staged / "train" / "Batch 1").iterdir()
    )
    for key, value in mapping.items():
        assert _hash(staged / "train" / key) == value["sha256"]

    selected = split_manifest[
        (split_manifest["split"] == "train") & (split_manifest["detector"] == "BSE")
    ]
    duplicate = selected.iloc[[0]].copy()
    alternate = split_manifest[
        (split_manifest["batch_name"] == duplicate.iloc[0]["batch_name"])
        & (split_manifest["detector"] == "BSE")
        & (split_manifest["cross_section_id"] != duplicate.iloc[0]["cross_section_id"])
    ].iloc[0]
    duplicate["relative_path"] = alternate["relative_path"]
    collision_manifest = pd.concat([split_manifest, duplicate], ignore_index=True)
    with pytest.raises(ValueError, match="Staging collision"):
        stage_split(collision_manifest, raw, tmp_path / "collision", "train")


def test_prepare_provenance_heldout_crop_validation_and_cache(tmp_path):
    raw = tmp_path / "raw"
    manifest, arrays = _make_dataset(raw)
    split_manifest = _split(manifest)
    staged_root = tmp_path / "staged"
    for split_name in ("train", "val", "test"):
        stage_split(split_manifest, raw, staged_root, split_name)

    run_dir = tmp_path / "run"
    prepared_root = tmp_path / "prepared"
    with pytest.raises(FileNotFoundError):
        prepare_heldout(tmp_path / "missing-run", staged_root, prepared_root, "val")

    train_result = prepare_train(run_dir, staged_root, prepared_root)
    staged_batch3_train = staged_root / "train" / "Batch 3"
    expected_calibration = next(iter(find_image_sets(str(staged_batch3_train))))
    provenance = json.loads((run_dir / "profile_provenance.json").read_text())
    assert train_result["calibration_image_name"] == expected_calibration
    assert provenance["calibration"]["image_name"] == expected_calibration
    profile_path = run_dir / "profile.json"
    original_profile = profile_path.read_text(encoding="utf-8")
    original_profile_sha = _hash(profile_path)
    assert json.loads(original_profile) == provenance["profile"]
    assert (
        json.loads((prepared_root / "train" / "_COMPLETE.json").read_text())["profile_sha256"]
        == original_profile_sha
    )

    profile_path.write_text('{"BSE": 9}', encoding="utf-8")
    with pytest.raises(ValueError, match="does not match profile provenance"):
        prepare_heldout(run_dir, staged_root, prepared_root, "val")
    profile_path.write_text(original_profile, encoding="utf-8")
    before = _hash(profile_path)
    prepare_heldout(run_dir, staged_root, prepared_root, "val")
    assert _hash(profile_path) == before

    train_manifest = split_manifest[split_manifest["split"] == "train"]
    crop_manifest = build_crop_manifest(
        prepared_root / "train",
        train_manifest,
        json.loads((staged_root / "train" / "staging_map.json").read_text()),
        original_profile_sha,
        "train",
    )
    expected = expected_crop_counts(train_manifest)
    assert verify_crop_counts(crop_manifest, expected) == {
        "actual": expected["total"],
        "expected": expected["total"],
    }
    assert len(crop_manifest) == 36
    assert set(crop_manifest["tile"]) == {512}
    area = pixel_area_summary(split_manifest, "train")
    assert area["retained_pixels"] == 18 * 2 * 512 * 512
    assert area["discarded_pixels"] == 18 * (1100 * 600 - 512 * 512 * 2)

    sample_row = crop_manifest.iloc[0]
    csv_values = np.loadtxt(sample_row["csv_path"], delimiter=",", dtype=np.float32)
    assert csv_values.shape == (512, 512)
    assert np.isfinite(csv_values).all()
    assert csv_values.min() >= 0 and csv_values.max() <= 1
    image_name = sample_row["cross_section_id"]
    source_row = train_manifest[
        (train_manifest["cross_section_id"] == image_name)
        & (train_manifest["detector"] == sample_row["detector"])
    ].iloc[0]
    arrays_for_set = arrays[(int(source_row["class_id"]) + 1, image_name)]
    norm, _, _ = normalize_images(arrays_for_set, json.loads(profile_path.read_text()))
    y, x = int(sample_row["y"]), int(sample_row["x"])
    expected_csv = io.StringIO()
    np.savetxt(
        expected_csv,
        norm[sample_row["detector"]][y : y + 512, x : x + 512],
        delimiter=",",
        fmt="%.3f",
    )
    expected_values = np.loadtxt(io.StringIO(expected_csv.getvalue()), delimiter=",", dtype=np.float32)
    np.testing.assert_array_equal(csv_values, expected_values)

    cache_path = prepared_root / "train" / "crops.npy"
    cache_crops_npy(crop_manifest, cache_path)
    cached = np.load(cache_path, mmap_mode="r")
    assert cached.shape == (len(crop_manifest), 512, 512)
    np.testing.assert_array_equal(cached[0], csv_values)
    assert (prepared_root / "train" / "crop_manifest.csv").is_file()
    assert len(list((prepared_root / "train" / "previews").glob("*.png"))) == 18


def test_prepare_train_refuses_existing_profile(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "profile.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare_train(run_dir, tmp_path / "staged", tmp_path / "prepared")
