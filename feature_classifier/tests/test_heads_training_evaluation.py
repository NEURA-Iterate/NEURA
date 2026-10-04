from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from conftest import make_image_sample, make_sem_image

import phaseseg.train as train_module
from phaseseg import CLASSES
from phaseseg.backbone import RandomBackbone
from phaseseg.evaluate import (
    compute_phase_kpis,
    evaluate_checkpoint,
    kpi_agreement,
    sliding_window_probabilities,
)
from phaseseg.features import extract_tiled_features
from phaseseg.heads import FusionHead, LinearProbe
from phaseseg.labels import IGNORE
from phaseseg.train import (
    aligned_crop,
    class_weights,
    holdout_image_split,
    load_or_create_pseudo_labels,
    masked_cross_entropy,
    prepare_pseudo_label_cache,
    stratified_image_splits,
    train_fold,
)


def test_crop_alignment_and_flips_preserve_pixel_feature_correspondence() -> None:
    pixel_grid = np.arange(56 * 56, dtype=np.float32).reshape(56, 56)
    image = np.broadcast_to(pixel_grid[None], (3, 56, 56)).copy()
    labels = (pixel_grid.astype(np.uint8) % len(CLASSES)).copy()
    features = np.arange(4 * 4 * 2, dtype=np.float32).reshape(4, 4, 2)

    cropped_image, cropped_features, cropped_labels = aligned_crop(
        image, features, labels, top=14, left=28, crop_size=28, horizontal_flip=True, vertical_flip=True
    )
    assert cropped_image.shape == (3, 28, 28)
    assert cropped_features.shape == (2, 2, 2)
    assert cropped_image[0, 0, 0] == image[0, 41, 55]
    assert cropped_features[0, 0, 0] == features[2, 3, 0]
    assert cropped_labels[0, 0] == labels[41, 55]


def test_masked_cross_entropy_ignores_unlabelled_pixels() -> None:
    target = torch.tensor([[[0, IGNORE]]])
    logits = torch.tensor([[[[3.0, -4.0]], [[0.0, 10.0]], [[0.0, 0.0]], [[0.0, 0.0]]]])
    original = masked_cross_entropy(logits, target)
    changed = logits.clone()
    changed[:, :, :, 1] = torch.tensor([[-100.0], [100.0], [-100.0], [-100.0]])
    assert torch.allclose(original, masked_cross_entropy(changed, target))
    assert class_weights([np.array([[0, 1, 1, IGNORE]], dtype=np.uint8)]).shape == (4,)


def test_heads_return_full_resolution_and_fusion_stays_small() -> None:
    features = torch.randn(2, 384, 4, 4)
    image = torch.randn(2, 3, 56, 56)
    linear = LinearProbe()
    fusion = FusionHead()
    assert linear(features, image).shape == (2, 4, 56, 56)
    assert fusion(features, image).shape == (2, 4, 56, 56)
    assert sum(parameter.numel() for parameter in fusion.parameters()) < 500_000


def test_image_splits_are_disjoint_and_stratified_by_batch() -> None:
    image_ids = [f"b{batch}-i{index}" for batch in range(3) for index in range(5)]
    batches = [f"Batch_{batch}" for batch in range(3) for _ in range(5)]
    splits = stratified_image_splits(image_ids, batches, folds=5, seed=0)
    validation_seen = []
    for train_ids, validation_ids in splits:
        assert set(train_ids).isdisjoint(validation_ids)
        validation_seen.extend(validation_ids)
        assert len(validation_ids) == 3
        assert {image_id.split("-")[0] for image_id in validation_ids} == {"b0", "b1", "b2"}
    assert sorted(validation_seen) == sorted(image_ids)


def test_holdout_image_split_is_deterministic_and_keeps_test_images_separate() -> None:
    image_ids = [f"b{batch}-i{index}" for batch in range(3) for index in range(7)]
    batches = [f"Batch_{batch}" for batch in range(3) for _ in range(7)]
    first = holdout_image_split(image_ids, batches, seed=19)
    second = holdout_image_split(image_ids, batches, seed=19)
    batch_by_id = dict(zip(image_ids, batches, strict=True))

    assert first == second
    assert len(first["train"]) == 15
    assert len(first["val"]) == 3
    assert len(first["test"]) == 3
    assert set(first["train"]).isdisjoint(first["val"])
    assert set(first["train"]).isdisjoint(first["test"])
    assert set(first["val"]).isdisjoint(first["test"])
    assert set(first["train"] + first["val"] + first["test"]) == set(image_ids)
    for role in ("val", "test"):
        assert {batch_by_id[image_id] for image_id in first[role]} == {
            "Batch_0",
            "Batch_1",
            "Batch_2",
        }


def test_all_split_trains_without_validation_and_saves_last_epoch(tmp_path, monkeypatch) -> None:
    samples = [
        make_image_sample(2, image_id="sample-1", batch="Batch_1"),
        make_image_sample(3, image_id="sample-2", batch="Batch_2"),
        make_image_sample(4, image_id="sample-3", batch="Batch_3"),
    ]
    images = [SimpleNamespace(image_id=sample.image_id, batch=sample.batch) for sample in samples]
    monkeypatch.setattr(train_module, "find_bse_images", lambda _path: images)
    monkeypatch.setattr(train_module, "load_training_samples", lambda *_args, **_kwargs: samples)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phaseseg.train",
            "--images",
            str(tmp_path / "images"),
            "--features",
            str(tmp_path / "features"),
            "--labels",
            "pseudo",
            "--head",
            "linear",
            "--split",
            "all",
            "--epochs",
            "2",
            "--crop-size",
            "56",
            "--crops-per-image",
            "1",
            "--batch-size",
            "1",
            "--device",
            "cpu",
            "--out",
            str(tmp_path / "out"),
        ],
    )

    train_module.main()

    split = json.loads((tmp_path / "out" / "split.json").read_text())
    checkpoint = torch.load(tmp_path / "out" / "model.pt", map_location="cpu", weights_only=False)
    metrics = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert split == {"train": ["sample-1", "sample-2", "sample-3"], "val": [], "test": []}
    assert checkpoint["train_image_ids"] == split["train"]
    assert checkpoint["validation_image_ids"] == checkpoint["test_image_ids"] == []
    assert metrics["metrics"]["last_epoch"] == 2


def test_pseudo_label_cache_is_uint8_and_separated_by_settings(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = np.array([[0, 255], [2, 3]], dtype=np.uint8)
    image = SimpleNamespace(image_id="sample")
    cache_dir = prepare_pseudo_label_cache(tmp_path, ignore_boundary_px=2)
    monkeypatch.setattr(train_module, "pseudo_labels", lambda *_args: labels.copy())
    assert np.array_equal(load_or_create_pseudo_labels(image, cache_dir), labels)
    assert np.load(cache_dir / "sample.npy", allow_pickle=False).dtype == np.uint8

    def unexpected_recompute(*_args):
        raise AssertionError("expected the cached label map to be loaded")

    monkeypatch.setattr(train_module, "pseudo_labels", unexpected_recompute)
    assert np.array_equal(load_or_create_pseudo_labels(image, cache_dir), labels)
    assert prepare_pseudo_label_cache(tmp_path, ignore_boundary_px=0) != cache_dir


def test_evaluate_checkpoint_uses_test_ids_and_saves_prediction(tmp_path, monkeypatch) -> None:
    import phaseseg.evaluate as evaluate_module

    checkpoint_path = tmp_path / "model.pt"
    model = LinearProbe(in_dim=2, n_classes=4)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "head": "linear",
            "in_dim": 2,
            "n_classes": 4,
            "validation_image_ids": ["validation-image"],
        },
        checkpoint_path,
    )
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps({"train": ["train-image"], "val": ["validation-image"], "test": ["test-image"]}),
        encoding="utf-8",
    )
    images = [
        SimpleNamespace(image_id="validation-image", batch="Batch_1"),
        SimpleNamespace(image_id="test-image", batch="Batch_2"),
    ]
    labels = np.full((28, 28), CLASSES.index("si"), dtype=np.uint8)
    probabilities = np.zeros((len(CLASSES), 28, 28), dtype=np.float32)
    probabilities[CLASSES.index("si")] = 1
    monkeypatch.setattr(evaluate_module, "find_bse_images", lambda _images_dir: images)
    monkeypatch.setattr(evaluate_module, "load_triplet", lambda _image: np.zeros((3, 28, 28), dtype=np.float32))
    monkeypatch.setattr(
        evaluate_module, "load_features", lambda *_args: np.zeros((2, 2, 2), dtype=np.float32)
    )
    monkeypatch.setattr(evaluate_module, "_load_labels", lambda *_args: labels)
    monkeypatch.setattr(evaluate_module, "load_or_create_pseudo_labels", lambda *_args, **_kwargs: labels)
    monkeypatch.setattr(
        evaluate_module,
        "sliding_window_probabilities",
        lambda *_args, **_kwargs: probabilities,
    )

    frame, _ = evaluate_checkpoint(
        tmp_path,
        tmp_path,
        "pseudo",
        checkpoint_path,
        tmp_path / "evaluation",
        device="cpu",
        split_json=split_path,
        split_role="test",
        save_predictions=True,
    )

    assert frame["image_id"].tolist() == ["test-image"]
    assert (tmp_path / "evaluation" / "predictions" / "test-image.npy").is_file()


def test_sliding_window_averages_probabilities_at_pixel_resolution() -> None:
    class FixedHead(torch.nn.Module):
        def forward(self, features, image):
            logits = torch.zeros((image.shape[0], 4, *image.shape[-2:]), device=image.device)
            logits[:, 1] = 2
            return logits

    image = np.random.default_rng(3).random((3, 63, 77), dtype=np.float32)
    backbone = RandomBackbone(dim=8, patch=14)
    features = extract_tiled_features(backbone, image, tile=28, overlap=14)
    probabilities = sliding_window_probabilities(
        FixedHead(), image, features, tile_size=28, overlap=14, patch_size=14
    )
    assert probabilities.shape == (4, 63, 77)
    assert np.allclose(probabilities.sum(axis=0), 1, atol=1e-6)
    assert np.allclose(probabilities[1], probabilities[1, 0, 0])


def test_kpi_agreement_is_zero_for_identical_masks() -> None:
    _, labels = make_sem_image(2)
    agreement = kpi_agreement(labels, labels)
    assert set(agreement) == {
        "si_fraction",
        "pore_fraction",
        "si_fraction_of_solids",
        "si_ecd_d50",
        "si_ecd_d50_raw",
    }
    assert all(values["absolute_error"] == 0 for values in agreement.values())


def test_kpi_d50_filters_small_specks_and_preserves_raw_value() -> None:
    labels = np.full((128, 128), CLASSES.index("graphite"), dtype=np.uint8)
    y, x = np.ogrid[:128, :128]
    disc = (y - 64) ** 2 + (x - 64) ** 2 <= 15**2
    labels[disc] = CLASSES.index("si")
    for row in range(4, 76, 6):
        labels[row, 4] = CLASSES.index("si")
        labels[row + 1, 10:12] = CLASSES.index("si")
        labels[row + 2, 20:23] = CLASSES.index("si")

    kpis = compute_phase_kpis(labels)
    disc_ecd = 2 * np.sqrt(disc.sum() / np.pi)
    assert abs(kpis["si_ecd_d50"] - disc_ecd) < 0.1
    assert kpis["si_ecd_d50_raw"] < 5


def test_random_backbone_to_fusion_head_end_to_end_iou_above_chance() -> None:
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        train_samples = [make_image_sample(seed, f"train-{seed}", "Batch_1") for seed in range(6)]
        val_samples = [make_image_sample(seed + 20, f"val-{seed}", "Batch_2") for seed in range(2)]
        model, metrics = train_fold(
            train_samples,
            val_samples,
            head_name="fusion",
            epochs=8,
            crop_size=56,
            crops_per_image=2,
            batch_size=2,
            learning_rate=1e-3,
            device="cpu",
            seed=0,
        )
        assert isinstance(model, FusionHead)
        assert metrics["mean_iou"] > 0.6
    finally:
        torch.set_num_threads(old_threads)
