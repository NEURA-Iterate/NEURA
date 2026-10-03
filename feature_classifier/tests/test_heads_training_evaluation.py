from __future__ import annotations

import numpy as np
import torch
from conftest import make_image_sample, make_sem_image

from phaseseg import CLASSES
from phaseseg.backbone import RandomBackbone
from phaseseg.evaluate import kpi_agreement, sliding_window_probabilities
from phaseseg.features import extract_tiled_features
from phaseseg.heads import FusionHead, LinearProbe
from phaseseg.labels import IGNORE
from phaseseg.train import (
    aligned_crop,
    class_weights,
    masked_cross_entropy,
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
    assert set(agreement) == {"si_fraction", "pore_fraction", "si_fraction_of_solids", "si_ecd_d50"}
    assert all(values["absolute_error"] == 0 for values in agreement.values())


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
