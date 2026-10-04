import io

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import TensorDataset
from torchvision import models

import micronet_cls.predict as predict_module
from micronet_cls.data import (
    SpatialAug,
    TwoViewDataset,
    balanced_sampler_weights,
    detector_indices,
)
from micronet_cls.evaluate import (
    aggregate_image_predictions,
    aggregate_predictions,
    bootstrap_group_metrics,
)
from micronet_cls.crop_manifest import EXTERNAL_CROP_PATTERN
from micronet_cls.model import InputPrep, simsiam_loss
from micronet_cls.predict import predict, predict_csv_with_model
from micronet_cls.raw_preprocess import (
    fit_agnostic_scale,
    normalize_single_image,
    tile_whole_image,
)
from micronet_cls.stage_b import verify_resume_checkpoint
from micronet_cls.training_utils import config_sha256
from micronet_cls.weights import load_micronet_encoder


def _fake_state_dict():
    torch.manual_seed(7)
    model = models.resnet50(weights=None)
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def test_micronet_weight_loader_fc_variants_and_extra_key(tmp_path):
    state = _fake_state_dict()
    with_fc = tmp_path / "with_fc.pth"
    without_fc = tmp_path / "without_fc.pth"
    invalid = tmp_path / "extra_key.pth"
    torch.save({"state_dict": state}, with_fc)
    torch.save({key: value for key, value in state.items() if not key.startswith("fc.")}, without_fc)
    extra = dict(state)
    extra["unexpected.weight"] = torch.ones(1)
    torch.save(extra, invalid)

    for path in (with_fc, without_fc):
        encoder, report = load_micronet_encoder(path)
        assert report["missing_keys"] == []
        assert report["unexpected_keys"] == []
        assert report["tensors_loaded"] == len(encoder.state_dict())
        assert report["verified_tensor_count"] >= 2
        assert all(report["verified_tensors"].values())
    encoder, report = load_micronet_encoder(with_fc)
    assert report["discarded_fc_keys"] == ["fc.bias", "fc.weight"]

    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_micronet_encoder(invalid)


def test_input_prep_repeats_grayscale_and_imagenet_normalizes():
    torch.manual_seed(2)
    crop = torch.rand(3, 512, 512)
    prep = InputPrep("none")
    output = prep(crop)
    assert output.shape == (3, 3, 512, 512)
    assert torch.equal(output[:, 0], output[:, 1])
    assert torch.equal(output[:, 1], output[:, 2])
    imagenet = InputPrep("imagenet")(crop.unsqueeze(1))
    assert imagenet.shape == (3, 3, 512, 512)
    assert not torch.equal(imagenet[:, 0], imagenet[:, 1])


def test_simsiam_loss_is_symmetric_and_detaches_targets():
    torch.manual_seed(3)
    p1 = torch.randn(5, 12, requires_grad=True)
    z1 = torch.randn(5, 12, requires_grad=True)
    p2 = torch.randn(5, 12, requires_grad=True)
    z2 = torch.randn(5, 12, requires_grad=True)
    loss = simsiam_loss(p1, z1, p2, z2)
    swapped = simsiam_loss(p2, z2, p1, z1)
    torch.testing.assert_close(loss, swapped)
    loss.backward()
    assert p1.grad is not None and p2.grad is not None
    assert z1.grad is None and z2.grad is None


def test_spatial_augmentation_and_two_views_keep_single_channel():
    image = torch.linspace(0, 1, 512 * 512).reshape(1, 512, 512)
    aug = SpatialAug(
        flips=True,
        rot90=True,
        rrc_scale=(0.5, 1.0),
        brightness_contrast=(0.1, 0.1),
    )
    first = aug(image)
    second = aug(image)
    assert first.shape == (1, 512, 512)
    assert second.shape == first.shape
    assert torch.isfinite(first).all()
    assert first.min() >= 0 and first.max() <= 1
    base = TensorDataset(
        image.unsqueeze(0),
        torch.tensor([1]),
        torch.tensor([9]),
    )
    view_a, view_b, class_id, row_index = TwoViewDataset(base, aug)[0]
    assert view_a.shape == view_b.shape == (1, 512, 512)
    assert class_id == 1 and row_index == 9


def test_balanced_sampler_class_and_detector_marginals():
    rows = []
    for class_id in (0, 1, 2):
        detector_counts = {"BSE": 2, "ETD": 3, "Inlens": 1}
        for detector, image_count in detector_counts.items():
            for image_number in range(image_count):
                crop_count = (image_number + 1) * 2
                for _ in range(crop_count):
                    rows.append(
                        {
                            "class_id": class_id,
                            "detector": detector,
                            "source_image_id": f"{class_id}-{detector}-{image_number}",
                        }
                    )
    manifest = pd.DataFrame(rows)
    weights = balanced_sampler_weights(manifest).numpy()
    class_mass = {
        class_id: weights[manifest["class_id"].to_numpy() == class_id].sum()
        for class_id in (0, 1, 2)
    }
    assert class_mass == pytest.approx({0: 1 / 3, 1: 1 / 3, 2: 1 / 3})
    for class_id in (0, 1, 2):
        detector_mass = {
            detector: weights[
                (manifest["class_id"].to_numpy() == class_id)
                & (manifest["detector"].to_numpy() == detector)
            ].sum()
            for detector in ("BSE", "ETD", "Inlens")
        }
        assert list(detector_mass.values()) == pytest.approx([1 / 9] * 3)


def test_inlens_filter_preserves_balanced_class_sampler_marginals():
    rows = []
    for class_id in (0, 1, 2):
        for detector in ("BSE", "ETD", "Inlens"):
            for crop_index in range((class_id + 1) * (1 + len(detector))):
                rows.append(
                    {
                        "class_id": class_id,
                        "detector": detector,
                        "source_image_id": f"{class_id}-{detector}-image",
                    }
                )
    manifest = pd.DataFrame(rows)
    selected_indices = detector_indices(manifest, ("Inlens",))
    selected = manifest.iloc[selected_indices].reset_index(drop=True)
    assert set(selected["detector"]) == {"Inlens"}

    weights = balanced_sampler_weights(selected).numpy()
    marginals = [
        weights[selected["class_id"].to_numpy() == class_id].sum()
        for class_id in (0, 1, 2)
    ]
    assert marginals == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_external_crop_filename_pattern():
    match = EXTERNAL_CROP_PATTERN.fullmatch(
        "External-img_0eryguqq_y512_x1024-Inlens.csv"
    )
    assert match is not None
    assert match.groups() == ("External", "img_0eryguqq", "512", "1024", "Inlens")


def test_image_group_aggregation_and_group_bootstrap():
    manifest = pd.DataFrame(
        [
            {
                "source_image_id": "g1-BSE",
                "group_id": "g1",
                "detector": "BSE",
                "class_id": 0,
                "batch_name": "Batch 1",
                "y": 0,
                "x": 0,
            },
            {
                "source_image_id": "g1-BSE",
                "group_id": "g1",
                "detector": "BSE",
                "class_id": 0,
                "batch_name": "Batch 1",
                "y": 0,
                "x": 512,
            },
            {
                "source_image_id": "g1-ETD",
                "group_id": "g1",
                "detector": "ETD",
                "class_id": 0,
                "batch_name": "Batch 1",
                "y": 0,
                "x": 0,
            },
            {
                "source_image_id": "g2-BSE",
                "group_id": "g2",
                "detector": "BSE",
                "class_id": 1,
                "batch_name": "Batch 2",
                "y": 0,
                "x": 0,
            },
            {
                "source_image_id": "g3-Inlens",
                "group_id": "g3",
                "detector": "Inlens",
                "class_id": 2,
                "batch_name": "Batch 3",
                "y": 0,
                "x": 0,
            },
        ]
    )
    probabilities = [
        [0.8, 0.1, 0.1],
        [0.6, 0.2, 0.2],
        [0.4, 0.4, 0.2],
        [0.1, 0.8, 0.1],
        [0.1, 0.2, 0.7],
    ]
    crops = [
        {"row_index": index, "probabilities": np.asarray(probability)}
        for index, probability in enumerate(probabilities)
    ]
    crop_predictions = aggregate_predictions(crops, manifest)
    images = aggregate_image_predictions(crop_predictions)
    image = images[images["source_image_id"] == "g1-BSE"].iloc[0]
    assert image["p0"] == pytest.approx(0.7)
    assert image["n_crops"] == 2
    assert image["image_loss"] == pytest.approx((-np.log(0.8) - np.log(0.6)) / 2)
    bootstrap = bootstrap_group_metrics(images, iterations=40, seed=42)
    assert all(0 <= value <= 1 for value in bootstrap["accuracy_95_ci"])
    assert all(0 <= value <= 1 for value in bootstrap["macro_f1_95_ci"])
    assert bootstrap["groups_per_batch"] == {"Batch 1": 1, "Batch 2": 1, "Batch 3": 1}


def test_stage_b_resume_accepts_legacy_default_tag_and_rejects_changes():
    config = {"init": "stage_a", "warmup_epochs": 5, "tag": ""}
    legacy_config = {key: value for key, value in config.items() if key != "tag"}
    checkpoint = {
        "epoch": 3,
        "config_sha256": config_sha256(legacy_config),
        "split_sha256": "split",
        "profile_sha256": "profile",
    }
    assert verify_resume_checkpoint(checkpoint, config, "split", "profile") == 4
    changed = dict(config, warmup_epochs=6)
    with pytest.raises(ValueError, match="config hash"):
        verify_resume_checkpoint(checkpoint, changed, "split", "profile")


class _FixedLogits(torch.nn.Module):
    def forward(self, images):
        mean = images.mean(dim=(1, 2, 3))
        return torch.stack((mean, mean + 0.2, 1 - mean), dim=1)


def test_csv_prediction_ignores_filename_and_raw_small_image_errors(tmp_path, monkeypatch):
    class _NoopPrep(torch.nn.Module):
        def forward(self, images):
            return images

    monkeypatch.setattr(
        predict_module,
        "load_classifier_checkpoint",
        lambda *_args, **_kwargs: (
            _FixedLogits(),
            _NoopPrep(),
            {},
            torch.device("cpu"),
        ),
    )
    values = np.full((512, 512), 0.25, dtype=np.float32)
    first = tmp_path / "Batch 1-img_bse_0_0-BSE.csv"
    second = tmp_path / "q_000.csv"
    np.savetxt(first, values, delimiter=",", fmt="%.3f")
    np.savetxt(second, values, delimiter=",", fmt="%.3f")
    assert predict("unused.pt", first, mode="csv") == predict(
        "unused.pt", second, mode="csv"
    )

    small = tmp_path / "query.png"
    Image.fromarray(np.full((100, 100), 128, dtype=np.uint8)).save(small)
    with pytest.raises(ValueError, match="yields zero crops under the edge policy"):
        predict("unused.pt", small, mode="raw", agnostic_scale=0.5)


def _raw_array(seed):
    rng = np.random.default_rng(seed)
    y = np.linspace(-8, 8, 1100, dtype=np.float32)[:, None]
    x = np.linspace(-4, 4, 600, dtype=np.float32)[None, :]
    return np.clip(128 + y + x + rng.normal(0, 20, (1100, 600)), 1, 240).astype(np.uint8)


def test_detector_agnostic_scale_normalization_and_tiles(tmp_path):
    paths = {}
    for index, detector in enumerate(("BSE", "ETD", "Inlens")):
        array = _raw_array(120 + index)
        path = tmp_path / f"calibration_{detector}.tif"
        Image.fromarray(np.repeat(array[..., None], 3, axis=2)).save(path)
        paths[detector] = path
    scale = fit_agnostic_scale(paths)
    assert np.isfinite(scale) and scale > 0

    image = _raw_array(123)
    normalized = normalize_single_image(image, scale)
    assert normalized.dtype == np.float32
    assert normalized.shape == image.shape
    assert np.isfinite(normalized).all()
    assert normalized.min() >= 0 and normalized.max() <= 1
    crops, positions = tile_whole_image(normalized, 512)
    assert len(crops) == (image.shape[0] // 512) * (image.shape[1] // 512) == 2
    assert positions == [(0, 0), (512, 0)]
    assert all(crop.shape == (512, 512) for crop in crops)


def test_single_image_surface_fallback_for_512_constant_cutout():
    image = np.full((512, 512), 128, dtype=np.uint8)
    normalized, info = normalize_single_image(image, 0.5, return_info=True)
    assert info["fallback_used"] is True
    assert normalized.shape == image.shape
    assert np.isfinite(normalized).all()
    assert normalized.min() >= 0 and normalized.max() <= 1
