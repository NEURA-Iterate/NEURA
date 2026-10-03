from __future__ import annotations

import numpy as np
import pytest
import torch

from phaseseg.backbone import RandomBackbone
from phaseseg.evaluate import sliding_window_probabilities
from phaseseg.features import (
    expected_feature_shape,
    extract_tiled_features,
    load_features,
    save_features,
    tile_starts,
)


def test_random_backbone_pads_input_to_patch_grid() -> None:
    backbone = RandomBackbone(dim=12, patch=14)
    image = torch.rand(2, 3, 29, 41)
    features = backbone(image)
    assert features.shape == (2, 12, 3, 3)
    assert all(not parameter.requires_grad for parameter in backbone.parameters())
    assert torch.equal(features, backbone(image))


def test_tiling_extracts_complete_aligned_feature_grid() -> None:
    backbone = RandomBackbone(dim=8, patch=14)
    image = np.random.default_rng(0).random((3, 45, 57), dtype=np.float32)
    features = extract_tiled_features(backbone, image, tile=28, overlap=14)
    batched_features = extract_tiled_features(backbone, image, tile=28, overlap=14, batch_size=3)
    assert features.shape == (4, 5, 8)
    assert np.isfinite(features).all()
    assert np.allclose(features, batched_features, atol=1e-6)
    assert tile_starts(600, 518, 56) == [0, 462]
    assert expected_feature_shape((45, 57), patch_size=14) == (4, 5)


def test_wide_non_aligned_image_feature_and_prediction_coverage() -> None:
    class FixedHead(torch.nn.Module):
        def forward(self, features, image):
            logits = torch.zeros((image.shape[0], 4, *image.shape[-2:]), device=image.device)
            logits[:, 2] = 1
            return logits

    image = np.random.default_rng(4).random((3, 300, 1000), dtype=np.float32)
    backbone = RandomBackbone(dim=2, patch=14)
    features = extract_tiled_features(backbone, image, batch_size=2)
    assert features.shape == (22, 72, 2)
    probabilities = sliding_window_probabilities(FixedHead(), image, features)
    assert probabilities.shape == (4, 300, 1000)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=0), 1, atol=1e-6)


def test_load_features_npy_npz_orientation_and_grid_validation(tmp_path) -> None:
    feature_map = np.arange(4 * 5 * 8, dtype=np.float32).reshape(4, 5, 8)
    metadata = {"channels": 8, "patch_size": 14, "upsample": 1}
    save_features(tmp_path, "native", feature_map, metadata)
    assert np.load(tmp_path / "native.npy").dtype == np.float16
    loaded_native = load_features(tmp_path, "native", (56, 70))
    assert loaded_native.dtype == np.float32
    assert np.allclose(loaded_native, feature_map.astype(np.float16).astype(np.float32))

    np.savez_compressed(tmp_path / "colleague.npz", features=feature_map.transpose(2, 0, 1))
    loaded_colleague = load_features(tmp_path, "colleague", (56, 70))
    assert loaded_colleague.shape == (4, 5, 8)
    assert np.array_equal(loaded_colleague, feature_map)

    with pytest.raises(ValueError, match="feature grid does not match"):
        load_features(tmp_path, "native", (56, 56))


@pytest.mark.slow
def test_public_dinov2_small_output_shape() -> None:
    from phaseseg.backbone import Dinov2Backbone

    backbone = Dinov2Backbone()
    with torch.inference_mode():
        output = backbone(torch.rand(1, 3, 518, 518))
    assert output.shape == (1, 384, 37, 37)
