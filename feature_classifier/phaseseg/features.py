from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from anode_qc.data import find_bse_images

from .backbone import Dinov2Backbone
from .data import load_triplet

NORMALISATION = {
    "input": "per-channel 1st/99th percentile to [0,1]",
    "backbone": "ImageNet mean/std (0.485,0.456,0.406)/(0.229,0.224,0.225)",
}


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if tile < 1 or overlap < 0 or overlap >= tile:
        raise ValueError("tile must be positive and overlap must be in [0, tile)")
    stride = tile - overlap
    starts = [0]
    while starts[-1] + tile < length:
        starts.append(starts[-1] + stride)
    return starts


def extract_tiled_features(
    backbone: torch.nn.Module,
    image: np.ndarray,
    tile: int = 518,
    overlap: int = 56,
    batch_size: int = 8,
) -> np.ndarray:
    """Extract and overlap-average a full-image HWC feature grid from a CHW image."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected a triplet image shaped (3,H,W), got {image.shape}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    patch = int(getattr(backbone, "patch_size", 14))
    upsample = int(getattr(backbone, "upsample", 1))
    if tile % patch or overlap % patch:
        raise ValueError(f"tile ({tile}) and overlap ({overlap}) must be multiples of patch size {patch}")
    height, width = image.shape[-2:]
    grid_height = math.ceil(height / patch) * upsample
    grid_width = math.ceil(width / patch) * upsample
    tile_grid = (tile // patch) * upsample
    sums = np.zeros((grid_height, grid_width, int(getattr(backbone, "dim", 384))), dtype=np.float32)
    counts = np.zeros((grid_height, grid_width, 1), dtype=np.float32)
    try:
        device = next(backbone.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    pending_positions: list[tuple[int, int]] = []
    pending_tiles: list[torch.Tensor] = []

    def run_pending() -> None:
        if not pending_tiles:
            return
        batch = torch.cat(pending_tiles).to(device)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            feature_batch = backbone(batch)
        if feature_batch.shape[0] != len(pending_positions):
            raise ValueError(
                f"Backbone returned {feature_batch.shape[0]} tiles for {len(pending_positions)} inputs"
            )
        if feature_batch.shape[-2:] != (tile_grid, tile_grid):
            raise ValueError(
                f"Backbone returned feature tile {tuple(feature_batch.shape[-2:])}; "
                f"expected {(tile_grid, tile_grid)}"
            )
        for index, (top, left) in enumerate(pending_positions):
            feature = feature_batch[index].permute(1, 2, 0).float().cpu().numpy()
            feature_top = top // patch * upsample
            feature_left = left // patch * upsample
            bottom = min(feature_top + tile_grid, grid_height)
            right = min(feature_left + tile_grid, grid_width)
            usable = feature[: bottom - feature_top, : right - feature_left]
            sums[feature_top:bottom, feature_left:right] += usable
            counts[feature_top:bottom, feature_left:right] += 1.0
        pending_positions.clear()
        pending_tiles.clear()

    for top in tile_starts(height, tile, overlap):
        for left in tile_starts(width, tile, overlap):
            tile_image = image[:, top : top + tile, left : left + tile]
            pad_height, pad_width = tile - tile_image.shape[1], tile - tile_image.shape[2]
            tensor = torch.from_numpy(np.ascontiguousarray(tile_image)).unsqueeze(0).float()
            if pad_height or pad_width:
                tensor = torch.nn.functional.pad(
                    tensor,
                    (0, pad_width, 0, pad_height),
                    mode="replicate",
                )
            pending_positions.append((top, left))
            pending_tiles.append(tensor.cpu())
            if len(pending_tiles) >= batch_size:
                run_pending()
    run_pending()
    if np.any(counts == 0):
        raise RuntimeError("Feature tiling left uncovered cells in the output grid")
    return (sums / counts).astype(np.float32)


def expected_feature_shape(
    image_shape: tuple[int, int],
    patch_size: int = 14,
    upsample: int = 1,
) -> tuple[int, int]:
    return (math.ceil(image_shape[0] / patch_size) * upsample, math.ceil(image_shape[1] / patch_size) * upsample)


def feature_upsample_factor(
    image_shape: tuple[int, int],
    feature_shape: tuple[int, int],
    patch_size: int = 14,
) -> int:
    base_h, base_w = expected_feature_shape(image_shape, patch_size)
    if feature_shape[0] % base_h or feature_shape[1] % base_w:
        raise ValueError(f"Feature grid {feature_shape} is not aligned with image {image_shape}")
    up_h, up_w = feature_shape[0] // base_h, feature_shape[1] // base_w
    if up_h != up_w:
        raise ValueError(f"Feature grid has inconsistent upsample factors: {(up_h, up_w)}")
    return up_h


def _orient_features(
    features: np.ndarray,
    expected_grid: tuple[int, int],
    channels: int | None,
    source: Path,
) -> np.ndarray:
    if features.ndim != 3:
        raise ValueError(f"{source}: features must be 3D (H,W,C) or (C,H,W), got {features.shape}")
    if tuple(features.shape[:2]) == expected_grid:
        oriented = features
    elif tuple(features.shape[1:]) == expected_grid:
        oriented = features.transpose(1, 2, 0)
    else:
        raise ValueError(
            f"{source}: feature grid does not match image/patch grid {expected_grid}; got {features.shape}"
        )
    if channels is not None and oriented.shape[-1] != channels:
        raise ValueError(f"{source}: expected {channels} feature channels, got {oriented.shape[-1]}")
    return np.asarray(oriented, dtype=np.float32)


def load_features(
    directory: str | Path,
    image_id: str,
    image_shape: tuple[int, int],
    patch_size: int = 14,
    upsample: int | None = None,
) -> np.ndarray:
    """Load the native .npy cache or a colleague .npz and return an HWC float32 feature map."""
    directory = Path(directory)
    meta_path = directory / "meta.json"
    metadata: dict[str, Any] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    patch_size = int(metadata.get("patch_size", patch_size))
    upsample = int(metadata.get("upsample", 1) if upsample is None else upsample)
    channels = metadata.get("channels")
    expected_grid = expected_feature_shape(image_shape, patch_size, upsample)

    npy_path = directory / f"{image_id}.npy"
    npz_path = directory / f"{image_id}.npz"
    if npy_path.exists():
        features = np.load(npy_path, allow_pickle=False)
        source = npy_path
    elif npz_path.exists():
        source = npz_path
        with np.load(npz_path, allow_pickle=False) as archive:
            if "features" not in archive:
                raise ValueError(f"{npz_path}: expected an array named 'features'")
            features = archive["features"]
    else:
        raise FileNotFoundError(f"No feature file for {image_id!r} in {directory}")
    return _orient_features(features, expected_grid, int(channels) if channels else None, source)


def save_features(
    directory: str | Path,
    image_id: str,
    features: np.ndarray,
    metadata: dict[str, Any],
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{image_id}.npy"
    np.save(path, np.asarray(features, dtype=np.float16), allow_pickle=False)
    (directory / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def _metadata(
    model_id: str,
    backbone: Dinov2Backbone,
    tile: int,
    overlap: int,
    batch_size: int,
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "upsample": backbone.upsample,
        "tile": tile,
        "overlap": overlap,
        "batch_size": batch_size,
        "channels": backbone.dim,
        "patch_size": backbone.patch_size,
        "normalisation": NORMALISATION,
        "feature_layout": "HWC float16",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract tiled frozen DINOv2 features for BSE image triplets.")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-id", default="facebook/dinov2-small")
    parser.add_argument("--upsample", type=int, default=1)
    parser.add_argument("--tile", type=int, default=518)
    parser.add_argument("--overlap", type=int, default=56)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    backbone = Dinov2Backbone(args.model_id, upsample=args.upsample, device=args.device)
    metadata = _metadata(args.model_id, backbone, args.tile, args.overlap, args.batch_size)
    args.out.mkdir(parents=True, exist_ok=True)
    for image in find_bse_images(args.images):
        image_started = time.perf_counter()
        triplet = load_triplet(image)
        features = extract_tiled_features(
            backbone,
            triplet,
            args.tile,
            args.overlap,
            batch_size=args.batch_size,
        )
        save_features(args.out, image.image_id, features, metadata)
        elapsed = time.perf_counter() - image_started
        print(f"{image.image_id}: {features.shape} elapsed={elapsed:.2f}s")


if __name__ == "__main__":
    main()
