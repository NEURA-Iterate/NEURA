"""Dataset download and BSE image loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

HF_REPO_ID = "gabrielgramicelli/NEURA-iterate-hack"
BSE_SUFFIX = "_BSE.tif"


@dataclass(frozen=True)
class BSEImage:
    image_id: str
    batch: str
    path: Path


def download_dataset(dest: Path, token: str | None = None) -> Path:
    """Download only the BSE images from the private HF dataset (needs HF_TOKEN or a cached login)."""
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            local_dir=str(dest),
            allow_patterns=[f"*{BSE_SUFFIX}"],
            token=token,
        )
    )


def find_bse_images(root: Path) -> list[BSEImage]:
    images = []
    for path in sorted(Path(root).rglob(f"*{BSE_SUFFIX}")):
        batch = next((p for p in path.parts[::-1] if p.startswith("Batch_")), path.parent.name)
        images.append(BSEImage(image_id=path.name[: -len(BSE_SUFFIX)], batch=batch, path=path))
    return images


def load_bse(path: Path, border_px: int = 2) -> np.ndarray:
    """Load a BSE TIFF as a 2D uint8 array.

    Files are grey images stored as 3 identical channels; some have corrupted
    outer columns, so a small border is cropped on every side.
    """
    img = tifffile.imread(path)
    if img.ndim == 3:
        img = img[..., 0]
    if img.dtype != np.uint8:
        raise ValueError(f"{path}: expected uint8, got {img.dtype}")
    if border_px > 0:
        img = img[border_px:-border_px, border_px:-border_px]
    return np.ascontiguousarray(img)
