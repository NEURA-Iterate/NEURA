from __future__ import annotations

import numpy as np
from anode_qc.config import Config
from anode_qc.data import BSEImage, load_bse


def _normalise_channel(channel: np.ndarray) -> np.ndarray:
    values = channel.astype(np.float32)
    low, high = np.percentile(values, [1, 99])
    if high <= low:
        return np.zeros(values.shape, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def load_triplet(img: BSEImage, border_px: int | None = None) -> np.ndarray:
    """Load robust-normalised BSE, ETD/SE and Inlens as a float32 (3,H,W) array."""
    if border_px is None:
        border_px = Config().preprocess.border_px

    bse = load_bse(img.path, border_px=border_px)
    channels = [bse]
    for detector in ("etd", "inlens"):
        sibling = img.sibling(detector)
        channel = bse if sibling is None else load_bse(sibling, border_px=border_px)
        if channel.shape != bse.shape:
            raise ValueError(
                f"{img.image_id}: {detector} shape {channel.shape} does not match BSE {bse.shape}"
            )
        channels.append(channel)
    return np.stack([_normalise_channel(channel) for channel in channels]).astype(np.float32)
