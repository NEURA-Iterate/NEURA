from __future__ import annotations

from pathlib import Path

import numpy as np
from anode_qc.config import Config
from anode_qc.data import BSEImage, load_bse
from anode_qc.pipeline import segment_image
from anode_qc.segment import CBD, GAP, GRAPHITE, PORE, SI
from PIL import Image
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries

from . import CLASSES

IGNORE = 255
CLASS_INDEX = {name: index for index, name in enumerate(CLASSES)}
ANODE_TO_PHASE = {
    PORE: CLASS_INDEX["pore"],
    GAP: CLASS_INDEX["pore"],
    GRAPHITE: CLASS_INDEX["graphite"],
    SI: CLASS_INDEX["si"],
    CBD: CLASS_INDEX["binder"],
}


def map_anode_labels(labels: np.ndarray) -> np.ndarray:
    source = np.asarray(labels)
    if source.ndim != 2:
        raise ValueError(f"Expected a 2D anode_qc label map, got shape {source.shape}")
    unknown = set(np.unique(source).tolist()) - set(ANODE_TO_PHASE)
    if unknown:
        raise ValueError(f"Unsupported anode_qc label codes: {sorted(unknown)}")
    mapped = np.full(source.shape, IGNORE, dtype=np.uint8)
    for source_code, phase_index in ANODE_TO_PHASE.items():
        mapped[source == source_code] = phase_index
    return mapped


def pseudo_labels(
    img: BSEImage,
    cfg: Config | None = None,
    ignore_boundary_px: int = 2,
) -> np.ndarray:
    """Build rule-derived labels and ignore a configurable band around phase boundaries.

    ``class_probabilities`` provides only separate P(Si) and P(pore) maps, not a
    four-class distribution; its confidence threshold cannot consistently cover
    graphite and binder. The boundary band is the conservative fallback.
    """
    if ignore_boundary_px < 0:
        raise ValueError("ignore_boundary_px must be non-negative")
    cfg = Config() if cfg is None else cfg
    border_px = cfg.preprocess.border_px
    raw = load_bse(img.path, border_px=border_px)

    def detector_image(detector: str) -> np.ndarray | None:
        path = img.sibling(detector)
        if path is None:
            return None
        result = load_bse(path, border_px=border_px)
        if result.shape != raw.shape:
            raise ValueError(f"{img.image_id}: {detector} shape {result.shape} does not match BSE {raw.shape}")
        return result

    result = segment_image(raw, cfg, etd=detector_image("etd"), inlens=detector_image("inlens"))
    labels = map_anode_labels(result.labels)
    if ignore_boundary_px:
        boundary = find_boundaries(labels, connectivity=1, mode="thick")
        near_boundary = ndi.distance_transform_edt(~boundary) <= ignore_boundary_px
        labels[near_boundary] = IGNORE
    return labels


def load_scribbles(path: str | Path) -> np.ndarray:
    """Load a 2D PNG or NumPy label map; 255 denotes unlabelled pixels."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        labels = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".png":
        labels = np.asarray(Image.open(path).convert("L"))
    else:
        raise ValueError(f"Unsupported scribble file extension: {path.suffix}")
    if labels.ndim != 2:
        raise ValueError(f"Scribble labels must be 2D, got shape {labels.shape}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"Scribble labels must be integer encoded, got {labels.dtype}")
    allowed = set(range(len(CLASSES))) | {IGNORE}
    unknown = set(np.unique(labels).tolist()) - allowed
    if unknown:
        raise ValueError(f"Unknown scribble label values: {sorted(unknown)}")
    return labels.astype(np.uint8, copy=False)
