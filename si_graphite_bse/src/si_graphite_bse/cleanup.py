"""Morphological cleanup of the raw Si mask."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.morphology import binary_opening, disk, remove_small_holes, remove_small_objects

from .config import CleanupConfig


def keep_compact_objects(mask: np.ndarray, core_radius: float) -> np.ndarray:
    """Keep only connected components that contain a pixel deeper than ``core_radius`` from the edge.

    Thin bright structures (edge brightening, porous binder/carbon-black mesh)
    never reach that depth; real Si particles keep their full extent.
    """
    if core_radius <= 0:
        return mask.copy()
    core = ndi.distance_transform_edt(mask) > core_radius
    labels, _ = ndi.label(mask)
    keep = np.unique(labels[core])
    return np.isin(labels, keep[keep > 0])


def keep_bright_objects(
    mask: np.ndarray, n: np.ndarray, seed_level: float, min_fraction: float
) -> np.ndarray:
    """Drop components where fewer than ``min_fraction`` of pixels reach the Si seed level.

    Real Si particles sit on the Si histogram peak; bright edge/mesh artefacts
    mostly hover just above the low threshold.
    """
    labels, count = ndi.label(mask)
    if count == 0 or min_fraction <= 0:
        return mask.copy()
    frac = ndi.mean(n > seed_level, labels, index=np.arange(1, count + 1))
    return np.isin(labels, np.flatnonzero(np.asarray(frac) >= min_fraction) + 1)


def keep_solid_objects(mask: np.ndarray, min_solidity: float) -> np.ndarray:
    """Drop irregular components (area / convex-hull area below ``min_solidity``).

    Faceted Si particles are compact; bright porous binder/carbon-black mesh is not.
    """
    if min_solidity <= 0:
        return mask
    labels, _ = ndi.label(mask)
    keep = [r.label for r in regionprops(labels) if r.solidity >= min_solidity]
    return np.isin(labels, keep)


def clean_si(si: np.ndarray, n: np.ndarray, seed_level: float, cfg: CleanupConfig) -> np.ndarray:
    """Remove thin rims, dim/irregular artefacts and specks from the raw Si mask; fill small internal holes."""
    mask = keep_compact_objects(si, cfg.si_core_radius_px)
    mask = keep_bright_objects(mask, n, seed_level, cfg.si_min_seed_fraction)
    if cfg.si_open_radius > 0:
        mask = binary_opening(mask, disk(cfg.si_open_radius))
    mask = remove_small_objects(mask, min_size=cfg.si_min_area_px)
    mask = remove_small_holes(mask, area_threshold=cfg.si_max_hole_area_px)
    return keep_solid_objects(mask, cfg.si_min_solidity)
