"""Single-image segmentation pipeline: BSE classes, optionally refined with ETD / Inlens."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cleanup import clean_si
from .config import Config
from .multimodal import MultiModal, refine
from .preprocess import Normalisation, denoise, estimate_normalisation, normalise
from .segment import CBD, GAP, GRAPHITE, PORE, SI, Thresholds, compose_labels, compute_thresholds, detect_si


@dataclass
class SegmentationResult:
    raw: np.ndarray
    n: np.ndarray
    norm: Normalisation
    thresholds: Thresholds
    si_raw: np.ndarray
    si: np.ndarray
    labels: np.ndarray
    bse_labels: np.ndarray
    mm: MultiModal | None = None


def compose_multimodal_labels(si: np.ndarray, mm: MultiModal) -> np.ndarray:
    labels = np.full(si.shape, GRAPHITE, dtype=np.uint8)
    labels[mm.cbd] = CBD
    labels[mm.gap] = GAP
    labels[mm.pore] = PORE
    labels[si] = SI
    return labels


def segment_image(
    raw: np.ndarray,
    cfg: Config,
    etd: np.ndarray | None = None,
    inlens: np.ndarray | None = None,
) -> SegmentationResult:
    if raw.ndim != 2:
        raise ValueError(f"BSE must be a 2D image, got shape {raw.shape}")
    for name, detector in (("ETD/SE", etd), ("Inlens", inlens)):
        if detector is not None and detector.shape != raw.shape:
            raise ValueError(
                f"{name} shape {detector.shape} does not match BSE shape {raw.shape}; "
                "co-registered detector images are required"
            )
    p = cfg.preprocess
    den = denoise(raw, p.denoise_sigma)
    norm = estimate_normalisation(den, p.floor_percentile, p.hist_smooth_levels)
    n = normalise(den, norm)
    thr = compute_thresholds(n, norm, cfg.segment)
    si_raw = detect_si(n, thr)
    si = clean_si(si_raw, n, thr.si_high, cfg.cleanup)
    bse_labels = compose_labels(n, si, thr)
    if etd is None or not cfg.multimodal.enabled:
        return SegmentationResult(raw, n, norm, thr, si_raw, si, bse_labels, bse_labels)
    mm = refine(
        si,
        bse_labels == GRAPHITE,
        bse_labels == PORE,
        etd,
        inlens,
        raw,
        p.denoise_sigma,
        p.floor_percentile,
        cfg.multimodal,
        bse_n=n,
    )
    si = si & ~mm.si_vetoed
    labels = compose_multimodal_labels(si, mm)
    return SegmentationResult(raw, n, norm, thr, si_raw, si, labels, bse_labels, mm)
