"""Single-image segmentation pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cleanup import clean_si
from .config import Config
from .preprocess import Normalisation, denoise, estimate_normalisation, normalise
from .segment import Thresholds, compose_labels, compute_thresholds, detect_si


@dataclass
class SegmentationResult:
    raw: np.ndarray
    n: np.ndarray
    norm: Normalisation
    thresholds: Thresholds
    si_raw: np.ndarray
    si: np.ndarray
    labels: np.ndarray


def segment_image(raw: np.ndarray, cfg: Config) -> SegmentationResult:
    p = cfg.preprocess
    den = denoise(raw, p.denoise_sigma)
    norm = estimate_normalisation(den, p.floor_percentile, p.hist_smooth_levels)
    n = normalise(den, norm)
    thr = compute_thresholds(n, norm, cfg.segment)
    si_raw = detect_si(n, thr)
    si = clean_si(si_raw, n, thr.si_high, cfg.cleanup)
    labels = compose_labels(n, si, thr)
    return SegmentationResult(raw, n, norm, thr, si_raw, si, labels)
