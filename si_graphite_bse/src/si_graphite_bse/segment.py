"""Three-class labelling of normalised BSE images: pore / graphite / silicon.

Si is thresholded relative to the graphite peak, never with a free global
method (Otsu / unconstrained GMM), because the Si histogram peak is small and
bright graphite edges or pore walls otherwise get labelled as Si.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks
from skimage.filters import apply_hysteresis_threshold

from .config import SegmentConfig
from .preprocess import Normalisation, smoothed_histogram

PORE, GRAPHITE, SI = 0, 1, 2
CLASS_NAMES = {PORE: "pore", GRAPHITE: "graphite", SI: "si"}

N_HIST_RANGE = (-0.5, 4.0)
N_HIST_BIN = 0.01


@dataclass(frozen=True)
class Thresholds:
    pore: float
    si_low: float
    si_high: float
    si_peak: float | None
    method: str


def compute_thresholds(n: np.ndarray, norm: Normalisation, cfg: SegmentConfig) -> Thresholds:
    """Si thresholds from the graphite/Si histogram valley, falling back to graphite mode + k*sigma.

    Hysteresis: pixels above ``si_high`` seed Si particles, which grow into
    connected pixels above ``si_low``.
    """
    s = norm.graphite_sigma_n
    centers, hist = smoothed_histogram(n.ravel()[::5], *N_HIST_RANGE, N_HIST_BIN, cfg.hist_smooth_bins)
    min_si = 1 + cfg.si_k_min * s
    peaks, _ = find_peaks(np.log1p(hist), prominence=cfg.si_peak_min_prominence)
    candidates = [p for p in peaks if centers[p] > min_si]
    pore = 1 - cfg.pore_k * s
    if not candidates:
        return Thresholds(pore, 1 + cfg.si_k_low * s, 1 + cfg.si_k_high * s, None, "sigma")
    si_peak = max(candidates, key=lambda p: hist[p])
    graphite_peak = int(np.argmin(np.abs(centers - 1.0)))
    valley = graphite_peak + int(np.argmin(hist[graphite_peak : si_peak + 1]))
    low = max(float(centers[valley]), min_si)
    high = low + cfg.si_seed_fraction * (float(centers[si_peak]) - low)
    return Thresholds(pore, low, high, float(centers[si_peak]), "valley")


def detect_si(n: np.ndarray, thr: Thresholds) -> np.ndarray:
    return apply_hysteresis_threshold(n, thr.si_low, thr.si_high)


def compose_labels(n: np.ndarray, si: np.ndarray, thr: Thresholds) -> np.ndarray:
    labels = np.full(n.shape, GRAPHITE, dtype=np.uint8)
    labels[n < thr.pore] = PORE
    labels[si] = SI
    return labels
