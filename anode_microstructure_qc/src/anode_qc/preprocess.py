"""Denoising and per-image intensity normalisation.

The normalised scale maps the pore floor to 0 and the graphite peak to 1, so
thresholds expressed on it are independent of detector brightness/contrast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

HIST_BIN = 0.25


@dataclass(frozen=True)
class Normalisation:
    floor: float
    graphite_mode: float
    graphite_sigma: float

    @property
    def scale(self) -> float:
        return self.graphite_mode - self.floor

    @property
    def graphite_sigma_n(self) -> float:
        return self.graphite_sigma / self.scale


def denoise(img: np.ndarray, sigma: float) -> np.ndarray:
    img = img.astype(np.float32)
    return ndi.gaussian_filter(img, sigma) if sigma > 0 else img


def smoothed_histogram(
    values: np.ndarray, lo: float, hi: float, bin_width: float, smooth_bins: float
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(lo, hi + bin_width, bin_width)
    hist, edges = np.histogram(values, bins=edges)
    hist = ndi.gaussian_filter1d(hist.astype(np.float64), smooth_bins)
    return 0.5 * (edges[:-1] + edges[1:]), hist


def _half_width(centers: np.ndarray, hist: np.ndarray, peak: int, step: int) -> float:
    half = hist[peak] / 2
    i = peak
    while 0 < i < len(hist) - 1 and hist[i] > half:
        i += step
    return abs(centers[i] - centers[peak])


def estimate_normalisation(
    den: np.ndarray, floor_percentile: float = 0.5, hist_smooth_levels: float = 3.0
) -> Normalisation:
    """Find the pore floor and the graphite peak (mode and width) of a denoised image.

    The graphite width is the narrower half-width at half maximum of the peak,
    so neither the pore nor the Si side of the histogram can inflate it.
    """
    sample = den.ravel()[::7]
    floor = float(np.percentile(sample, floor_percentile))
    centers, hist = smoothed_histogram(sample, 0, 256, HIST_BIN, hist_smooth_levels / HIST_BIN)
    search = centers > floor + 5
    peak = int(np.flatnonzero(search)[np.argmax(hist[search])])
    hwhm = min(_half_width(centers, hist, peak, -1), _half_width(centers, hist, peak, +1))
    return Normalisation(floor=floor, graphite_mode=float(centers[peak]), graphite_sigma=hwhm / 1.1774)


def normalise(den: np.ndarray, norm: Normalisation) -> np.ndarray:
    return ((den - norm.floor) / norm.scale).astype(np.float32)
