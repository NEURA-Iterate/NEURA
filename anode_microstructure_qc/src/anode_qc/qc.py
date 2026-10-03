"""Quality-control images: overlays and annotated histograms."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .pipeline import SegmentationResult
from .preprocess import smoothed_histogram
from .segment import CBD, GAP, N_HIST_BIN, N_HIST_RANGE, PORE, SI

SI_COLOUR = np.array([255, 140, 0], dtype=np.float32)
PORE_COLOUR = np.array([30, 60, 255], dtype=np.float32)
CBD_COLOUR = np.array([40, 200, 40], dtype=np.float32)
GAP_COLOUR = np.array([255, 0, 255], dtype=np.float32)
CLASS_COLOURS = ((SI, SI_COLOUR), (PORE, PORE_COLOUR), (CBD, CBD_COLOUR), (GAP, GAP_COLOUR))


def overlay(raw: np.ndarray, labels: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    rgb = np.repeat(raw[..., None], 3, axis=2).astype(np.float32)
    for cls, colour in CLASS_COLOURS:
        m = labels == cls
        rgb[m] = (1 - alpha) * rgb[m] + alpha * colour
    return rgb.astype(np.uint8)


def densest_si_crop(si: np.ndarray, size: tuple[int, int] = (800, 1200)) -> tuple[int, int]:
    """Top-left corner of the crop with the most Si (coarse grid search)."""
    h, w = si.shape
    ch, cw = min(size[0], h), min(size[1], w)
    best, best_yx = -1, (0, 0)
    for y in range(0, h - ch + 1, ch // 2):
        for x in range(0, w - cw + 1, cw // 2):
            s = int(si[y : y + ch, x : x + cw].sum())
            if s > best:
                best, best_yx = s, (y, x)
    return best_yx


def save_qc(res: SegmentationResult, out_dir: Path, image_id: str, downsample: int = 4) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ov = overlay(res.raw, res.labels)
    Image.fromarray(ov[::downsample, ::downsample]).save(out_dir / f"{image_id}_overlay_small.png")
    y, x = densest_si_crop(res.si)
    crop = (slice(y, y + 800), slice(x, x + 1200))
    raw_rgb = np.repeat(res.raw[crop][..., None], 3, axis=2)
    top = np.concatenate([raw_rgb, ov[crop]], axis=1)
    if res.mm is not None:
        etd = np.clip(res.mm.etd_n[crop] * 110, 0, 255).astype(np.uint8)
        etd_rgb = np.repeat(etd[..., None], 3, axis=2)
        before = overlay(res.raw[crop], res.bse_labels[crop])
        top = np.concatenate([top, np.concatenate([etd_rgb, before], axis=1)], axis=0)
    Image.fromarray(top).save(out_dir / f"{image_id}_crop.png")

    centers, hist = smoothed_histogram(res.n.ravel()[::5], *N_HIST_RANGE, N_HIST_BIN, 2.0)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.semilogy(centers, hist + 1, color="k", lw=1)
    t = res.thresholds
    ax.axvline(t.pore, color="tab:blue", ls="--", label=f"pore < {t.pore:.2f}")
    ax.axvline(t.si_low, color="tab:orange", ls="--", label=f"Si low {t.si_low:.2f}")
    ax.axvline(t.si_high, color="tab:red", ls="--", label=f"Si seed {t.si_high:.2f}")
    ax.axvline(1.0, color="grey", lw=0.5)
    ax.set_xlabel("normalised intensity (pore floor = 0, graphite peak = 1)")
    ax.set_ylabel("pixels")
    ax.set_title(f"{image_id} ({t.method})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{image_id}_hist.png", dpi=90)
    plt.close(fig)
