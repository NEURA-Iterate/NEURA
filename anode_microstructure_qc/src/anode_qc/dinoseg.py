"""Foundation-model segmentation: frozen DINO patch features of BSE, ETD/SE and Inlens, fused by a small decoder.

The backbone (DINOv3, or DINOv2 when DINOv3 is not accessible) stays frozen and is run once per detector at
full resolution; its patch features are cached. Only a small decoder is trained: it reduces the concatenated
features of the three detectors, upsamples them to pixels and refines boundaries using the raw detector images.
An ensemble of decoders trained on different specimens gives model-disagreement uncertainty, which is carried
through the existing KPI functions. Thin gaps and cracks stay with the dedicated ETD detectors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from torch import nn

from .segment import CBD, GAP, GRAPHITE, PORE, SI

DEFAULT_BACKBONE = "facebook/dinov3-vits16-pretrain-lvd1689m"
FALLBACK_BACKBONE = "facebook/dinov2-small"
CLASSES = (PORE, GRAPHITE, SI, CBD)
IGNORE = 255
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
_STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]


def stretch(img: np.ndarray) -> np.ndarray:
    """Per-image percentile stretch to [0, 1]; removes brightness/contrast differences between sessions."""
    lo, hi = np.percentile(img[::4, ::4], [0.5, 99.5])
    return np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)


def load_backbone(model_id: str | None = None):
    """Frozen backbone; tries DINOv3 first and falls back to DINOv2 (DINOv3 weights are gated on the Hub)."""
    from transformers import AutoModel

    last: Exception | None = None
    for mid in [model_id] if model_id else [DEFAULT_BACKBONE, FALLBACK_BACKBONE]:
        try:
            return AutoModel.from_pretrained(mid).eval(), mid
        except OSError as e:
            last = e
    raise RuntimeError(f"could not load a DINO backbone: {last}")


def _starts(n: int, tile: int) -> list[int]:
    if n <= tile:
        return [0]
    return [*range(0, n - tile, tile), n - tile]


@torch.inference_mode()
def patch_features(model, img: np.ndarray, tile: int = 448, batch: int = 8) -> np.ndarray:
    """Patch features (H // p, W // p, C) of a 2D image, computed in p-aligned tiles at native resolution."""
    p = model.config.patch_size
    skip = 1 + int(getattr(model.config, "num_register_tokens", 0) or 0)
    x = stretch(img)
    hp, wp = x.shape[0] // p * p, x.shape[1] // p * p
    x = x[:hp, :wp]
    th, tw = min(tile // p * p, hp), min(tile // p * p, wp)
    boxes = [(y, xx) for y in _starts(hp, th) for xx in _starts(wp, tw)]
    out = np.zeros((hp // p, wp // p, model.config.hidden_size), np.float32)
    cnt = np.zeros((hp // p, wp // p, 1), np.float32)
    for i in range(0, len(boxes), batch):
        chunk = boxes[i : i + batch]
        px = np.stack(
            [(np.repeat(x[None, y : y + th, xx : xx + tw], 3, 0) - _MEAN) / _STD for y, xx in chunk]
        )
        tok = model(pixel_values=torch.from_numpy(px)).last_hidden_state[:, skip:]
        tok = tok.reshape(len(chunk), th // p, tw // p, -1).float().numpy()
        for (y, xx), t in zip(chunk, tok, strict=True):
            out[y // p : (y + th) // p, xx // p : (xx + tw) // p] += t
            cnt[y // p : (y + th) // p, xx // p : (xx + tw) // p] += 1
    return (out / cnt).astype(np.float16)


def draft_targets(labels: np.ndarray, erode_px: int = 2) -> np.ndarray:
    """Confident training pixels from a draft label map: class interiors only; edges and gaps are unknown."""
    t = np.full(labels.shape, IGNORE, np.uint8)
    for c in CLASSES:
        m = labels == c
        t[ndi.binary_erosion(m, iterations=erode_px) if erode_px else m] = c
    return t


@dataclass
class Specimen:
    """One field of view: cached features of all detectors on the patch grid plus stretched detector images."""

    name: str
    batch: str
    feats: np.ndarray  # (C, h + 2, w + 2) float16, padded by one patch cell on every side
    imgs: np.ndarray  # (D, h * p, w * p) float16
    patch: int
    targets: np.ndarray | None = None  # (h * p, w * p) uint8, IGNORE for unknown

    @classmethod
    def build(cls, name, batch, feats: list[np.ndarray], imgs: list[np.ndarray], patch: int, targets=None):
        f = np.concatenate(feats, axis=-1).transpose(2, 0, 1)
        f = np.pad(f, ((0, 0), (1, 1), (1, 1)), mode="edge")
        h, w = (f.shape[1] - 2) * patch, (f.shape[2] - 2) * patch
        im = np.stack([stretch(i)[:h, :w] for i in imgs]).astype(np.float16)
        return cls(name, batch, f, im, patch, None if targets is None else targets[:h, :w])

    @property
    def shape(self) -> tuple[int, int]:
        return self.imgs.shape[1], self.imgs.shape[2]

    def window(self, y: int, x: int, size_y: int, size_x: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Features (with one cell of context on each side) and images of a p-aligned pixel window."""
        p = self.patch
        gy, gx = y // p, x // p
        f = self.feats[:, gy : gy + size_y // p + 2, gx : gx + size_x // p + 2]
        im = self.imgs[:, y : y + size_y, x : x + size_x]
        return torch.from_numpy(f.astype(np.float32)), torch.from_numpy(im.astype(np.float32))


class FusionDecoder(nn.Module):
    """Reduce fused patch features, upsample to pixels and refine edges with the raw detector images."""

    def __init__(self, feat_ch: int, n_img: int, patch: int, hidden: int = 64, n_classes: int = len(CLASSES)):
        super().__init__()
        self.patch = patch
        self.reduce = nn.Sequential(
            nn.Conv2d(feat_ch, hidden, 1), nn.GELU(), nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU()
        )
        self.refine = nn.Sequential(
            nn.Conv2d(hidden + n_img, 48, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(48, 48, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv2d(48, n_classes, 1),
        )

    def forward(self, feats: torch.Tensor, imgs: torch.Tensor) -> torch.Tensor:
        p = self.patch
        f = self.reduce(feats)
        f = F.interpolate(f, scale_factor=p, mode="bilinear", align_corners=False)[:, :, p:-p, p:-p]
        return self.refine(torch.cat([f, imgs], 1))


def train_decoder(
    specimens: list[Specimen],
    iters: int = 600,
    crop: int = 224,
    batch: int = 8,
    lr: float = 2e-3,
    seed: int = 0,
) -> FusionDecoder:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    s0 = specimens[0]
    p = s0.patch
    crop = crop // p * p
    model = FusionDecoder(s0.feats.shape[0], s0.imgs.shape[0], p)
    counts = np.zeros(len(CLASSES))
    for s in specimens:
        t = s.targets[::8, ::8]
        counts += [(t == c).sum() for c in CLASSES]
    weight = torch.tensor(1 / np.sqrt(np.maximum(counts / counts.sum(), 1e-4)), dtype=torch.float32)
    weight /= weight.mean()
    lut = np.full(256, IGNORE, np.int64)
    lut[list(CLASSES)] = np.arange(len(CLASSES))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=iters)
    model.train()
    for _ in range(iters):
        fs, ims, ts = [], [], []
        for _ in range(batch):
            s = specimens[rng.integers(len(specimens))]
            h, w = s.shape
            y = int(rng.integers(0, (h - crop) // p + 1)) * p
            x = int(rng.integers(0, (w - crop) // p + 1)) * p
            f, im = s.window(y, x, crop, crop)
            t = torch.from_numpy(lut[s.targets[y : y + crop, x : x + crop]])
            im = im * float(rng.uniform(0.85, 1.15)) + float(rng.uniform(-0.05, 0.05))
            if rng.random() < 0.5:
                f, im, t = f.flip(-1), im.flip(-1), t.flip(-1)
            fs.append(f), ims.append(im), ts.append(t)
        logits = model(torch.stack(fs), torch.stack(ims))
        loss = F.cross_entropy(logits, torch.stack(ts), weight=weight, ignore_index=IGNORE)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return model.eval()


@torch.inference_mode()
def predict_proba(model: FusionDecoder, s: Specimen, tile: int = 896) -> np.ndarray:
    """Class probabilities (len(CLASSES), H, W) for a whole field, in p-aligned tiles."""
    p = s.patch
    h, w = s.shape
    th, tw = min(tile // p * p, h), min(tile // p * p, w)
    out = np.zeros((len(CLASSES), h, w), np.float32)
    for y in _starts(h, th):
        for x in _starts(w, tw):
            f, im = s.window(y, x, th, tw)
            out[:, y : y + th, x : x + tw] = torch.softmax(model(f[None], im[None]), 1)[0].numpy()
    return out


def to_labels(proba: np.ndarray, gap: np.ndarray | None = None) -> np.ndarray:
    """Most likely class; thin ETD gaps from the dedicated detector are kept unless the model says Si."""
    lab = np.asarray(CLASSES, np.uint8)[proba.argmax(0)]
    if gap is not None:
        g = gap[: lab.shape[0], : lab.shape[1]]
        lab[g & (lab != SI)] = GAP
    return lab


def class_iou(pred: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    known = targets != IGNORE
    out = {}
    for c, name in zip(CLASSES, ("pore", "graphite", "si", "cbd"), strict=True):
        a, b = (pred == c) & known, targets == c
        out[f"iou_{name}"] = float((a & b).sum() / max((a | b).sum(), 1))
    out["pixel_agreement"] = float((pred[known] == targets[known]).mean())
    return out


def entropy(proba: np.ndarray) -> np.ndarray:
    return -(proba * np.log(np.clip(proba, 1e-8, 1))).sum(0) / np.log(len(CLASSES))
