"""Cache frozen DINO patch features of every field (BSE, ETD/SE, Inlens) for the decoder in ``anode_qc.dinoseg``.

    python scripts/dino_extract_features.py --data <Hackathon-Polaron> --out dino_cache [--model facebook/dinov2-small]

Without ``--model`` DINOv3 ViT-S/16 is tried first (gated on the Hub: accept the licence and set HF_TOKEN), then
DINOv2-small. One ``<image_id>.npz`` per field (no batch label in the name) holds ``bse``, ``etd`` and ``inlens``
arrays of shape (H // patch, W // patch, hidden) in float16 (~190 MB per field for the small models).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from anode_qc.cli import load_detectors
from anode_qc.config import Config
from anode_qc.data import find_bse_images
from anode_qc.dinoseg import load_backbone, patch_features


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--tile", type=int, default=448)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    model, mid = load_backbone(a.model)
    out = a.out / mid.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)
    print("backbone", mid, "patch", model.config.patch_size, flush=True)
    cfg = Config()
    for im in find_bse_images(a.data):
        f = out / f"{im.image_id}.npz"
        if f.exists():
            continue
        t = time.time()
        raw, etd, inl = load_detectors(im, cfg)
        if etd is None or inl is None:
            print(f"skip {f.name}: missing ETD/SE or Inlens", flush=True)
            continue
        views = {"bse": raw, "etd": etd, "inlens": inl}
        np.savez(f, **{k: patch_features(model, v, a.tile) for k, v in views.items()})
        print(f.name, raw.shape, f"{time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
