"""Frozen pretrained encoder -> one embedding per view per sample.

Each image is downsampled, robustly normalised to [0, 1], tiled into square
patches and passed through a frozen ImageNet/DINOv2 backbone; the per-view
embedding is the mean over tiles. Output columns are ``emb__<view>__<model>__<i>``
in the same one-row-per-sample CSV the evaluation harness reads. No training
happens here: with ~30 independent samples there is nothing to fit safely.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

from kpis import VIEWS, load_gray, robust_normalise

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_model(name: str) -> tuple[torch.nn.Module, int]:
    if name.startswith("dinov2"):
        # weights + code pre-fetched into torch.hub.get_dir(); offline-safe
        model = torch.hub.load(torch.hub.get_dir() + "/facebookresearch_dinov2_main", name, source="local")
        dim = model.embed_dim
    elif name == "resnet50":
        import torchvision
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = torch.nn.Identity()
        dim = 2048
    else:
        raise ValueError(name)
    return model.eval(), dim


def tiles(img01: np.ndarray, tile: int, stride: int) -> np.ndarray:
    h, w = img01.shape
    ys = list(range(0, h - tile + 1, stride)) or [0]
    xs = list(range(0, w - tile + 1, stride)) or [0]
    out = [img01[y:y + tile, x:x + tile] for y in ys for x in xs]
    return np.stack(out)


@torch.no_grad()
def embed_image(model: torch.nn.Module, img: np.ndarray, tile: int, stride: int, batch: int) -> np.ndarray:
    t = tiles(robust_normalise(img), tile, stride)
    feats = []
    for i in range(0, len(t), batch):
        x = torch.from_numpy(t[i:i + batch]).float().unsqueeze(1).repeat(1, 3, 1, 1)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        feats.append(model(x).cpu().numpy())
    return np.concatenate(feats).mean(axis=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", default="features_emb.csv")
    p.add_argument("--model", default="dinov2_vits14", choices=["dinov2_vits14", "resnet50"])
    p.add_argument("--downsample", type=int, default=2)
    p.add_argument("--tile", type=int, default=224)
    p.add_argument("--stride", type=int, default=224)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    model, dim = load_model(args.model)
    manifest = pd.read_csv(args.manifest, dtype={"batch_id": str}).fillna({"batch_id": ""})
    rows = []
    t0 = time.time()
    for i, r in enumerate(manifest.to_dict("records")):
        row = {"sample_id": r["sample_id"], "batch_id": r["batch_id"]}
        for view in VIEWS:
            path = r.get(f"view_{view}_path")
            if not isinstance(path, str) or not path:
                continue
            img = load_gray(path, args.downsample)
            vec = embed_image(model, img, args.tile, args.stride, args.batch)
            row.update({f"emb__{view}__{args.model}__{k}": float(v) for k, v in enumerate(vec)})
        rows.append(row)
        print(f"[{i + 1}/{len(manifest)}] {r['sample_id']} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} samples x {len(df.columns) - 2} embedding dims to {args.out}")


if __name__ == "__main__":
    main()
