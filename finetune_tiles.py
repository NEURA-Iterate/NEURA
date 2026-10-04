"""Experiment B: fine-tune a small pretrained CNN end to end on augmented tiles.

Folds hold out whole samples (one per batch per fold); the held-out verdict is
the mean tile probability.  The same code runs the sample-level label-shuffle
control (--shuffle-seed) so the real-label result can be compared to chance.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torchvision

from augment_train import TILE, augment, tile_grid, to_224
from encoder import IMAGENET_MEAN, IMAGENET_STD
from kpis import load_gray, robust_normalise


def load_tiles(manifest: str, view: str, batches: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    m = pd.read_csv(manifest, dtype={"batch_id": str}).fillna("")
    tiles, meta = [], []
    for r in m.to_dict("records"):
        if r["batch_id"] not in batches or not r.get(f"view_{view}_path"):
            continue
        img = robust_normalise(load_gray(r[f"view_{view}_path"], 2))
        for y, x in tile_grid(img):
            tiles.append(img[y:y + TILE, x:x + TILE].astype(np.float32))
            meta.append(dict(sample_id=r["sample_id"], batch_id=r["batch_id"]))
    return np.stack(tiles), pd.DataFrame(meta)


def make_model(n_classes: int) -> torch.nn.Module:
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Linear(net.fc.in_features, n_classes)
    return net


def batch_tensor(tiles: list[np.ndarray]) -> torch.Tensor:
    x = torch.from_numpy(np.stack(tiles).astype(np.float32)).unsqueeze(1).repeat(1, 3, 1, 1)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def train_fold(tiles: np.ndarray, y: np.ndarray, train_idx: np.ndarray, n_classes: int, epochs: int,
               photometric: bool, rng: np.random.Generator, lr: float, batch_size: int) -> torch.nn.Module:
    net = make_model(n_classes)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * int(np.ceil(len(train_idx) / batch_size)))
    net.train()
    for _ in range(epochs):
        order = rng.permutation(train_idx)
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            xb = batch_tensor([to_224(augment(tiles[j], rng, photometric)) for j in idx])
            loss = torch.nn.functional.cross_entropy(net(xb), torch.from_numpy(y[idx]), label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return net.eval()


@torch.no_grad()
def predict_sample(net: torch.nn.Module, tiles: np.ndarray, idx: np.ndarray) -> np.ndarray:
    probs = []
    for i in range(0, len(idx), 32):
        xb = batch_tensor([to_224(tiles[j]) for j in idx[i:i + 32]])
        probs.append(torch.softmax(net(xb), dim=1).numpy())
    return np.concatenate(probs).mean(axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--view", default="BSE")
    ap.add_argument("--batches", default="1,2")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--photometric", action="store_true")
    ap.add_argument("--shuffle-seed", type=int, default=-1, help=">=0: permute sample labels with this seed (control)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    batches = a.batches.split(",")
    tiles, meta = load_tiles(a.manifest, a.view, batches)
    labels = meta.groupby("sample_id").batch_id.first()
    if a.shuffle_seed >= 0:
        labels = pd.Series(np.random.default_rng(a.shuffle_seed).permutation(labels.values), index=labels.index)
    classes = sorted(labels.unique())
    y = meta.sample_id.map(labels).map({c: i for i, c in enumerate(classes)}).to_numpy()
    # folds: one held-out sample per class per fold (grouped by sample)
    per_class = {c: list(rng.permutation(labels.index[labels == c])) for c in classes}
    n_folds = max(len(v) for v in per_class.values())
    folds = [[per_class[c][k] for c in classes if k < len(per_class[c])] for k in range(n_folds)]
    rows, t0 = [], time.time()
    for k, held in enumerate(folds):
        tr = ~meta.sample_id.isin(held).to_numpy()
        net = train_fold(tiles, y, np.where(tr)[0], len(classes), a.epochs, a.photometric, rng, a.lr, a.batch_size)
        for sid in held:
            idx = np.where((meta.sample_id == sid).to_numpy())[0]
            p = predict_sample(net, tiles, idx)
            rows.append(dict(sample_id=sid, true=labels[sid], pred=classes[int(p.argmax())],
                             **{f"p_{c}": float(p[i]) for i, c in enumerate(classes)}))
        print(f"fold {k + 1}/{n_folds} done ({time.time() - t0:.0f}s): " + ", ".join(f"{r['sample_id']} {r['true']}->{r['pred']}" for r in rows[-len(held):]), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    summary = dict(view=a.view, batches=a.batches, photometric=a.photometric, shuffle_seed=a.shuffle_seed,
                   epochs=a.epochs, n=len(df), accuracy=float((df.pred == df.true).mean()),
                   confusion=pd.crosstab(df.true, df.pred).to_dict())
    print(json.dumps(summary))
    with open(a.out.replace(".csv", ".json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
