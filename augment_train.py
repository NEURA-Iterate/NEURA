"""Train on aggressively augmented tiles, evaluate on whole held-out samples.

Experiment A: tiles -> augmentations -> frozen DINOv2 embedding (cached) ->
logistic head trained on tiles of the training samples; the held-out sample's
verdict is the mean tile probability.  Grouping is by sample_id, so no tile
or view of a held-out sample is ever seen in training.  A sample-level
label-shuffle control is run through the identical pipeline.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from encoder import IMAGENET_MEAN, IMAGENET_STD, load_model
from kpis import VIEWS, load_gray, robust_normalise

TILE = 384
OUT_RES = 224


def tile_grid(img01: np.ndarray, tile: int = TILE) -> list[tuple[int, int]]:
    h, w = img01.shape
    return [(y, x) for y in range(0, h - tile + 1, tile) for x in range(0, w - tile + 1, tile)]


def augment(t: np.ndarray, rng: np.random.Generator, photometric: bool) -> np.ndarray:
    """Geometric: flips, 90-degree rotations, random crop 60-100% rescaled.
    Photometric (optional): gamma, contrast, brightness, Gaussian noise, blur.
    All keep the tile a plausible SEM micrograph of the same material."""
    if rng.random() < 0.5:
        t = t[:, ::-1]
    if rng.random() < 0.5:
        t = t[::-1, :]
    t = np.rot90(t, int(rng.integers(4)))
    s = int(t.shape[0] * rng.uniform(0.6, 1.0))
    y = int(rng.integers(0, t.shape[0] - s + 1))
    x = int(rng.integers(0, t.shape[1] - s + 1))
    t = t[y:y + s, x:x + s]
    if photometric:
        t = np.clip(t, 0, 1) ** rng.uniform(0.7, 1.4)
        t = (t - 0.5) * rng.uniform(0.7, 1.3) + 0.5 + rng.uniform(-0.1, 0.1)
        if rng.random() < 0.7:
            t = ndi.gaussian_filter(t, rng.uniform(0.0, 1.0))
        t = t + rng.normal(0, rng.uniform(0.0, 0.05), t.shape)
    return np.clip(np.ascontiguousarray(t), 0, 1)


def to_224(t: np.ndarray) -> np.ndarray:
    z = OUT_RES / t.shape[0]
    return ndi.zoom(t, z, order=1)[:OUT_RES, :OUT_RES]


@torch.no_grad()
def embed(model: torch.nn.Module, batch: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(batch).float().unsqueeze(1).repeat(1, 3, 1, 1)
    return model((x - IMAGENET_MEAN) / IMAGENET_STD).cpu().numpy()


def build_cache(manifest: str, out: Path, n_aug: int, photometric: bool, seed: int, model_name: str) -> None:
    rng = np.random.default_rng(seed)
    model, _ = load_model(model_name)
    m = pd.read_csv(manifest, dtype={"batch_id": str}).fillna("")
    recs, feats = [], []
    t0 = time.time()
    for r in m.to_dict("records"):
        if not r["batch_id"]:
            continue
        for view in VIEWS:
            path = r.get(f"view_{view}_path")
            if not path:
                continue
            img = robust_normalise(load_gray(path, 2))
            for ti, (y, x) in enumerate(tile_grid(img)):
                t = img[y:y + TILE, x:x + TILE]
                copies = [to_224(t)] + [to_224(augment(t, rng, photometric)) for _ in range(n_aug)]
                feats.append(embed(model, np.stack(copies)))
                for k in range(n_aug + 1):
                    recs.append(dict(sample_id=r["sample_id"], batch_id=r["batch_id"], view=view, tile=ti, aug=k))
        print(f"{r['sample_id']} done ({time.time() - t0:.0f}s)", flush=True)
    np.save(out.with_suffix(".npy"), np.concatenate(feats).astype(np.float32))
    pd.DataFrame(recs).to_csv(out.with_suffix(".csv"), index=False)


def sample_level_loo(meta: pd.DataFrame, X: np.ndarray, labels: pd.Series, view: str | None,
                     train_aug: bool, C: float) -> pd.DataFrame:
    """labels: sample_id -> batch label (possibly shuffled). Returns one row per sample."""
    sel = meta.view == view if view else np.ones(len(meta), bool)
    meta = meta[sel].reset_index(drop=True)
    X = X[sel.to_numpy()]
    y_tile = meta.sample_id.map(labels).to_numpy()
    classes = sorted(labels.unique())
    rows = []
    for sid in labels.index:
        tr = (meta.sample_id != sid).to_numpy() & ((meta.aug > 0).to_numpy() if train_aug else (meta.aug == 0).to_numpy())
        te = (meta.sample_id == sid).to_numpy()
        if te.sum() == 0:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
        clf.fit(X[tr], y_tile[tr])
        p_plain = clf.predict_proba(X[te & (meta.aug == 0).to_numpy()]).mean(axis=0)
        p_tta = clf.predict_proba(X[te]).mean(axis=0)
        row = dict(sample_id=sid, true=labels[sid], pred=classes[int(np.argmax(p_plain))],
                   pred_tta=classes[int(np.argmax(p_tta))])
        row.update({f"p_{c}": p_plain[i] for i, c in enumerate(classes)})
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--manifest", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--n-aug", type=int, default=6)
    c.add_argument("--photometric", action="store_true")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--model", default="dinov2_vits14")
    e = sub.add_parser("eval")
    e.add_argument("--cache", required=True)
    e.add_argument("--batches", default="1,2")
    e.add_argument("--C", type=float, default=0.01)
    e.add_argument("--n-shuffles", type=int, default=20)
    e.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "cache":
        build_cache(a.manifest, Path(a.out), a.n_aug, a.photometric, a.seed, a.model)
        return
    meta = pd.read_csv(Path(a.cache).with_suffix(".csv"), dtype={"batch_id": str})
    X = np.load(Path(a.cache).with_suffix(".npy"))
    keep = meta.batch_id.isin(a.batches.split(","))
    meta, X = meta[keep].reset_index(drop=True), X[keep.to_numpy()]
    labels = meta.groupby("sample_id").batch_id.first()
    rng = np.random.default_rng(0)
    results, preds = [], {}
    for view in VIEWS + [None]:
        for train_aug in (False, True):
            name = f"{view or 'all'}|{'aug' if train_aug else 'noaug'}"
            if view is None:
                per = [sample_level_loo(meta, X, labels, v, train_aug, a.C) for v in VIEWS]
                pcols = [c for c in per[0].columns if c.startswith("p_")]
                df = per[0][["sample_id", "true"]].copy()
                avg = sum(p.set_index("sample_id")[pcols].reindex(df.sample_id).fillna(1 / len(pcols)).to_numpy() for p in per) / len(per)
                df["pred"] = [pcols[i][2:] for i in avg.argmax(axis=1)]
                df[pcols] = avg
            else:
                df = sample_level_loo(meta, X, labels, view, train_aug, a.C)
            acc = float((df.pred == df.true).mean())
            null = []
            for _ in range(a.n_shuffles):
                sh = pd.Series(rng.permutation(labels.values), index=labels.index)
                if view is None:
                    per = [sample_level_loo(meta, X, sh, v, train_aug, a.C) for v in VIEWS]
                    avg = sum(p.set_index("sample_id")[pcols].reindex(df.sample_id).fillna(1 / len(pcols)).to_numpy() for p in per) / len(per)
                    null.append(float(np.mean([pcols[i][2:] for i in avg.argmax(axis=1)] == sh.reindex(df.sample_id).values)))
                else:
                    d2 = sample_level_loo(meta, X, sh, view, train_aug, a.C)
                    null.append(float((d2.pred == d2.true).mean()))
            results.append(dict(config=name, n=len(df), accuracy=acc, shuffle_mean=float(np.mean(null)),
                                shuffle_p95=float(np.percentile(null, 95)),
                                p_value=float((np.sum(np.array(null) >= acc) + 1) / (len(null) + 1)),
                                confusion=json.dumps(pd.crosstab(df.true, df.pred).to_dict())))
            preds[name] = df
            print(results[-1], flush=True)
    pd.DataFrame(results).to_csv(a.out, index=False)
    pd.concat(preds, names=["config"]).reset_index(level=0).to_csv(Path(a.out).with_name(Path(a.out).stem + "_predictions.csv"), index=False)


if __name__ == "__main__":
    main()
