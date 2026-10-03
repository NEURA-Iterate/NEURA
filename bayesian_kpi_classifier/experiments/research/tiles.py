"""Tile-level KPIs: within-image vs between-batch variation, and tile-trained classifiers."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "batch_cv"))
from rotating_cv import BASE

from fastnb import (
    CLASSES,
    best_subset,
    fit,
    kpi_loglik,
    loo_loglik,
    rotating_rounds,
    summarise,
    transform,
)

COL = {"Batch_1": "#1f77b4", "Batch_2": "#ff7f0e", "Batch_3": "#2ca02c"}


def variance_parts(tiles: pd.DataFrame) -> pd.DataFrame:
    """Share of transformed-KPI variance between batches, between images within a batch, and within images."""
    rows = []
    for src, t in tiles.groupby("source"):
        for k, kind in BASE.items():
            z = pd.Series(transform(t[[k]].values, [kind])[:, 0], index=t.index)
            z = z.replace([np.inf, -np.inf], np.nan)
            ok = z.notna()
            tt, z = t[ok], z[ok]
            img_mean = z.groupby(tt.image_id).transform("mean")
            bat_mean = z.groupby(tt.batch).transform("mean")
            tot = ((z - z.mean()) ** 2).sum()
            rows.append({
                "source": src, "kpi": k,
                "between_batch": ((bat_mean - z.mean()) ** 2).sum() / tot,
                "between_image_within_batch": ((img_mean - bat_mean) ** 2).sum() / tot,
                "within_image": ((z - img_mean) ** 2).sum() / tot,
            })
    return pd.DataFrame(rows)


def plot_variance(v: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, (src, d) in zip(axes, v.groupby("source")):
        d = d.set_index("kpi").loc[list(BASE)]
        left = np.zeros(len(d))
        for col, c in (("between_batch", "#2ca02c"), ("between_image_within_batch", "#9467bd"), ("within_image", "#bbbbbb")):
            ax.barh(range(len(d)), d[col], left=left, color=c, label=col.replace("_", " "))
            left += d[col].values
        ax.set_yticks(range(len(d)), d.index, fontsize=8)
        ax.set_title(f"{src} masks: where does each KPI vary? (4 tiles per image)")
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "tile_variance_decomposition.png", dpi=120)
    plt.close(fig)


def plot_tiles(tiles: pd.DataFrame, out: Path):
    pairs = [("frac_pore", "graphite_crack_density"), ("si_cv_w256", "graphite_aspect_ratio_median")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for r, (src, t) in enumerate(tiles.groupby("source")):
        for c, (kx, ky) in enumerate(pairs):
            ax = axes[r, c]
            for b in CLASSES:
                d = t[t.batch == b]
                ax.scatter(d[kx], d[ky], s=10, color=COL[b], alpha=0.35)
                m = d.groupby("image_id")[[kx, ky]].mean()
                ax.scatter(m[kx], m[ky], s=45, color=COL[b], edgecolor="k", lw=0.6, label=b)
            ax.set_xlabel(kx)
            ax.set_ylabel(ky)
            ax.set_title(f"{src} masks: tiles (small) and image means (large)", fontsize=10)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "tile_scatter.png", dpi=120)
    plt.close(fig)


def tile_classifier(tiles: pd.DataFrame, src: str, kpis_fixed=None):
    """Train on tiles of training images, predict an image by averaging tile log-posteriors."""
    t = tiles[tiles.source == src].sort_values(["image_id", "tile"]).reset_index(drop=True)
    img = t.groupby("image_id").agg(batch=("batch", "first")).reset_index()
    kpis = list(BASE)
    Z = transform(t[kpis].values, list(BASE.values()))
    Z = np.where(np.isfinite(Z), Z, np.nan)
    Z = np.where(np.isnan(Z), np.nanmean(Z, 0), Z)
    rows = []
    for r, trm, tem in rotating_rounds(img):
        tr_ids, te_ids = set(img.image_id[trm]), set(img.image_id[tem])
        mtr = t.image_id.isin(tr_ids).values
        # subset search on image-level LOO (tile-mean rows) to stay comparable
        im_tr = t[mtr].groupby("image_id")
        Zimg = np.vstack([Z[mtr][(t[mtr].image_id == i).values].mean(0) for i in im_tr.groups])
        yimg = np.array([t[mtr][t[mtr].image_id == i].batch.iloc[0] for i in im_tr.groups])
        yi = np.array([CLASSES.index(c) for c in yimg])
        if kpis_fixed is None:
            best, _ = best_subset(loo_loglik(Zimg, yimg), yi, balanced=True)
        else:
            best = [kpis.index(k) for k in kpis_fixed]
        w = np.zeros(len(kpis))
        w[best] = 1
        params = fit(Z[mtr], t.batch.values[mtr])
        for iid in te_ids:
            m = (t.image_id == iid).values
            ll = np.einsum("mkp,p->mk", kpi_loglik(Z[m], params), w).mean(0, keepdims=True)
            P = np.exp(ll - ll.max())
            P /= P.sum()
            rows.append({"round": r, "image_id": iid, "true": t.batch[m].iloc[0], **{f"P_{c}": P[0, i] for i, c in enumerate(CLASSES)}})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    tiles = pd.read_csv(a.tiles)
    v = variance_parts(tiles)
    v.to_csv(a.out / "tile_variance_decomposition.csv", index=False)
    plot_variance(v, a.out)
    plot_tiles(tiles, a.out)
    res = []
    for src in ("rule", "learned"):
        p = tile_classifier(tiles, src)
        p.to_csv(a.out / f"pred__tiles_{src}.csv", index=False)
        res.append({"table": f"tiles_{src}", "strategy": "subset_bal", **summarise(p)})
        print(res[-1], flush=True)
    pd.DataFrame(res).to_csv(a.out / "tile_classifier_summary.csv", index=False)


if __name__ == "__main__":
    main()
