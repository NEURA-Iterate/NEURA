"""Crack representations: per-batch distributions of crack metrics and crack maps."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import kruskal, mannwhitneyu

CLASSES = ("Batch_1", "Batch_2", "Batch_3")
COL = {"Batch_1": "#1f77b4", "Batch_2": "#ff7f0e", "Batch_3": "#2ca02c"}
GRAPHITE = 1
METRICS = {
    "graphite_crack_len_density": "Crack length density\n(crack px / 1e4 graphite px)",
    "graphite_crack_count_density": "Crack count density\n(cracks / 1e6 graphite px)",
    "graphite_crack_mean_len": "Mean crack length (px)",
    "graphite_crack_p90_len": "Crack length p90 (px)",
    "graphite_frac_particles_cracked": "Fraction of graphite\ncomponents with a crack",
    "graphite_crack_frac_horizontal": "Fraction of crack length\nwithin 30 deg of horizontal",
    "graphite_crack_cv_w256": "Crack-density variability\n(CV over 256 px windows)",
    "graphite_crack_len_density_k4": "Crack length density,\nlooser threshold (k=4)",
    "graphite_crack_len_density_k8": "Crack length density,\nstricter threshold (k=8)",
}


def _p(d, k):
    g = [d.loc[d.batch == b, k].dropna() for b in CLASSES]
    return (kruskal(*g).pvalue, mannwhitneyu(g[0], g[1]).pvalue,
            mannwhitneyu(g[0], g[2]).pvalue, mannwhitneyu(g[1], g[2]).pvalue)


def distributions(cf: pd.DataFrame, out: Path) -> pd.DataFrame:
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    rows = []
    for ax, (k, lab) in zip(axes.flat, METRICS.items()):
        txt = []
        for j, (src, mk, off) in enumerate((("rule", "o", -0.17), ("learned", "^", 0.17))):
            d = cf[cf.source == src]
            for i, b in enumerate(CLASSES):
                v = d.loc[d.batch == b, k].values
                x = i + off + np.random.default_rng(i + 10 * j).uniform(-0.06, 0.06, len(v))
                ax.scatter(x, v, s=18, color=COL[b], marker=mk, alpha=0.8 if src == "rule" else 0.5, edgecolor="none")
                ax.hlines(np.nanmedian(v), i + off - 0.12, i + off + 0.12, color="k", lw=2 if src == "rule" else 1)
            kw, p12, p13, p23 = _p(d, k)
            rows.append({"metric": k, "source": src, **{f"median_{b}": d.loc[d.batch == b, k].median() for b in CLASSES},
                         "p_kruskal": kw, "p_B1_B2": p12, "p_B1_B3": p13, "p_B2_B3": p23})
            txt.append(f"{src}: B1-B2 p={p12:.2f}, B2-B3 p={p23:.3f}")
        ax.set_xticks(range(3), ["B1", "B2", "B3"])
        ax.set_title(lab + "\n" + "\n".join(txt), fontsize=8)
    axes.flat[0].scatter([], [], marker="o", color="grey", label="rule masks (left)")
    axes.flat[0].scatter([], [], marker="^", color="grey", label="learned masks (right)")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Graphite crack representations by batch (bar = median; Mann-Whitney p-values)")
    fig.tight_layout()
    fig.savefig(out / "crack_representations.png", dpi=110)
    plt.close(fig)
    return pd.DataFrame(rows)


def _unpack(z, key):
    shape = tuple(z[f"{key}_shape"])
    return np.unpackbits(z[key])[: shape[0] * shape[1]].reshape(shape).astype(bool)


def crack_maps(cf: pd.DataFrame, mask_dir: Path, images: Path, out: Path, crop=900):
    from anode_qc.data import find_bse_images, load_bse

    paths = {im.image_id: im.path for im in find_bse_images(images)}
    rule = cf[cf.source == "rule"]
    picks = []
    for b in CLASSES:
        d = rule[rule.batch == b]
        picks.append(d.iloc[(d.graphite_crack_len_density - d.graphite_crack_len_density.median()).abs().argsort().iloc[0]])
    fig, axes = plt.subplots(3, 3, figsize=(18, 13), gridspec_kw={"width_ratios": [2.2, 1, 1]})
    for r, row in enumerate(picks):
        z = np.load(mask_dir / f"{row.image_id}.npz")
        bse = load_bse(paths[row.image_id]).astype(float)
        lo, hi = np.percentile(bse, [1, 99.5])
        bse = np.clip((bse - lo) / (hi - lo), 0, 1)
        cr, cl = _unpack(z, "rule_cracks"), _unpack(z, "learned_cracks")  # 1/2 resolution
        g4 = z["rule_labels"] == GRAPHITE  # 1/4 resolution
        w4 = 64  # 256 full-res px
        h4, wd4 = (g4.shape[0] // w4) * w4, (g4.shape[1] // w4) * w4
        garea = g4[:h4, :wd4].reshape(h4 // w4, w4, wd4 // w4, w4).sum((1, 3)) * 16.0
        c2 = cr[: 2 * h4, : 2 * wd4]
        clen = c2.reshape(h4 // w4, 2 * w4, wd4 // w4, 2 * w4).sum((1, 3)) * 2.0
        dens = np.where(garea >= 0.2 * 256 * 256, clen / np.maximum(garea, 1) * 1e4, np.nan)
        ax = axes[r, 0]
        ax.imshow(bse[::4, ::4], cmap="gray", extent=(0, bse.shape[1], bse.shape[0], 0))
        im = ax.imshow(dens, cmap="inferno", alpha=0.55, extent=(0, wd4 * 4, h4 * 4, 0), vmin=0,
                       vmax=np.nanpercentile(dens, 98))
        plt.colorbar(im, ax=ax, fraction=0.02, label="crack px / 1e4 graphite px")
        ax.set_title(f"{row.batch} {row.image_id}: local graphite crack density (rule masks), 256 px windows; "
                     f"image value {row.graphite_crack_len_density:.0f}; grey = <20% graphite", fontsize=9)
        iy, ix = np.unravel_index(np.nanargmax(np.nan_to_num(dens, nan=-1)), dens.shape)
        cy, cx = iy * 256 + 128, ix * 256 + 128
        y0 = int(np.clip(cy - crop // 2, 0, bse.shape[0] - crop))
        x0 = int(np.clip(cx - crop // 2, 0, bse.shape[1] - crop))
        ax.add_patch(plt.Rectangle((x0, y0), crop, crop, fill=False, ec="cyan", lw=1.5))
        sub = bse[y0: y0 + crop, x0: x0 + crop]
        for c, (name, m, colr) in enumerate((("rule-mask cracks", cr, (1, 0.1, 0.1)), ("learned-mask cracks", cl, (0.1, 0.9, 1)))):
            mm = np.kron(m[y0 // 2: (y0 + crop) // 2, x0 // 2: (x0 + crop) // 2], np.ones((2, 2), bool))[: sub.shape[0], : sub.shape[1]]
            rgb = np.dstack([sub] * 3)
            rgb[ndi.binary_dilation(mm)] = colr
            axes[r, c + 1].imshow(rgb)
            axes[r, c + 1].set_title(f"{name} on BSE (zoom = cyan box, {crop} px)", fontsize=9)
            axes[r, c + 1].axis("off")
    fig.suptitle("Where the cracks are (ETD ridge detector inside graphite/Si); one median-crack-density image per batch")
    fig.tight_layout()
    fig.savefig(out / "crack_maps.png", dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cracks", type=Path, required=True)
    ap.add_argument("--mask-dir", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cf = pd.read_csv(a.cracks)
    distributions(cf, a.out).to_csv(a.out / "crack_representation_tests.csv", index=False)
    crack_maps(cf, a.mask_dir, a.images, a.out)


if __name__ == "__main__":
    main()
