"""Descriptive plots of batch separation for rule-based and learned-mask KPIs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "batch_cv"))
from rotating_cv import BASE

from fastnb import CLASSES, fit, kpi_loglik, posterior, transform

COL = {"Batch_1": "#1f77b4", "Batch_2": "#ff7f0e", "Batch_3": "#2ca02c"}
LABEL = {
    "frac_pore": "Porosity (fraction)",
    "graphite_crack_density": "Graphite crack density\n(crack px / 1e4 graphite px)",
    "si_cv_w256": "Si heterogeneity\n(CV of Si fraction, 256 px windows)",
    "graphite_aspect_ratio_median": "Graphite aspect ratio (median)",
    "si_fraction_of_solids": "Si fraction of solids",
    "si_ecd_d50": "Si D50 (px)",
    "si_contact_graphite": "Si-graphite contact fraction",
    "cbd_fraction_of_solids": "Binder fraction of solids",
}


def flagged(d):
    return d.trust_flags.fillna("").astype(str) != ""


def distributions(rule, learned, out):
    ks = list(LABEL)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, k in zip(axes.flat, ks):
        for j, (src, d, off) in enumerate((("rule", rule, -0.17), ("learned", learned, 0.17))):
            for i, b in enumerate(CLASSES):
                v = d.loc[d.batch == b, k].values
                x = i + off + np.random.default_rng(i + 10 * j).uniform(-0.06, 0.06, len(v))
                ax.scatter(x, v, s=18, color=COL[b], marker="o" if src == "rule" else "^",
                           alpha=0.8 if src == "rule" else 0.55, edgecolor="none")
                ax.hlines(np.median(v), i + off - 0.12, i + off + 0.12, color="k", lw=2 if src == "rule" else 1)
        ax.set_xticks(range(3), ["B1", "B2", "B3"])
        ax.set_title(LABEL[k], fontsize=9)
    axes.flat[0].scatter([], [], marker="o", color="grey", label="rule masks (left)")
    axes.flat[0].scatter([], [], marker="^", color="grey", label="learned masks (right)")
    axes.flat[0].legend(fontsize=8, loc="best")
    fig.suptitle("Per-image KPI values by batch (bar = batch median)")
    fig.tight_layout()
    fig.savefig(out / "kpi_distributions.png", dpi=120)
    plt.close(fig)


def decision_regions(d, pairs, out, name, title):
    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5.2))
    for ax, (kx, ky) in zip(np.atleast_1d(axes), pairs):
        kinds = [BASE.get(kx, "log"), BASE.get(ky, "log")]
        Z = transform(d[[kx, ky]].values, kinds)
        params = fit(Z, d.batch.values)
        x = d[kx].values
        yv = d[ky].values
        gx = np.linspace(x.min() - 0.15 * np.ptp(x), x.max() + 0.15 * np.ptp(x), 250)
        gy = np.linspace(yv.min() - 0.15 * np.ptp(yv), yv.max() + 0.15 * np.ptp(yv), 250)
        GX, GY = np.meshgrid(gx, gy)
        G = transform(np.column_stack([GX.ravel(), GY.ravel()]).clip(1e-6, None), kinds)
        P = posterior(kpi_loglik(G, params), np.ones(2)).reshape(*GX.shape, 3)
        top = P.argmax(-1)
        cmap = matplotlib.colors.ListedColormap([COL[c] for c in CLASSES])
        ax.contourf(GX, GY, top, levels=[-0.5, 0.5, 1.5, 2.5], cmap=cmap, alpha=0.15)
        ax.contour(GX, GY, P.max(-1), levels=[0.6, 0.9], colors="k", linewidths=[0.6, 1.0], linestyles=[":", "-"])
        fl = flagged(d).values
        for b in CLASSES:
            m = (d.batch == b).values
            ax.scatter(x[m & ~fl], yv[m & ~fl], color=COL[b], s=40, edgecolor="k", lw=0.5, label=b)
            ax.scatter(x[m & fl], yv[m & fl], facecolor="none", edgecolor=COL[b], s=50, lw=1.5)
        ax.set_xlabel(LABEL.get(kx, kx))
        ax.set_ylabel(LABEL.get(ky, ky))
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle(title + "  (shading = most probable batch; dotted/solid = top probability 0.6/0.9; hollow = trust-flagged)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / name, dpi=120)
    plt.close(fig)


def lda_projection(rule, learned, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (src, d) in zip(axes, (("rule masks", rule), ("learned masks", learned))):
        Z = transform(d[list(BASE)].values, list(BASE.values()))
        Z = (Z - Z.mean(0)) / Z.std(0)
        lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto", n_components=2).fit(Z, d.batch)
        P = lda.transform(Z)
        fl = flagged(d).values
        for b in CLASSES:
            m = (d.batch == b).values
            ax.scatter(P[m & ~fl, 0], P[m & ~fl, 1], color=COL[b], s=40, edgecolor="k", lw=0.5, label=b)
            ax.scatter(P[m & fl, 0], P[m & fl, 1], facecolor="none", edgecolor=COL[b], s=50, lw=1.5)
        coef = pd.DataFrame(lda.scalings_[:, :2], index=list(BASE), columns=["LD1", "LD2"])
        top1 = coef.LD1.abs().sort_values(ascending=False).index[:3]
        top2 = coef.LD2.abs().sort_values(ascending=False).index[:3]
        ax.set_xlabel("LD1: " + ", ".join(f"{k} ({coef.LD1[k]:+.2f})" for k in top1), fontsize=7)
        ax.set_ylabel("LD2: " + ", ".join(f"{k} ({coef.LD2[k]:+.2f})" for k in top2), fontsize=7)
        ax.set_title(f"LDA on all 12 KPIs, {src} (descriptive, fit on all images)", fontsize=10)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "lda_projection.png", dpi=120)
    plt.close(fig)


def shift_plot(rule, learned, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, (kx, ky) in zip(axes, (("frac_pore", "graphite_crack_density"), ("si_cv_w256", "graphite_aspect_ratio_median"))):
        for b in CLASSES:
            m = (rule.batch == b).values
            ax.scatter(rule.loc[m, kx], rule.loc[m, ky], color=COL[b], s=35, edgecolor="k", lw=0.5, label=f"{b} rule")
            ax.scatter(learned.loc[m, kx], learned.loc[m, ky], color=COL[b], s=35, marker="^", alpha=0.6)
            for xr, yr, xl, yl in zip(rule.loc[m, kx], rule.loc[m, ky], learned.loc[m, kx], learned.loc[m, ky]):
                ax.annotate("", (xl, yl), (xr, yr), arrowprops={"arrowstyle": "->", "color": COL[b], "lw": 0.7, "alpha": 0.6})
        ax.set_xlabel(LABEL[kx])
        ax.set_ylabel(LABEL[ky])
    axes[0].legend(fontsize=8)
    fig.suptitle("How each image moves from rule-based (circle) to learned masks (triangle)")
    fig.tight_layout()
    fig.savefig(out / "rule_to_learned_shift.png", dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", type=Path, required=True)
    ap.add_argument("--learned", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rule = pd.read_csv(a.rule).sort_values("image_id").reset_index(drop=True)
    learned = pd.read_csv(a.learned).set_index("image_id").loc[rule.image_id].reset_index()
    distributions(rule, learned, a.out)
    pairs = [("frac_pore", "graphite_crack_density"), ("frac_pore", "si_cv_w256"), ("graphite_crack_density", "si_cv_w256")]
    decision_regions(rule, pairs, a.out, "decision_regions_rule.png", "Rule-mask KPIs")
    decision_regions(learned, pairs, a.out, "decision_regions_learned.png", "Learned-mask KPIs")
    lda_projection(rule, learned, a.out)
    shift_plot(rule, learned, a.out)


if __name__ == "__main__":
    main()
