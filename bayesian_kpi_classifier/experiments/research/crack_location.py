"""Split detected cracks by where they sit: next to Si, next to pores, or inside graphite.

Uses the 1/4-resolution labels and 1/2-resolution crack skeletons saved by
extract_features.py, so the numbers approximate the full-resolution definition.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from crack_viz import COL, _unpack

PORE, GRAPHITE, SI = 0, 1, 2
CLASSES = ("Batch_1", "Batch_2", "Batch_3")


def locate(labels4: np.ndarray, cracks2: np.ndarray, rim: int = 2) -> dict:
    """rim is in 1/4-res px (rim=2 -> 8 full-res px)."""
    si = labels4 == SI
    pore = labels4 == PORE
    g = labels4 == GRAPHITE
    near_si = ndi.binary_dilation(si, iterations=rim)
    near_pore = ndi.binary_dilation(pore, iterations=rim)
    ys, xs = np.nonzero(cracks2)
    ys4 = np.minimum(ys // 2, labels4.shape[0] - 1)
    xs4 = np.minimum(xs // 2, labels4.shape[1] - 1)
    in_g = g[ys4, xs4]
    ns = near_si[ys4, xs4]
    npo = near_pore[ys4, xs4] & ~ns
    interior = in_g & ~ns & ~npo
    garea = g.sum() * 16.0
    n = max(len(ys), 1)
    return {
        "crack_frac_near_si": ns.sum() / n,
        "crack_frac_near_pore": npo.sum() / n,
        "crack_frac_graphite_interior": interior.sum() / n,
        "graphite_interior_crack_density": interior.sum() * 2 / garea * 1e4,
        "graphite_crack_density_approx": in_g.sum() * 2 / garea * 1e4,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-dir", type=Path, required=True)
    ap.add_argument("--cracks", type=Path, required=True, help="crack_features.csv (for image ids and batches)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ids = pd.read_csv(a.cracks)[["image_id", "batch"]].drop_duplicates()
    rows = []
    for iid, b in ids.itertuples(index=False):
        z = np.load(a.mask_dir / f"{iid}.npz")
        for src in ("rule", "learned"):
            rows.append({"image_id": iid, "batch": b, "source": src,
                         **locate(z[f"{src}_labels"], _unpack(z, f"{src}_cracks"))})
    d = pd.DataFrame(rows)
    d.to_csv(a.out / "crack_location.csv", index=False)
    cols = ["crack_frac_near_si", "crack_frac_near_pore", "crack_frac_graphite_interior", "graphite_interior_crack_density"]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    tests = []
    for ax, k in zip(axes, cols):
        txt = []
        for j, (src, mk, off) in enumerate((("rule", "o", -0.17), ("learned", "^", 0.17))):
            s = d[d.source == src]
            for i, bb in enumerate(CLASSES):
                v = s.loc[s.batch == bb, k].values
                ax.scatter(i + off + np.random.default_rng(i + 10 * j).uniform(-0.06, 0.06, len(v)), v, s=18,
                           color=COL[bb], marker=mk, alpha=0.8 if src == "rule" else 0.5, edgecolor="none")
                ax.hlines(np.median(v), i + off - 0.12, i + off + 0.12, color="k", lw=2 if src == "rule" else 1)
            g = [s.loc[s.batch == bb, k] for bb in CLASSES]
            p12, p23 = mannwhitneyu(g[0], g[1]).pvalue, mannwhitneyu(g[1], g[2]).pvalue
            tests.append({"metric": k, "source": src, **{f"median_{bb}": x.median() for bb, x in zip(CLASSES, g)},
                          "p_B1_B2": p12, "p_B1_B3": mannwhitneyu(g[0], g[2]).pvalue, "p_B2_B3": p23})
            txt.append(f"{src}: B1-B2 p={p12:.2f}, B2-B3 p={p23:.3f}")
        ax.set_xticks(range(3), ["B1", "B2", "B3"])
        ax.set_title(k + "\n" + "\n".join(txt), fontsize=8)
    fig.suptitle("Where detected cracks sit (within 8 px of Si, of pore, or inside graphite); circles rule, triangles learned")
    fig.tight_layout()
    fig.savefig(a.out / "crack_location.png", dpi=110)
    pd.DataFrame(tests).to_csv(a.out / "crack_location_tests.csv", index=False)
    print(d.groupby(["source", "batch"])[cols + ["graphite_crack_density_approx"]].median().round(3).to_string())
    print(pd.DataFrame(tests).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
