"""Score an ensemble of held-out prediction files and draw the per-image probability chart."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "batch_cv"))
from rotating_cv import plot_per_image_probabilities

from fastnb import CLASSES, summarise

COLS = [f"P_{c}" for c in CLASSES]


def tier(p: float, high=0.9, low=0.6) -> str:
    return "high" if p >= high else ("medium" if p >= low else "review")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, nargs="+", required=True, help="pred__*.csv files from strategies.py")
    ap.add_argument("--how", choices=("geo", "mean"), default="geo")
    ap.add_argument("--kpis", type=Path, required=True, help="KPI table with trust_flags")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    preds = [pd.read_csv(p) for p in a.preds]
    out = preds[0][["round", "image_id", "true"]].copy()
    stack = np.stack([p[COLS].values for p in preds])
    if a.how == "mean":
        P = stack.mean(0)
    else:
        P = np.exp(np.log(np.clip(stack, 1e-12, None)).mean(0))
        P /= P.sum(1, keepdims=True)
    out[COLS] = P
    out["pred"] = np.array(CLASSES)[P.argmax(1)]
    out["correct"] = out.pred == out.true
    out["tier"] = [tier(x) for x in P.max(1)]
    out.to_csv(a.out / "predictions.csv", index=False)
    flags = pd.read_csv(a.kpis).set_index("image_id").trust_flags.fillna("").astype(str)
    img = out.groupby("image_id").agg(true=("true", "first"), **{c: (c, "mean") for c in COLS},
                                      tiers_seen=("tier", lambda s: ",".join(sorted(set(s))))).reset_index()
    img["pred"] = np.array(CLASSES)[img[COLS].values.argmax(1)]
    img["correct"] = img.pred == img.true
    img["trust_flags"] = flags.loc[img.image_id].values
    img.to_csv(a.out / "per_image.csv", index=False)
    plot_per_image_probabilities(img, a.out / "per_image_probabilities.png")
    s = summarise(out)
    rel = out.groupby("tier").agg(n=("correct", "size"), accuracy=("correct", "mean"))
    conf = pd.crosstab(out.true, out.pred)
    txt = "\n".join([f"{k}: {v}" for k, v in s.items()]) + "\n\nTier reliability:\n" + rel.to_string() + "\n\nConfusion (rows true):\n" + conf.to_string()
    (a.out / "summary.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
