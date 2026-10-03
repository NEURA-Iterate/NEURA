"""Split-half reliability of each KPI.

Compute every KPI separately on the left and right halves of each image. A KPI
is only usable for batch comparison if the two halves of the same sample agree
much better than different samples do. Reports, per KPI: Spearman correlation
between halves across samples, the within-sample half-difference MAD, the
between-sample MAD, and reliability = 1 - within_var / total_var (ICC-like).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

from kpis import DOWNSAMPLE, VIEW_FUNCS, VIEWS, load_gray


def halves_for_sample(row: dict, downsample: int = DOWNSAMPLE) -> list[dict]:
    out = []
    for side in ("left", "right"):
        rec = {"sample_id": row["sample_id"], "batch_id": row.get("batch_id", ""), "half": side}
        for view in VIEWS:
            path = row.get(f"view_{view}_path")
            if not isinstance(path, str) or not path:
                continue
            img = load_gray(path, downsample)
            w = img.shape[1] // 2
            part = img[:, :w] if side == "left" else img[:, w:]
            rec.update(VIEW_FUNCS[view](part, downsample))
        out.append(rec)
    return out


def reliability_table(halves: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in halves.columns if c.startswith("kpi__")]
    left = halves[halves.half == "left"].set_index("sample_id")[cols]
    right = halves[halves.half == "right"].set_index("sample_id")[cols].loc[left.index]
    rows = []
    for c in cols:
        a, b = left[c].to_numpy(float), right[c].to_numpy(float)
        ok = ~(np.isnan(a) | np.isnan(b))
        a, b = a[ok], b[ok]
        mean = (a + b) / 2
        within_var = np.mean((a - b) ** 2) / 2
        total_var = np.var(np.concatenate([a, b]), ddof=1)
        rows.append(dict(kpi=c, n=int(ok.sum()), spearman_halves=float(stats.spearmanr(a, b).correlation),
                         reliability=float(1 - within_var / total_var) if total_var > 0 else np.nan,
                         within_half_mad=float(np.median(np.abs(a - b))), between_sample_mad=float(stats.median_abs_deviation(mean)),
                         median=float(np.median(mean))))
    return pd.DataFrame(rows).sort_values("reliability", ascending=False).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", default="kpi_reliability.csv")
    p.add_argument("--halves-out", default=None, help="optional CSV with the per-half KPI values")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    manifest = pd.read_csv(args.manifest, dtype={"batch_id": str}).fillna({"batch_id": ""})
    with ProcessPoolExecutor(args.workers) as ex:
        recs = [r for rs in ex.map(halves_for_sample, manifest.to_dict("records")) for r in rs]
    halves = pd.DataFrame(recs)
    if args.halves_out:
        halves.to_csv(args.halves_out, index=False)
    table = reliability_table(halves)
    table.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
