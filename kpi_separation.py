"""Per-KPI separation check: does each KPI vary less within batches than between them?

For every feature column: per-batch median, pooled within-batch spread (MAD),
standardised between-batch range (max|median_a - median_b| / MAD), Kruskal-Wallis
p-value, and the accuracy of a one-feature nearest-median classifier under the same
leave-one-sample-out protocol. Sorted by separation so weak KPIs are obvious.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

import evaluation as ev


def separation_table(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    df = ev.labelled_rows(df)
    batches = sorted(df[ev.BATCH_COL].unique())
    rows = []
    for col in feature_cols:
        x = df[col].astype(float)
        groups = [x[df[ev.BATCH_COL] == b].dropna().to_numpy() for b in batches]
        medians = [np.median(g) if len(g) else np.nan for g in groups]
        mads = [np.median(np.abs(g - m)) * 1.4826 for g, m in zip(groups, medians) if len(g)]
        pooled_mad = float(np.median(mads)) if mads else np.nan
        rng = float(np.nanmax(medians) - np.nanmin(medians)) if len(medians) > 1 else np.nan
        try:
            p = stats.kruskal(*[g for g in groups if len(g) > 1]).pvalue if sum(len(g) > 1 for g in groups) > 1 else np.nan
        except ValueError:
            p = np.nan
        loo = ev.leave_one_sample_out(df[[ev.ID_COL, ev.BATCH_COL, col]], [col], ev.ProfileDistanceScorer)
        row = {"feature": col, "view": ev.view_of(col), "pooled_within_mad": pooled_mad,
               "between_range": rng, "effect_size": rng / pooled_mad if pooled_mad and pooled_mad > 0 else np.nan,
               "kruskal_p": p, "univariate_loo_accuracy": loo.accuracy()}
        for b, m in zip(batches, medians):
            row[f"median_batch_{b}"] = m
        rows.append(row)
    return pd.DataFrame(rows).sort_values("effect_size", ascending=False).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True)
    p.add_argument("--feature-prefix", default=None)
    p.add_argument("--out", default="kpi_separation.csv")
    args = p.parse_args()
    df = pd.read_csv(args.features)
    cols = ev.feature_columns(df, args.feature_prefix)
    table = separation_table(df, cols)
    table.to_csv(args.out, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 55):
        print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
