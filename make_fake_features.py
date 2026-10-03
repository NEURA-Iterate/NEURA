"""Generate a temporary feature table so evaluation.py can run before real KPIs exist.

One row per sample, keyed by sample_id. Column names follow the shared contract
``kpi__<view>__<name>__<unit>`` so the real feature table can replace this file
without changing the evaluation harness.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

KPI_COLUMNS = [
    "kpi__BSE__porosity__frac",
    "kpi__BSE__bright_phase_fraction__frac",
    "kpi__BSE__platelet_major_axis_d50__px",
    "kpi__SE__pore_d50__px",
    "kpi__SE__thickness__px",
    "kpi__InLens__edge_network_fraction__frac",
    "kpi__InLens__platelet_orientation_anisotropy__ratio",
]

# Baseline value per KPI and the shift applied for each batch. Batch 3 is the
# largest group (mirrors the real dataset: 7 / 7 / 17+ samples).
BASE = np.array([0.30, 0.05, 180.0, 25.0, 2100.0, 0.12, 1.8])
SPREAD = np.array([0.03, 0.01, 15.0, 3.0, 120.0, 0.02, 0.2])
SHIFT = {
    "1": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "2": np.array([0.04, 0.03, -10.0, 4.0, -150.0, 0.0, 0.0]),
    "3": np.array([-0.03, 0.0, 20.0, -3.0, 0.0, 0.04, 0.4]),
}


def make_features(n_per_batch: dict[str, int], n_unlabelled: int, noise: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for batch, n in n_per_batch.items():
        for _ in range(n):
            values = BASE + SHIFT[batch] + rng.normal(0.0, noise, len(BASE)) * SPREAD
            rows.append({"sample_id": f"fake_{len(rows):03d}", "batch_id": batch, **dict(zip(KPI_COLUMNS, values))})
    for _ in range(n_unlabelled):
        batch = rng.choice(list(n_per_batch))
        values = BASE + SHIFT[batch] + rng.normal(0.0, noise, len(BASE)) * SPREAD
        rows.append({"sample_id": f"fake_{len(rows):03d}", "batch_id": "", **dict(zip(KPI_COLUMNS, values))})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="features_fake.csv")
    parser.add_argument("--n1", type=int, default=7)
    parser.add_argument("--n2", type=int, default=7)
    parser.add_argument("--n3", type=int, default=17)
    parser.add_argument("--unlabelled", type=int, default=9, help="rows with blank batch_id (test samples)")
    parser.add_argument("--noise", type=float, default=1.0, help="multiplier on within-batch spread; >1.5 makes batches overlap")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    df = make_features({"1": args.n1, "2": args.n2, "3": args.n3}, args.unlabelled, args.noise, args.seed)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows ({df['batch_id'].astype(bool).sum()} labelled) to {args.out}")


if __name__ == "__main__":
    main()
