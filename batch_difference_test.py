"""Do two batches differ as populations? Energy-distance permutation test.

For each pair of batches, compute the energy distance between the two sets of
sample-level KPI vectors (each KPI robust-scaled on all samples of the pair -
this is an unsupervised, label-free test), and get its p-value by relabelling.
For 7 vs 7 all 3432 unique splits are enumerated exactly; larger pairs use
random permutations. Also reports a baseline self-split null: the energy
distance between random halves of one batch, which is the "no change" band.
Per-KPI univariate energy-distance p-values are reported too (uncorrected).
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import energy_distance as energy_1d

import evaluation as ev


def robust_scale(X: np.ndarray) -> np.ndarray:
    med = np.nanmedian(X, axis=0)
    mad = np.nanmedian(np.abs(X - med), axis=0) * 1.4826
    sd = np.nanstd(X, axis=0)
    scale = np.where(mad > 1e-12, mad, np.where(sd > 1e-12, sd, 1.0))
    return np.clip(np.nan_to_num((X - med) / scale), -5, 5)


def energy_distance(A: np.ndarray, B: np.ndarray) -> float:
    def offdiag_mean(D: np.ndarray) -> float:
        n = len(D)
        return float(D.sum() / (n * (n - 1))) if n > 1 else 0.0

    dab = cdist(A, B).mean()
    return float(2 * dab - offdiag_mean(cdist(A, A)) - offdiag_mean(cdist(B, B)))


def perm_pvalue(X: np.ndarray, n_a: int, max_perms: int, seed: int) -> tuple[float, float, int]:
    n = len(X)
    stat = energy_distance(X[:n_a], X[n_a:])
    from math import comb
    total = comb(n, n_a)
    idx = np.arange(n)
    if total <= max_perms:
        splits = (np.array(c) for c in itertools.combinations(idx, n_a))
        count = total
    else:
        rng = np.random.default_rng(seed)
        splits = (rng.permutation(idx)[:n_a] for _ in range(max_perms))
        count = max_perms
    ge = 0
    for a in splits:
        mask = np.zeros(n, bool)
        mask[a] = True
        if energy_distance(X[mask], X[~mask]) >= stat - 1e-12:
            ge += 1
    return stat, ge / count, count


def self_split_null(X: np.ndarray, n_half: int, n_draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_draws):
        p = rng.permutation(len(X))
        out.append(energy_distance(X[p[:n_half]], X[p[n_half:2 * n_half]]))
    return np.array(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True)
    p.add_argument("--feature-prefix", default="kpi__")
    p.add_argument("--max-perms", type=int, default=5000)
    p.add_argument("--out", default="batch_difference.csv")
    args = p.parse_args()
    df = ev.labelled_rows(pd.read_csv(args.features, dtype={ev.BATCH_COL: str}))
    cols = ev.feature_columns(df, args.feature_prefix)
    batches = sorted(df[ev.BATCH_COL].unique())
    rows = []
    for a, b in itertools.combinations(batches, 2):
        sub = df[df[ev.BATCH_COL].isin([a, b])]
        Xa = sub[sub[ev.BATCH_COL] == a][cols].to_numpy(float)
        Xb = sub[sub[ev.BATCH_COL] == b][cols].to_numpy(float)
        X = robust_scale(np.vstack([Xa, Xb]))
        stat, pval, count = perm_pvalue(X, len(Xa), args.max_perms, 0)
        nulls = {}
        for name, Xs in ((a, X[:len(Xa)]), (b, X[len(Xa):])):
            if len(Xs) >= 6:
                nulls[name] = self_split_null(Xs, len(Xs) // 2, 500, 0)
        null_txt = "; ".join(f"batch {k} self-split ({len(X[:len(Xa)]) // 2 if k == a else len(Xb) // 2} v same) median {np.median(v):.2f}, 95th {np.percentile(v, 95):.2f}" for k, v in nulls.items())
        rows.append(dict(pair=f"{a} vs {b}", n_a=len(Xa), n_b=len(Xb), energy_distance=round(stat, 3), p_value=round(pval, 4), n_splits=count,
                         **{f"self_split_p95_batch_{k}": round(float(np.percentile(v, 95)), 3) for k, v in nulls.items()}))
        print(f"batch {a} (n={len(Xa)}) vs {b} (n={len(Xb)}): energy distance {stat:.3f}, permutation p = {pval:.4f} over {count} splits | {null_txt}")
    pd.DataFrame(rows).to_csv(args.out, index=False)
    # per-KPI univariate, pair with smallest multivariate p and 1 vs 2
    uni = []
    for a, b in itertools.combinations(batches, 2):
        xa = df[df[ev.BATCH_COL] == a]
        xb = df[df[ev.BATCH_COL] == b]
        for c in cols:
            va, vb = xa[c].dropna().to_numpy(), xb[c].dropna().to_numpy()
            stat = energy_1d(va, vb)
            pooled = np.concatenate([va, vb])
            rng = np.random.default_rng(1)
            null = [energy_1d(*np.split(rng.permutation(pooled), [len(va)])) for _ in range(2000)]
            uni.append(dict(pair=f"{a} vs {b}", kpi=c, energy_1d=stat, p_value=float((np.array(null) >= stat - 1e-12).mean())))
    u = pd.DataFrame(uni).sort_values(["pair", "p_value"])
    u.to_csv(args.out.replace(".csv", "_per_kpi.csv"), index=False)
    for pair, g in u.groupby("pair"):
        print(f"\n{pair}: KPIs with uncorrected p < 0.05: {int((g.p_value < 0.05).sum())}/{len(g)}  (Bonferroni threshold {0.05/len(g):.4f}: {int((g.p_value < 0.05/len(g)).sum())})")
        print(g.head(5).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
