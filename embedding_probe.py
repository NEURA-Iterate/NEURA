"""What does a frozen embedding encode? Leave-one-sample-out ridge probe.

For each target column (material ``kpi__`` and imaging ``qa__`` measures), fit
standardise -> PCA(n_components) -> RidgeCV on the embeddings of the other
samples, predict the held-out sample, and report LOO R^2 and Spearman rho.
Everything (scaler, PCA, ridge alpha) is refit inside each fold.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import evaluation as ev


def loo_probe(X: np.ndarray, y: np.ndarray, n_components: int) -> tuple[float, float]:
    pred = np.empty_like(y)
    for i in range(len(y)):
        m = np.ones(len(y), bool)
        m[i] = False
        pipe = make_pipeline(StandardScaler(), PCA(n_components=min(n_components, m.sum() - 1)),
                             RidgeCV(alphas=np.logspace(-2, 4, 13)))
        pipe.fit(X[m], y[m])
        pred[i] = pipe.predict(X[[i]])[0]
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    rho = stats.spearmanr(y, pred).correlation
    return float(r2), float(rho)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embeddings", required=True)
    p.add_argument("--targets", required=True, help="CSV with sample_id and kpi__/qa__ columns")
    p.add_argument("--embedding-prefix", default="emb__")
    p.add_argument("--n-components", type=int, default=8)
    p.add_argument("--out", default="embedding_probe.csv")
    args = p.parse_args()
    emb = pd.read_csv(args.embeddings)
    tgt = pd.read_csv(args.targets)
    df = emb.merge(tgt.drop(columns=[c for c in tgt.columns if c == ev.BATCH_COL]), on=ev.ID_COL)
    emb_cols = [c for c in emb.columns if c.startswith(args.embedding_prefix)]
    target_cols = [c for c in tgt.columns if c.startswith(("kpi__", "qa__"))]
    rows = []
    for t in target_cols:
        ok = df[t].notna() & df[emb_cols].notna().all(axis=1)
        if ok.sum() < 8 or df.loc[ok, t].std() == 0:
            continue
        r2, rho = loo_probe(df.loc[ok, emb_cols].to_numpy(float), df.loc[ok, t].to_numpy(float), args.n_components)
        rows.append({"target": t, "kind": t.split("__")[0], "view": ev.view_of(t), "n": int(ok.sum()), "loo_r2": r2, "spearman": rho})
    table = pd.DataFrame(rows).sort_values("loo_r2", ascending=False).reset_index(drop=True)
    table.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(table.round(3).to_string(index=False))
    print("\nmedian LOO R^2 by kind:", table.groupby("kind")["loo_r2"].median().round(3).to_dict())


if __name__ == "__main__":
    main()
