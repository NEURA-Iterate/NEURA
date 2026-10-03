"""Compare ways of building the batch classifier on the same 17 rotating rounds.

Every choice that uses labels (subsets, weights, LDA, stage features) is made inside
each round on that round's training images only.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "batch_cv"))
from rotating_cv import BASE

from fastnb import (
    CLASSES,
    best_subset,
    class_weights,
    fit,
    fit_weights,
    kpi_loglik,
    log_loss,
    loo_loglik,
    posterior,
    rotating_rounds,
    summarise,
    transform,
)

CRACK_EXTRA = {
    "graphite_crack_count_density": "log",
    "graphite_crack_mean_len": "log",
    "graphite_crack_p90_len": "log",
    "graphite_frac_particles_cracked": "logit",
    "graphite_crack_len_per_large_particle": "log",
    "graphite_crack_frac_horizontal": "logit",
    "graphite_crack_cv_w256": "log",
    "graphite_crack_len_density_k4": "log",
    "graphite_crack_len_density_k8": "log",
}
ENGINEERED = {
    "si_cv_mean": "log",          # geometric mean of si_cv at 256/512/1024
    "crack_per_pore": "log",      # graphite crack density / pore fraction
    "si_to_cbd": "log",           # Si / binder fraction of solids
}


def add_engineered(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["si_cv_mean"] = np.exp(np.log(d[["si_cv_w256", "si_cv_w512", "si_cv_w1024"]]).mean(1))
    d["crack_per_pore"] = d["graphite_crack_density"] / d["frac_pore"]
    d["si_to_cbd"] = d["si_fraction_of_solids"] / d["cbd_fraction_of_solids"]
    return d


def load_tables(rule_csv, learned_csv, crack_csv=None) -> dict[str, tuple[pd.DataFrame, dict]]:
    rule = add_engineered(pd.read_csv(rule_csv)).sort_values("image_id").reset_index(drop=True)
    learned = add_engineered(pd.read_csv(learned_csv)).set_index("image_id").loc[rule.image_id].reset_index()
    hybrid = learned.copy()
    for c in ("graphite_crack_density", "si_crack_density"):
        hybrid[c] = rule[c].values
    hybrid = add_engineered(hybrid)
    base_eng = {**BASE, **ENGINEERED}
    tables = {
        "rule": (rule, dict(BASE)),
        "learned": (learned, dict(BASE)),
        "hybrid": (hybrid, dict(BASE)),
        "rule+eng": (rule, base_eng),
        "hybrid+eng": (hybrid, base_eng),
    }
    union = rule[["image_id", "batch"]].copy()
    upool = {}
    for tag, t in (("R", rule), ("L", learned)):
        for k, v in BASE.items():
            union[f"{tag}:{k}"] = t[k].values
            upool[f"{tag}:{k}"] = v
    tables["union"] = (union, upool)
    if crack_csv is not None and Path(crack_csv).exists():
        cf = pd.read_csv(crack_csv)
        for src, tname in (("rule", "rule"), ("learned", "learned")):
            c = cf[cf.source == src].set_index("image_id").loc[rule.image_id]
            t = tables[tname][0].copy()
            for k in CRACK_EXTRA:
                t[k] = c[k].values
            tables[f"{tname}+cracks"] = (t, {**BASE, **CRACK_EXTRA})
        h = tables["hybrid"][0].copy()
        c = cf[cf.source == "rule"].set_index("image_id").loc[rule.image_id]
        for k in CRACK_EXTRA:
            h[k] = c[k].values
        tables["hybrid+cracks"] = (h, {**BASE, **CRACK_EXTRA})
    return tables


def _subset_search(score, p, max_size, candidates=None):
    cand = list(range(p)) if candidates is None else list(candidates)
    best, best_s = None, np.inf
    for size in range(1, max_size + 1):
        for idx in combinations(cand, size):
            w = np.zeros(p)
            w[list(idx)] = 1.0
            s = score(w)
            if s < best_s - 1e-12:
                best, best_s = list(idx), s
    return best


def predict_round(strategy: str, Ztr, ytr, Zte, max_size=3):
    """Return (probs (m,3), description of what was chosen)."""
    yi = np.array([CLASSES.index(c) for c in ytr])
    p = Ztr.shape[1]
    if strategy in ("subset", "subset_bal", "subset4_bal"):
        ll = loo_loglik(Ztr, ytr)
        best, _ = best_subset(ll, yi, max_size=4 if strategy == "subset4_bal" else max_size,
                              balanced=strategy != "subset")
        w = np.zeros(p)
        w[best] = 1.0
        return posterior(kpi_loglik(Zte, fit(Ztr, ytr)), w), {"kpis": best}
    if strategy.startswith("weights"):
        l2 = float(strategy.split("_")[1])
        ll = loo_loglik(Ztr, ytr)
        w = fit_weights(ll, yi, balanced=True, l2=l2)
        return posterior(kpi_loglik(Zte, fit(Ztr, ytr)), w), {"weights": w.round(3).tolist()}
    if strategy == "two_stage":
        ll = loo_loglik(Ztr, ytr)
        b3 = (yi == 2).astype(int)
        sw = class_weights(b3, 2)

        def stage1(w):
            P3 = posterior(ll, w)[:, 2]
            pb = np.stack([1 - P3, P3], 1)
            return log_loss(pb, b3, sw)

        s1 = _subset_search(stage1, p, max_size)
        w1 = np.zeros(p)
        w1[s1] = 1.0
        P3 = posterior(kpi_loglik(Zte, fit(Ztr, ytr)), w1)[:, 2]
        m12 = yi < 2
        cls12 = CLASSES[:2]
        ll12 = loo_loglik(Ztr[m12], ytr[m12], cls12)
        s2, _ = best_subset(ll12, yi[m12], max_size=max_size, balanced=True)
        w2 = np.zeros(p)
        w2[s2] = 1.0
        P12 = posterior(kpi_loglik(Zte, fit(Ztr[m12], ytr[m12], cls12)), w2)
        probs = np.column_stack([(1 - P3) * P12[:, 0], (1 - P3) * P12[:, 1], P3])
        return probs, {"stage1": s1, "stage2": s2}
    if strategy == "lda2":
        mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-12
        lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto", n_components=2).fit((Ztr - mu) / sd, ytr)
        Ptr, Pte = lda.transform((Ztr - mu) / sd), lda.transform((Zte - mu) / sd)
        return posterior(kpi_loglik(Pte, fit(Ptr, ytr)), np.ones(2)), {"lda": True}
    raise ValueError(strategy)


def run(table: pd.DataFrame, pool: dict, strategy: str):
    kpis = list(pool)
    Z = transform(table[kpis].values, list(pool.values()))
    y = table.batch.values
    rows, chosen = [], []
    for r, trm, tem in rotating_rounds(table):
        P, info = predict_round(strategy, Z[trm], y[trm], Z[tem])
        info = {k: ([kpis[j] for j in v] if k in ("kpis", "stage1", "stage2") else v) for k, v in info.items()}
        chosen.append({"round": r, **{k: json.dumps(v) for k, v in info.items()}})
        for iid, t, pr in zip(table.image_id[tem], y[tem], P):
            rows.append({"round": r, "image_id": iid, "true": t, **{f"P_{c}": pr[i] for i, c in enumerate(CLASSES)}})
    return pd.DataFrame(rows), pd.DataFrame(chosen)


def ensemble(preds: list[pd.DataFrame], how: str) -> pd.DataFrame:
    cols = [f"P_{c}" for c in CLASSES]
    out = preds[0][["round", "image_id", "true"]].copy()
    stack = np.stack([p[cols].values for p in preds])
    if how == "mean":
        P = stack.mean(0)
    else:
        P = np.exp(np.log(np.clip(stack, 1e-12, None)).mean(0))
        P /= P.sum(1, keepdims=True)
    out[cols] = P
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", type=Path, required=True)
    ap.add_argument("--learned", type=Path, required=True)
    ap.add_argument("--cracks", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tables", nargs="*")
    ap.add_argument("--strategies", nargs="*", default=["subset", "subset_bal", "subset4_bal", "weights_0.03", "weights_0.1", "two_stage", "lda2"])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    tables = load_tables(a.rule, a.learned, a.cracks)
    names = a.tables or list(tables)
    summary, store = [], {}
    for tn in names:
        t, pool = tables[tn]
        for s in a.strategies:
            pred, chosen = run(t, pool, s)
            store[(tn, s)] = pred
            pred.to_csv(a.out / f"pred__{tn}__{s}.csv", index=False)
            chosen.to_csv(a.out / f"chosen__{tn}__{s}.csv", index=False)
            summary.append({"table": tn, "strategy": s, **summarise(pred)})
            print(tn, s, {k: round(v, 3) if isinstance(v, float) else v for k, v in summary[-1].items() if k not in ("table", "strategy", "n")}, flush=True)
    for s in a.strategies:
        if ("rule", s) in store and ("learned", s) in store:
            for how in ("mean", "geo"):
                pred = ensemble([store[("rule", s)], store[("learned", s)]], how)
                pred.to_csv(a.out / f"pred__ens_{how}__{s}.csv", index=False)
                summary.append({"table": f"ensemble_{how}(rule,learned)", "strategy": s, **summarise(pred)})
    pd.DataFrame(summary).to_csv(a.out / "summary.csv", index=False)


if __name__ == "__main__":
    main()
