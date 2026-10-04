"""Vectorised diagonal Bayesian Student-t classifier (same maths as
BayesianStudentTClassifier(covariance="diag") with point measurements) plus the
17-round rotating evaluation used in experiments/batch_cv.

With diagonal covariance each KPI contributes an independent log-likelihood term, so
per-KPI LOO terms can be cached once per round and re-used for subset search and
KPI weighting.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import logsumexp

CLASSES = ("Batch_1", "Batch_2", "Batch_3")


def transform(x: np.ndarray, kinds) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    for j, k in enumerate(kinds):
        if k == "log":
            x[:, j] = np.log(np.clip(x[:, j], 1e-12, None))
        elif k == "logit":
            p = np.clip(x[:, j], 1e-9, 1 - 1e-9)
            x[:, j] = np.log(p / (1 - p))
    return x


def fit(z: np.ndarray, y: np.ndarray, classes=CLASSES, s: float = 2.0, k0: float = 0.01):
    """Return loc (K,p), scale (K,p), df (K,) of the per-KPI Student-t predictives."""
    n, p = z.shape
    present = [c for c in classes if (y == c).any()]
    K = len(present)
    ss = np.zeros(p)
    for c in present:
        zc = z[y == c]
        ss += ((zc - zc.mean(0)) ** 2).sum(0)
    pooled = ss / (n - K)
    pooled = pooled + 1e-9 * max(pooled.mean(), 1e-12)
    mu0 = z.mean(0)
    psi0 = (2 + s) * pooled
    loc = np.full((len(classes), p), np.nan)
    scale = np.full((len(classes), p), np.nan)
    df = np.full(len(classes), np.nan)
    for i, c in enumerate(classes):
        zc = z[y == c]
        nc = len(zc)
        if nc == 0:
            continue
        xbar = zc.mean(0)
        kn = k0 + nc
        psin = psi0 + ((zc - xbar) ** 2).sum(0) + (k0 * nc / kn) * (xbar - mu0) ** 2
        df[i] = nc + s + 2
        loc[i] = (k0 * mu0 + nc * xbar) / kn
        scale[i] = np.sqrt(psin * (kn + 1) / (kn * df[i]))
    return loc, scale, df


def kpi_loglik(z: np.ndarray, params) -> np.ndarray:
    """(m, K, p) per-KPI predictive log densities; NaN for absent classes."""
    loc, scale, df = params
    return stats.t.logpdf(z[:, None, :], df[None, :, None], loc=loc[None], scale=scale[None])


def loo_loglik(z: np.ndarray, y: np.ndarray, classes=CLASSES) -> np.ndarray:
    """(n, K, p) per-KPI log densities of each training row under a fit without it."""
    out = np.empty((len(z), len(classes), z.shape[1]))
    for i in range(len(z)):
        keep = np.arange(len(z)) != i
        out[i] = kpi_loglik(z[i : i + 1], fit(z[keep], y[keep], classes))[0]
    return out


def posterior(ll: np.ndarray, weights: np.ndarray, log_prior=None) -> np.ndarray:
    """ll (m,K,p), weights (p,) -> (m,K) posterior probabilities."""
    lp = np.einsum("mkp,p->mk", np.nan_to_num(ll, nan=-1e9), weights)
    if log_prior is not None:
        lp = lp + log_prior
    return np.exp(lp - logsumexp(lp, axis=1, keepdims=True))


def log_loss(prob: np.ndarray, y_idx: np.ndarray, sample_weight=None) -> float:
    l = -np.log(np.clip(prob[np.arange(len(y_idx)), y_idx], 1e-12, None))
    return float(np.average(l, weights=sample_weight))


def class_weights(y_idx: np.ndarray, K: int) -> np.ndarray:
    counts = np.bincount(y_idx, minlength=K).astype(float)
    return 1.0 / counts[y_idx]


def best_subset(ll_loo, y_idx, max_size=3, balanced=False, candidates=None):
    p = ll_loo.shape[2]
    cand = list(range(p)) if candidates is None else list(candidates)
    sw = class_weights(y_idx, ll_loo.shape[1]) if balanced else None
    best, best_score = None, np.inf
    for size in range(1, max_size + 1):
        for idx in combinations(cand, size):
            w = np.zeros(p)
            w[list(idx)] = 1.0
            sc = log_loss(posterior(ll_loo, w), y_idx, sw)
            if sc < best_score - 1e-12:
                best, best_score = list(idx), sc
    return best, best_score


def fit_weights(ll_loo, y_idx, balanced=True, l2=0.1, init=None):
    """Non-negative per-KPI weights minimising (balanced) LOO log-loss + l2 * sum(w^2)."""
    p = ll_loo.shape[2]
    sw = class_weights(y_idx, ll_loo.shape[1]) if balanced else None
    f = lambda w: log_loss(posterior(ll_loo, w), y_idx, sw) + l2 * float(w @ w)
    x0 = np.full(p, 0.3) if init is None else init
    r = minimize(f, x0, bounds=[(0.0, 2.0)] * p, method="L-BFGS-B")
    return r.x


def rotating_rounds(data: pd.DataFrame):
    """Same held-out order as experiments/batch_cv/rotating_cv.py."""
    rng0 = np.random.default_rng(0)
    order = {b: rng0.permutation(sorted(g.image_id)).tolist() for b, g in data.groupby("batch")}
    n_rounds = max(len(v) for v in order.values())
    for r in range(n_rounds):
        held = [order[b][r % len(order[b])] for b in order]
        yield r, ~data.image_id.isin(held).values, data.image_id.isin(held).values


def summarise(pred: pd.DataFrame) -> dict:
    """pred has columns image_id, true, P_Batch_1..3."""
    P = pred[[f"P_{c}" for c in CLASSES]].values
    y = pred.true.map({c: i for i, c in enumerate(CLASSES)}).values
    top = P.argmax(1)
    per_batch = {c: float((top[y == i] == i).mean()) for i, c in enumerate(CLASSES)}
    img = pred.groupby("image_id").agg(true=("true", "first"), **{f"P_{c}": (f"P_{c}", "mean") for c in CLASSES})
    itop = img[[f"P_{c}" for c in CLASSES]].values.argmax(1)
    iy = img.true.map({c: i for i, c in enumerate(CLASSES)}).values
    img_per_batch = [float((itop[iy == i] == i).mean()) for i in range(3)]
    return {
        "correct": int((top == y).sum()),
        "n": len(y),
        "accuracy": float((top == y).mean()),
        "balanced_accuracy": float(np.mean(list(per_batch.values()))),
        "log_loss": log_loss(P, y),
        "balanced_log_loss": log_loss(P, y, class_weights(y, 3)),
        **{f"acc_{c}": v for c, v in per_batch.items()},
        "image_correct": int((itop == iy).sum()),
        "image_balanced_accuracy": float(np.mean(img_per_batch)),
    }
