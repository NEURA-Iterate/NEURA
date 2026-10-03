"""Bayesian Student-t batch classifier for image-derived KPIs with measurement uncertainty.

Model (on a transformed KPI scale, e.g. log sizes / logit fractions):

* Each batch k has KPI vectors ~ N(mu_k, Sigma_k) with a Normal-Inverse-Wishart prior.
  The prior centres Sigma_k on the pooled within-batch covariance (empirical Bayes) and
  mu_k on the grand mean, so small batches are shrunk towards shared structure.
* Integrating out (mu_k, Sigma_k) gives a multivariate Student-t posterior predictive.
* A new sample's measurement noise S (from segmentation perturbations) is integrated out
  exactly by Monte Carlo: p(x | k) = E_{z ~ N(x, S)} [ t_k(z) ].
* P(k | x) = pi_k p(x | k) / sum_j pi_j p(x | j).
* A typicality p-value per batch (Hotelling-style F test on the predictive) flags samples
  that fit none of the batches; the posterior alone always sums to 1.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy import stats
from scipy.special import logsumexp

_EPS = 1e-6
TRANSFORMS = ("identity", "log", "logit")


def _check_kinds(kinds: Sequence[str], p: int) -> list[str]:
    kinds = list(kinds)
    if len(kinds) != p:
        raise ValueError(f"expected {p} transforms, got {len(kinds)}")
    for k in kinds:
        if k not in TRANSFORMS:
            raise ValueError(f"unknown transform {k!r}; choose from {TRANSFORMS}")
    return kinds


def transform(x: np.ndarray, kinds: Sequence[str]) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = x.copy()
    for j, k in enumerate(kinds):
        if k == "log":
            out[..., j] = np.log(np.clip(x[..., j], _EPS, None))
        elif k == "logit":
            v = np.clip(x[..., j], _EPS, 1 - _EPS)
            out[..., j] = np.log(v) - np.log1p(-v)
    return out


def _transform_derivative(x: np.ndarray, kinds: Sequence[str]) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    d = np.ones_like(x)
    for j, k in enumerate(kinds):
        if k == "log":
            d[..., j] = 1.0 / np.clip(x[..., j], _EPS, None)
        elif k == "logit":
            v = np.clip(x[..., j], _EPS, 1 - _EPS)
            d[..., j] = 1.0 / (v * (1 - v))
    return d


def _psd_clip(a: np.ndarray) -> np.ndarray:
    a = 0.5 * (a + a.T)
    w, v = np.linalg.eigh(a)
    return (v * np.clip(w, 0.0, None)) @ v.T


@dataclass
class Measurement:
    """KPI vector of one sample plus its measurement covariance, on the raw KPI scale.

    Build with one of the constructors; ``cov`` may be None for a noise-free value.
    ``runs`` keeps the raw perturbed-run vectors so the transform can be applied per run.
    """

    x: np.ndarray
    cov: np.ndarray | None = None
    runs: np.ndarray | None = None

    @classmethod
    def from_sd(
        cls, x: Sequence[float], sd: Sequence[float] | None = None
    ) -> Measurement:
        x = np.asarray(x, dtype=float)
        cov = None if sd is None else np.diag(np.asarray(sd, dtype=float) ** 2)
        return cls(x=x, cov=cov)

    @classmethod
    def from_cov(cls, x: Sequence[float], cov: np.ndarray) -> Measurement:
        return cls(x=np.asarray(x, dtype=float), cov=np.asarray(cov, dtype=float))

    @classmethod
    def from_runs(cls, runs: np.ndarray) -> Measurement:
        """KPI vectors from repeated runs with perturbed segmentation settings (rows = runs)."""
        runs = np.atleast_2d(np.asarray(runs, dtype=float))
        cov = np.cov(runs, rowvar=False, ddof=1) if len(runs) > 1 else None
        return cls(
            x=runs.mean(axis=0),
            cov=None if cov is None else np.atleast_2d(cov),
            runs=runs,
        )

    def transformed(self, kinds: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """Mean and covariance on the transformed scale (per-run transform, else delta method)."""
        p = len(kinds)
        if self.runs is not None and len(self.runs) > 1:
            z = transform(self.runs, kinds)
            return z.mean(axis=0), np.atleast_2d(np.cov(z, rowvar=False, ddof=1))
        z = transform(self.x, kinds)
        if self.cov is None:
            return z, np.zeros((p, p))
        jac = _transform_derivative(self.x, kinds)
        return z, jac[:, None] * self.cov * jac[None, :]


@dataclass
class Prediction:
    classes: list
    kpis: list[str]
    probabilities: dict
    log_likelihood: dict
    typicality_pvalue: dict
    kpi_log_likelihood: dict
    predicted: object
    runner_up: object
    outlier: bool
    ambiguous: bool
    kpi_evidence: dict = field(default_factory=dict)

    def summary(self) -> str:
        probs = ", ".join(f"{c}: {self.probabilities[c]:.2f}" for c in self.classes)
        lines = [
            (
                f"Predicted batch: {self.predicted} "
                f"(P = {self.probabilities[self.predicted]:.2f}); "
                f"runner-up: {self.runner_up}"
            ),
            f"Posterior: {probs}",
            "Typicality p-values: "
            + ", ".join(f"{c}: {self.typicality_pvalue[c]:.3f}" for c in self.classes),
        ]
        if self.outlier:
            lines.append(
                "FLAG: sample is atypical for every batch (possible new/defective batch)."
            )
        if self.ambiguous:
            lines.append("FLAG: ambiguous between top batches.")
        ev = sorted(self.kpi_evidence.items(), key=lambda kv: -abs(kv[1]))
        lines.append(
            f"Per-KPI log-evidence {self.predicted} vs {self.runner_up} (+ favours {self.predicted}): "
            + ", ".join(f"{k}: {v:+.2f}" for k, v in ev)
        )
        return "\n".join(lines)


class BayesianStudentTClassifier:
    """Conjugate Normal-Inverse-Wishart classifier with measurement-noise marginalisation.

    Parameters
    ----------
    kpis : KPI names (column order of X).
    transforms : per-KPI transform, one of "identity", "log", "logit" (logit expects [0, 1]).
    covariance : "full" (correlated KPIs, multivariate t) or "diag" (independent KPIs;
        per-KPI evidence then adds up exactly to the total).
    prior_strength : extra pseudo-observations behind the prior on Sigma_k (> 0); the
        prior has nu0 = p + 1 + prior_strength degrees of freedom. Larger values shrink
        each batch covariance more towards the pooled within-batch covariance.
    prior_kappa : pseudo-observations behind the prior on mu_k (grand mean). Keep small.
    class_prior : None for uniform, "empirical" for batch frequencies, or a mapping.
    n_draws : Monte Carlo draws used to integrate the new sample's measurement noise.
    """

    def __init__(
        self,
        kpis: Sequence[str],
        transforms: Sequence[str] | None = None,
        covariance: str = "full",
        prior_strength: float = 2.0,
        prior_kappa: float = 0.01,
        class_prior: None | str | Mapping = None,
        n_draws: int = 4000,
        outlier_alpha: float = 0.01,
        ambiguous_threshold: float = 0.7,
        random_state: int = 0,
    ):
        if covariance not in ("full", "diag"):
            raise ValueError("covariance must be 'full' or 'diag'")
        if prior_strength <= 0 or prior_kappa <= 0:
            raise ValueError("prior_strength and prior_kappa must be > 0")
        self.kpis = list(kpis)
        self.p = len(self.kpis)
        self.transforms = _check_kinds(transforms or ["identity"] * self.p, self.p)
        self.covariance = covariance
        self.prior_strength = float(prior_strength)
        self.prior_kappa = float(prior_kappa)
        self.class_prior = class_prior
        self.n_draws = int(n_draws)
        self.outlier_alpha = float(outlier_alpha)
        self.ambiguous_threshold = float(ambiguous_threshold)
        self.random_state = random_state

    def fit(
        self,
        X: np.ndarray,
        labels: Sequence,
        reference_measurements: Sequence[Measurement] | None = None,
    ) -> BayesianStudentTClassifier:
        """Fit batch profiles from labelled reference samples (raw KPI scale, one row each).

        If ``reference_measurements`` is given (one per row of X), their average transformed
        noise covariance is treated as already contained in the batch spread, so only a new
        sample's *excess* noise is added at prediction time.
        """
        X = np.asarray(X, dtype=float)
        labels = np.asarray(labels)
        if X.ndim != 2 or X.shape[1] != self.p:
            raise ValueError(f"X must have shape (n, {self.p})")
        if len(labels) != len(X):
            raise ValueError("labels and X length differ")
        Z = transform(X, self.transforms)
        self.classes_ = list(dict.fromkeys(labels.tolist()))
        K, n, p = len(self.classes_), len(Z), self.p
        if n - K < 1:
            raise ValueError("need more reference samples than batches")

        scatter = np.zeros((p, p))
        for c in self.classes_:
            zc = Z[labels == c]
            d = zc - zc.mean(axis=0)
            scatter += d.T @ d
        pooled = scatter / (n - K)
        pooled += np.eye(p) * 1e-9 * max(np.trace(pooled) / p, 1e-12)
        if self.covariance == "diag":
            pooled = np.diag(np.diag(pooled))
        self.pooled_cov_ = pooled

        mu0 = Z.mean(axis=0)
        k0, s = self.prior_kappa, self.prior_strength
        nu0 = p + 1 + s
        # Psi0 = nu0 * pooled (nu0 pseudo-observations of the pooled covariance) keeps
        # held-out typicality p-values calibrated; centring the prior mean on pooled
        # instead makes them overconfident for small batches. Diag: per-KPI p = 1.
        psi0 = (nu0 if self.covariance == "full" else 2 + s) * pooled

        self.loc_, self.shape_, self.df_, self.counts_ = {}, {}, {}, {}
        for c in self.classes_:
            zc = Z[labels == c]
            nc = len(zc)
            xbar = zc.mean(axis=0)
            d = zc - xbar
            kn, nn = k0 + nc, nu0 + nc
            mun = (k0 * mu0 + nc * xbar) / kn
            dm = (xbar - mu0)[:, None]
            psin = psi0 + d.T @ d + (k0 * nc / kn) * (dm @ dm.T)
            df = nn - p + 1  # = nc + s + 2, identical for full and diag
            shape = psin * (kn + 1) / (kn * df)
            if self.covariance == "diag":
                shape = np.diag(np.diag(shape))
            self.loc_[c], self.shape_[c], self.df_[c], self.counts_[c] = (
                mun,
                shape,
                df,
                nc,
            )

        if self.class_prior is None:
            prior = {c: 1.0 / K for c in self.classes_}
        elif self.class_prior == "empirical":
            prior = {c: self.counts_[c] / n for c in self.classes_}
        else:
            tot = sum(self.class_prior[c] for c in self.classes_)
            prior = {c: self.class_prior[c] / tot for c in self.classes_}
        self.log_prior_ = {c: np.log(prior[c]) for c in self.classes_}

        self.reference_noise_ = np.zeros((p, p))
        if reference_measurements is not None:
            if len(reference_measurements) != n:
                raise ValueError("need one reference measurement per row of X")
            covs = [m.transformed(self.transforms)[1] for m in reference_measurements]
            self.reference_noise_ = np.mean(covs, axis=0)
        return self

    def _logpdf(self, c, z: np.ndarray) -> np.ndarray:
        loc, shape, df = self.loc_[c], self.shape_[c], self.df_[c]
        if self.covariance == "diag":
            sd = np.sqrt(np.diag(shape))
            return stats.t.logpdf(z, df, loc=loc, scale=sd).sum(axis=-1)
        return np.atleast_1d(
            stats.multivariate_t.logpdf(z, loc=loc, shape=shape, df=df)
        )

    def _kpi_logpdf(self, c, z: np.ndarray) -> np.ndarray:
        sd = np.sqrt(np.diag(self.shape_[c]))
        return stats.t.logpdf(z, self.df_[c], loc=self.loc_[c], scale=sd)

    def excess_noise(self, measurement: Measurement) -> tuple[np.ndarray, np.ndarray]:
        z, s_new = measurement.transformed(self.transforms)
        return z, _psd_clip(s_new - self.reference_noise_)

    def predict(self, measurement: Measurement) -> Prediction:
        z, s_eff = self.excess_noise(measurement)
        rng = np.random.default_rng(self.random_state)
        w, v = np.linalg.eigh(s_eff)
        if np.max(w) <= 1e-15:
            draws = z[None, :]
        else:
            root = v * np.sqrt(np.clip(w, 0.0, None))
            draws = z + rng.standard_normal((self.n_draws, self.p)) @ root.T
        log_m = np.log(len(draws))

        loglik, kpi_ll, pvals = {}, {}, {}
        for c in self.classes_:
            loglik[c] = float(logsumexp(self._logpdf(c, draws)) - log_m)
            per = logsumexp(self._kpi_logpdf(c, draws), axis=0) - log_m
            kpi_ll[c] = dict(zip(self.kpis, per.tolist()))
            pvals[c] = self._typicality(c, z, s_eff)

        logpost = np.array([loglik[c] + self.log_prior_[c] for c in self.classes_])
        post = np.exp(logpost - logsumexp(logpost))
        probs = dict(zip(self.classes_, post.tolist()))
        order = np.argsort(-post)
        top = self.classes_[order[0]]
        second = self.classes_[order[1]] if len(order) > 1 else top
        evidence = {k: kpi_ll[top][k] - kpi_ll[second][k] for k in self.kpis}
        return Prediction(
            classes=list(self.classes_),
            kpis=list(self.kpis),
            probabilities=probs,
            log_likelihood=loglik,
            typicality_pvalue=pvals,
            kpi_log_likelihood=kpi_ll,
            predicted=top,
            runner_up=second,
            outlier=bool(max(pvals.values()) < self.outlier_alpha),
            ambiguous=bool(probs[top] < self.ambiguous_threshold),
            kpi_evidence=evidence,
        )

    def _typicality(self, c, z: np.ndarray, s_eff: np.ndarray) -> float:
        """P-value of the squared Mahalanobis distance under the predictive (approximate).

        Uses d^2 / p ~ F(p, df) for a multivariate t, with the noise folded into the shape
        matrix by matching covariances: shape + S * (df - 2) / df.
        """
        df = self.df_[c]
        shape = self.shape_[c] + s_eff * (df - 2) / df
        d = z - self.loc_[c]
        d2 = float(d @ np.linalg.solve(shape, d))
        return float(stats.f.sf(d2 / self.p, self.p, df))


def leave_one_out(
    X: np.ndarray,
    labels: Sequence,
    measurements: Sequence[Measurement] | None = None,
    **classifier_kwargs,
) -> dict:
    """Leave-one-sample-out evaluation. Each held-out sample is predicted with its own noise.

    Returns accuracy, mean log-loss, Brier score, the uniform-guess log-loss baseline log(K),
    and per-sample predictions.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    n = len(X)
    preds, losses, briers = [], [], []
    classes = list(dict.fromkeys(labels.tolist()))
    for i in range(n):
        keep = np.arange(n) != i
        clf = BayesianStudentTClassifier(**classifier_kwargs)
        ref = (
            None
            if measurements is None
            else [measurements[j] for j in np.flatnonzero(keep)]
        )
        clf.fit(X[keep], labels[keep], reference_measurements=ref)
        m = measurements[i] if measurements is not None else Measurement.from_sd(X[i])
        pred = clf.predict(m)
        p_true = max(pred.probabilities.get(labels[i], 0.0), 1e-12)
        losses.append(-np.log(p_true))
        briers.append(
            sum(
                (pred.probabilities.get(c, 0.0) - (c == labels[i])) ** 2
                for c in classes
            )
        )
        preds.append(pred)
    acc = float(np.mean([p.predicted == y for p, y in zip(preds, labels)]))
    return {
        "accuracy": acc,
        "log_loss": float(np.mean(losses)),
        "brier": float(np.mean(briers)),
        "baseline_log_loss": float(np.log(len(classes))),
        "predictions": preds,
    }


def kpi_separation(
    X: np.ndarray, labels: Sequence, kpis: Sequence[str], transforms=None
) -> dict:
    """ANOVA F ratio per KPI on the transformed scale: between-batch / within-batch variance.

    Within-batch variance of measured reference samples already includes their measurement
    noise, so noisy KPIs are penalised automatically.
    """
    kinds = _check_kinds(transforms or ["identity"] * len(kpis), len(kpis))
    Z = transform(np.asarray(X, dtype=float), kinds)
    labels = np.asarray(labels)
    classes = list(dict.fromkeys(labels.tolist()))
    n, K = len(Z), len(classes)
    grand = Z.mean(axis=0)
    between = sum(
        (labels == c).sum() * (Z[labels == c].mean(0) - grand) ** 2 for c in classes
    )
    within = sum(
        ((Z[labels == c] - Z[labels == c].mean(0)) ** 2).sum(0) for c in classes
    )
    f = (between / (K - 1)) / (within / (n - K))
    return dict(sorted(zip(kpis, f.tolist()), key=lambda kv: -kv[1]))


def rank_kpi_subsets(
    X: np.ndarray,
    labels: Sequence,
    kpis: Sequence[str],
    transforms: Sequence[str] | None = None,
    measurements: Sequence[Measurement] | None = None,
    max_size: int | None = None,
    **classifier_kwargs,
) -> list[dict]:
    """Rank KPI subsets by leave-one-out log-loss (lower is better).

    The best score is optimistically biased because the subset is chosen on the same data.
    """
    X = np.asarray(X, dtype=float)
    kinds = _check_kinds(transforms or ["identity"] * len(kpis), len(kpis))
    results = []
    for size in range(1, (max_size or len(kpis)) + 1):
        for idx in combinations(range(len(kpis)), size):
            idx = list(idx)
            sub_m = None
            if measurements is not None:
                sub_m = [_subset_measurement(m, idx) for m in measurements]
            r = leave_one_out(
                X[:, idx],
                labels,
                sub_m,
                kpis=[kpis[j] for j in idx],
                transforms=[kinds[j] for j in idx],
                **classifier_kwargs,
            )
            results.append(
                {
                    "kpis": [kpis[j] for j in idx],
                    "log_loss": r["log_loss"],
                    "accuracy": r["accuracy"],
                    "brier": r["brier"],
                }
            )
    return sorted(results, key=lambda r: r["log_loss"])


def _subset_measurement(m: Measurement, idx: list[int]) -> Measurement:
    return Measurement(
        x=m.x[idx],
        cov=None if m.cov is None else m.cov[np.ix_(idx, idx)],
        runs=None if m.runs is None else m.runs[:, idx],
    )
