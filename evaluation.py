"""Leave-one-sample-out evaluation harness.

Input: a CSV with one row per labelled sample::

    sample_id,batch_id,kpi__BSE__porosity__frac,kpi__SE__thickness__px,...

For every labelled sample the whole sample is held out (all views, all patches
are already aggregated into its single row). Feature scaling and one profile
per batch are fitted on the remaining samples only, the held-out sample is
scored against every batch profile, and the prediction is recorded. Rows with a
blank ``batch_id`` are ignored here (they are the unseen test samples).

Outputs (in ``--out``): ``predictions.csv``, ``confusion_matrix.csv``,
``mistakes.csv``, ``summary.json``.

Scorers are pluggable (``--method``). Encoder embeddings are just another
feature CSV; an EM / likelihood model is just another scorer class.
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ID_COL = "sample_id"
BATCH_COL = "batch_id"
MAD_TO_SD = 1.4826
EPS = 1e-9

_VIEW_RE = re.compile(r"^kpi__([^_]+(?:_[^_]+)*?)__")


def view_of(column: str) -> str:
    """Return the view encoded in ``kpi__<view>__<name>__<unit>`` or ``default``."""
    m = _VIEW_RE.match(column)
    return m.group(1) if m else "default"


def feature_columns(df: pd.DataFrame, prefix: str | None = None) -> list[str]:
    cols = [c for c in df.columns if c not in (ID_COL, BATCH_COL)]
    if prefix:
        cols = [c for c in cols if c.startswith(prefix)]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        raise ValueError("no numeric feature columns found")
    return cols


def column_weights(columns: list[str], view_weights: dict[str, float] | None) -> np.ndarray:
    """Each view contributes ``view_weight`` in total, split evenly over its columns."""
    view_weights = view_weights or {}
    views = [view_of(c) for c in columns]
    counts = pd.Series(views).value_counts().to_dict()
    return np.array([view_weights.get(v, 1.0) / counts[v] for v in views], dtype=float)


class ProfileDistanceScorer:
    """Robust-scaled, view-weighted Euclidean distance to each batch's median profile."""

    name = "profile"

    def __init__(self, view_weights: dict[str, float] | None = None, clip_z: float = 5.0):
        self.view_weights = view_weights
        self.clip_z = clip_z

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ProfileDistanceScorer":
        self.columns = list(X.columns)
        values = X.to_numpy(dtype=float)
        self.center = np.nanmedian(values, axis=0)
        mad = np.nanmedian(np.abs(values - self.center), axis=0) * MAD_TO_SD
        sd = np.nanstd(values, axis=0)
        self.scale = np.where(mad > EPS, mad, np.where(sd > EPS, sd, 1.0))
        self.weights = column_weights(self.columns, self.view_weights)
        scaled = np.clip((values - self.center) / self.scale, -self.clip_z, self.clip_z)
        self.batches = sorted(y.astype(str).unique())
        self.profiles = {b: np.nanmedian(scaled[(y.astype(str) == b).to_numpy()], axis=0) for b in self.batches}
        return self

    def _scaled(self, x: pd.Series) -> np.ndarray:
        z = (x[self.columns].to_numpy(dtype=float) - self.center) / self.scale
        return np.clip(z, -self.clip_z, self.clip_z)

    def scores(self, x: pd.Series) -> dict[str, float]:
        """Lower is better."""
        z = self._scaled(x)
        return {b: float(np.sqrt(np.nansum(self.weights * (z - p) ** 2))) for b, p in self.profiles.items()}

    def contributions(self, x: pd.Series, winner: str, runner_up: str) -> pd.Series:
        """Per-feature share of (runner-up distance^2 - winner distance^2); positive favours the winner."""
        z = self._scaled(x)
        diff = self.weights * ((z - self.profiles[runner_up]) ** 2 - (z - self.profiles[winner]) ** 2)
        return pd.Series(diff, index=self.columns).sort_values(ascending=False)


class GaussianScorer(ProfileDistanceScorer):
    """Diagonal Gaussian per batch; score is the negative log-likelihood (lower is better).

    Variances are shrunk toward the pooled variance because batches have very few
    samples. This is the simplest stand-in for the optional EM branch.
    """

    name = "gaussian"

    def __init__(self, view_weights: dict[str, float] | None = None, shrinkage: float = 0.5, clip_z: float = 5.0):
        super().__init__(view_weights, clip_z)
        self.shrinkage = shrinkage

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GaussianScorer":
        super().fit(X, y)
        scaled = np.clip((X.to_numpy(dtype=float) - self.center) / self.scale, -self.clip_z, self.clip_z)
        pooled = np.nanvar(scaled, axis=0) + EPS
        self.variances = {}
        for b in self.batches:
            v = np.nanvar(scaled[(y.astype(str) == b).to_numpy()], axis=0)
            self.variances[b] = self.shrinkage * pooled + (1 - self.shrinkage) * v + EPS
        return self

    def scores(self, x: pd.Series) -> dict[str, float]:
        z = self._scaled(x)
        out = {}
        for b in self.batches:
            var = self.variances[b]
            nll = 0.5 * np.nansum(self.weights * ((z - self.profiles[b]) ** 2 / var + np.log(2 * np.pi * var)))
            out[b] = float(nll)
        return out

    def contributions(self, x: pd.Series, winner: str, runner_up: str) -> pd.Series:
        z = self._scaled(x)
        per = {}
        for b in (winner, runner_up):
            var = self.variances[b]
            per[b] = 0.5 * self.weights * ((z - self.profiles[b]) ** 2 / var + np.log(2 * np.pi * var))
        return pd.Series(per[runner_up] - per[winner], index=self.columns).sort_values(ascending=False)


SCORERS = {ProfileDistanceScorer.name: ProfileDistanceScorer, GaussianScorer.name: GaussianScorer}


@dataclass
class LooResult:
    predictions: pd.DataFrame
    batches: list[str]
    warnings: list[str] = field(default_factory=list)

    @property
    def evaluable(self) -> pd.DataFrame:
        return self.predictions[self.predictions["evaluable"]]

    def accuracy(self) -> float | None:
        ev = self.evaluable
        return float((ev["predicted_batch"] == ev["true_batch"]).mean()) if len(ev) else None

    def confusion_matrix(self) -> pd.DataFrame:
        ev = self.evaluable
        cm = pd.crosstab(ev["true_batch"], ev["predicted_batch"]).reindex(index=self.batches, columns=self.batches, fill_value=0)
        cm.index.name = "true_batch"
        cm.columns.name = "predicted_batch"
        return cm

    def mistakes(self) -> pd.DataFrame:
        ev = self.evaluable
        return ev[ev["predicted_batch"] != ev["true_batch"]].copy()


def labelled_rows(df: pd.DataFrame) -> pd.DataFrame:
    if ID_COL not in df.columns or BATCH_COL not in df.columns:
        raise ValueError(f"CSV needs '{ID_COL}' and '{BATCH_COL}' columns")
    if df[ID_COL].duplicated().any():
        dupes = df.loc[df[ID_COL].duplicated(), ID_COL].tolist()
        raise ValueError(f"duplicate sample_id values: {dupes}")
    batch = df[BATCH_COL].astype("string").str.strip()
    keep = batch.notna() & (batch != "")
    out = df[keep].copy()
    out[BATCH_COL] = batch[keep].str.replace(r"\.0$", "", regex=True)
    return out.reset_index(drop=True)


def leave_one_sample_out(
    df: pd.DataFrame,
    feature_cols: list[str],
    scorer_factory,
    min_train_per_batch: int = 2,
    review_margin: float = 0.15,
    top_k: int = 3,
) -> LooResult:
    df = labelled_rows(df)
    batches = sorted(df[BATCH_COL].unique())
    counts = df[BATCH_COL].value_counts()
    notes: list[str] = []
    for b in batches:
        if counts[b] - 1 < min_train_per_batch:
            notes.append(
                f"batch {b} has {counts[b]} labelled sample(s); holding one out leaves {counts[b] - 1} < "
                f"{min_train_per_batch}, so its samples cannot be evaluated against their own batch"
            )
    rows = []
    for i in range(len(df)):
        held = df.iloc[i]
        train = df.drop(index=i)
        train_counts = train[BATCH_COL].value_counts()
        candidates = [b for b in batches if train_counts.get(b, 0) >= min_train_per_batch]
        true_batch = str(held[BATCH_COL])
        record = {ID_COL: held[ID_COL], "true_batch": true_batch, "evaluable": true_batch in candidates and len(candidates) >= 2}
        if len(candidates) < 2:
            record.update(predicted_batch=None, margin=None, margin_rel=None, review_flag=None, top_contributors="")
            for b in batches:
                record[f"score_batch_{b}"] = np.nan
            rows.append(record)
            continue
        train = train[train[BATCH_COL].isin(candidates)]
        scorer = scorer_factory().fit(train[feature_cols], train[BATCH_COL])
        scores = scorer.scores(held)
        ranked = sorted(scores, key=scores.get)
        winner, runner_up = ranked[0], ranked[1]
        margin = scores[runner_up] - scores[winner]
        denom = abs(scores[winner]) if abs(scores[winner]) > EPS else EPS
        margin_rel = margin / denom
        contrib = scorer.contributions(held, winner, runner_up).head(top_k)
        record.update(
            predicted_batch=winner,
            runner_up_batch=runner_up,
            margin=margin,
            margin_rel=margin_rel,
            review_flag=bool(margin_rel < review_margin),
            top_contributors="; ".join(f"{c}={v:+.3f}" for c, v in contrib.items()),
        )
        for b in batches:
            record[f"score_batch_{b}"] = scores.get(b, np.nan)
        rows.append(record)
    pred = pd.DataFrame(rows)
    pred["correct"] = pred["evaluable"] & (pred["predicted_batch"] == pred["true_batch"])
    return LooResult(pred, batches, notes)


def shuffle_baseline(df: pd.DataFrame, feature_cols: list[str], scorer_factory, n_shuffles: int, seed: int, **kw) -> dict:
    """Accuracy of the same LOO procedure with batch labels permuted (what chance looks like at this n)."""
    df = labelled_rows(df)
    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(n_shuffles):
        shuffled = df.copy()
        shuffled[BATCH_COL] = rng.permutation(shuffled[BATCH_COL].to_numpy())
        acc = leave_one_sample_out(shuffled, feature_cols, scorer_factory, **kw).accuracy()
        if acc is not None:
            accs.append(acc)
    if not accs:
        return {"n_shuffles": 0}
    arr = np.array(accs)
    return {
        "n_shuffles": len(arr),
        "mean_accuracy": float(arr.mean()),
        "p95_accuracy": float(np.percentile(arr, 95)),
        "max_accuracy": float(arr.max()),
    }


def parse_view_weights(text: str | None) -> dict[str, float] | None:
    if not text:
        return None
    out = {}
    for part in text.split(","):
        k, v = part.split("=")
        out[k.strip()] = float(v)
    return out


def run(args: argparse.Namespace) -> dict:
    df = pd.read_csv(args.features)
    feature_cols = feature_columns(df, args.feature_prefix)
    view_weights = parse_view_weights(args.view_weights)
    scorer_cls = SCORERS[args.method]
    factory = lambda: scorer_cls(view_weights=view_weights)  # noqa: E731
    kw = dict(min_train_per_batch=args.min_train_per_batch, review_margin=args.review_margin, top_k=args.top_k)

    result = leave_one_sample_out(df, feature_cols, factory, **kw)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(out / "predictions.csv", index=False)
    result.confusion_matrix().to_csv(out / "confusion_matrix.csv")
    result.mistakes().to_csv(out / "mistakes.csv", index=False)

    labelled = labelled_rows(df)
    summary = {
        "features_csv": str(args.features),
        "method": args.method,
        "n_features": len(feature_cols),
        "features": feature_cols,
        "view_weights": view_weights or {v: 1.0 for v in sorted({view_of(c) for c in feature_cols})},
        "n_labelled_samples": int(len(labelled)),
        "n_unlabelled_rows_ignored": int(len(df) - len(labelled)),
        "samples_per_batch": {str(k): int(v) for k, v in labelled[BATCH_COL].value_counts().sort_index().items()},
        "min_train_per_batch": args.min_train_per_batch,
        "n_evaluable": int(result.predictions["evaluable"].sum()),
        "n_not_evaluable": int((~result.predictions["evaluable"]).sum()),
        "accuracy": result.accuracy(),
        "n_correct": int(result.predictions["correct"].sum()),
        "n_mistakes": int(len(result.mistakes())),
        "n_review_flagged": int(result.evaluable["review_flag"].fillna(False).astype(bool).sum()),
        "median_margin_rel": float(result.evaluable["margin_rel"].median()) if len(result.evaluable) else None,
        "confusion_matrix": result.confusion_matrix().to_dict(),
        "mistakes": result.mistakes()[[ID_COL, "true_batch", "predicted_batch", "margin_rel", "top_contributors"]].to_dict("records"),
        "warnings": result.warnings,
    }
    if args.shuffles > 0:
        summary["label_shuffle_baseline"] = shuffle_baseline(df, feature_cols, factory, args.shuffles, args.seed, **kw)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def print_summary(summary: dict) -> None:
    print(f"method={summary['method']}  features={summary['n_features']}  labelled samples={summary['n_labelled_samples']} "
          f"(per batch: {summary['samples_per_batch']}); unlabelled rows ignored={summary['n_unlabelled_rows_ignored']}")
    acc = summary["accuracy"]
    print(f"leave-one-sample-out accuracy: {acc:.3f}" if acc is not None else "leave-one-sample-out accuracy: n/a",
          f" on {summary['n_evaluable']} evaluable samples ({summary['n_not_evaluable']} not evaluable); "
          f"{summary['n_review_flagged']} flagged for review (small margin)")
    base = summary.get("label_shuffle_baseline")
    if base and base.get("n_shuffles"):
        print(f"label-shuffle baseline: mean {base['mean_accuracy']:.3f}, 95th pct {base['p95_accuracy']:.3f} over {base['n_shuffles']} shuffles")
    print("confusion matrix (rows=true, cols=predicted):")
    print(pd.DataFrame(summary["confusion_matrix"]).to_string())
    if summary["mistakes"]:
        print(f"mistakes ({summary['n_mistakes']}):")
        for m in summary["mistakes"]:
            print(f"  {m[ID_COL]}: true {m['true_batch']} -> predicted {m['predicted_batch']} "
                  f"(rel. margin {m['margin_rel']:.2f}); drivers: {m['top_contributors']}")
    for w in summary["warnings"]:
        print(f"WARNING: {w}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", required=True, help="CSV with sample_id, batch_id and numeric feature columns")
    p.add_argument("--out", default="eval_out", help="output directory")
    p.add_argument("--method", choices=sorted(SCORERS), default="profile")
    p.add_argument("--feature-prefix", default=None, help="only use columns starting with this prefix, e.g. kpi__ or emb__")
    p.add_argument("--view-weights", default=None, help="e.g. 'BSE=1,SE=1,InLens=0.5'; default 1 per view")
    p.add_argument("--min-train-per-batch", type=int, default=2, help="training samples a batch needs to form a profile")
    p.add_argument("--review-margin", type=float, default=0.15, help="relative margin below which a prediction is flagged")
    p.add_argument("--top-k", type=int, default=3, help="number of contributing features to report")
    p.add_argument("--shuffles", type=int, default=20, help="label-shuffle repetitions for the chance baseline (0 to skip)")
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        summary = run(args)
    print_summary(summary)


if __name__ == "__main__":
    main()
