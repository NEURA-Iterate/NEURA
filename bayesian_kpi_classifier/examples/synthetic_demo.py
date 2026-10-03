"""Run the classifier on a small, synthetic three-batch example."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from neura_uq import (
    BayesianStudentTClassifier,
    Measurement,
    kpi_separation,
    leave_one_out,
    rank_kpi_subsets,
)

HERE = Path(__file__).resolve().parent
KPIS = ["si_frac", "porosity", "d50", "agglom"]
TRANSFORMS = ["logit", "logit", "log", "log"]


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader), reader.fieldnames or []


def main() -> None:
    reference_rows, reference_fields = read_csv(HERE / "reference_example.csv")
    sample_rows, _ = read_csv(HERE / "sample_example.csv")
    X = np.asarray([[float(row[kpi]) for kpi in KPIS] for row in reference_rows])
    labels = [row["batch"] for row in reference_rows]
    reference_measurements = None
    if any(f"{kpi}_sd" in reference_fields for kpi in KPIS):
        reference_measurements = [
            Measurement.from_sd(
                [float(row[kpi]) for kpi in KPIS],
                [float(row.get(f"{kpi}_sd") or 0) for kpi in KPIS],
            )
            for row in reference_rows
        ]

    classifier = BayesianStudentTClassifier(
        kpis=KPIS,
        transforms=TRANSFORMS,
        covariance="diag",
        random_state=7,
    ).fit(X, labels, reference_measurements=reference_measurements)

    sample_row = sample_rows[0]
    sample = Measurement.from_sd(
        [float(sample_row[kpi]) for kpi in KPIS],
        [float(sample_row[f"{kpi}_sd"]) for kpi in KPIS],
    )
    print("In-distribution sample with measurement SD:")
    print(classifier.predict(sample).summary())

    center = np.array([float(sample_row[kpi]) for kpi in KPIS])
    run_sd = np.array([0.006, 0.012, 0.08, 0.035])
    runs = np.random.default_rng(2025).normal(center, run_sd, size=(12, len(KPIS)))
    print("\nSample represented by perturbed segmentation runs:")
    print(classifier.predict(Measurement.from_runs(runs)).summary())

    outlier = Measurement.from_sd([0.75, 0.82, 18.0, 2.8])
    print("\nOutlier sample:")
    print(classifier.predict(outlier).summary())

    loo = leave_one_out(
        X,
        labels,
        measurements=reference_measurements,
        kpis=KPIS,
        transforms=TRANSFORMS,
        covariance="diag",
    )
    print("\nLeave-one-out metrics:")
    for key in ("accuracy", "log_loss", "brier", "baseline_log_loss"):
        print(f"  {key}: {loo[key]:.4f}")

    print("\nKPI separation:")
    for kpi, score in kpi_separation(X, labels, KPIS, TRANSFORMS).items():
        print(f"  {kpi}: {score:.3f}")

    ranked = rank_kpi_subsets(
        X,
        labels,
        KPIS,
        transforms=TRANSFORMS,
        measurements=reference_measurements,
        max_size=3,
        covariance="diag",
        n_draws=500,
    )
    print("\nTop five KPI subsets by leave-one-out log-loss:")
    for result in ranked[:5]:
        print(
            f"  {', '.join(result['kpis'])}: log_loss={result['log_loss']:.4f}, "
            f"accuracy={result['accuracy']:.3f}"
        )


if __name__ == "__main__":
    main()
