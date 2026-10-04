"""Command-line interface for classifying a sample from CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .classifier import (
    BayesianStudentTClassifier,
    Measurement,
    kpi_separation,
    leave_one_out,
)


def _read_csv(path: str) -> tuple[list[str], list[dict[str, str]]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"{path} must have a CSV header")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} must contain at least one data row")
    return reader.fieldnames, rows


def _float(row: dict[str, str], column: str, *, default: float | None = None) -> float:
    value = row.get(column, "")
    if value is None or not value.strip():
        if default is not None:
            return default
        raise ValueError(f"missing value in column {column!r}")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"non-numeric value {value!r} in column {column!r}") from exc


def _matrix(rows: list[dict[str, str]], kpis: Sequence[str]) -> np.ndarray:
    return np.asarray([[_float(row, kpi) for kpi in kpis] for row in rows], dtype=float)


def _sd(row: dict[str, str], fields: Sequence[str], kpis: Sequence[str]) -> list[float]:
    return [_float(row, f"{kpi}_sd", default=0.0) for kpi in kpis]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neura-classify",
        description="Assign a KPI sample to one of the reference batches.",
    )
    parser.add_argument("--reference", required=True, help="reference-batch CSV")
    parser.add_argument("--sample", required=True, help="sample or perturbed-runs CSV")
    parser.add_argument("--kpis", help="comma-separated KPI column names")
    parser.add_argument(
        "--transforms",
        help="comma-separated transforms (identity, log, or logit), in KPI order",
    )
    parser.add_argument("--covariance", choices=("full", "diag"), default="full")
    parser.add_argument("--prior-strength", type=float, default=2.0)
    parser.add_argument(
        "--loo", action="store_true", help="also report leave-one-out metrics"
    )
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        reference_fields, reference_rows = _read_csv(args.reference)
        sample_fields, sample_rows = _read_csv(args.sample)
        if "batch" not in reference_fields:
            raise ValueError("reference CSV must contain a 'batch' column")
        if "sample_id" not in reference_fields:
            raise ValueError("reference CSV must contain a 'sample_id' column")

        if args.kpis:
            kpis = [value.strip() for value in args.kpis.split(",") if value.strip()]
            if not kpis:
                raise ValueError("--kpis must contain at least one column name")
        else:
            kpis = [
                field
                for field in reference_fields
                if field not in ("batch", "sample_id") and not field.endswith("_sd")
            ]
        if not kpis:
            raise ValueError("no KPI columns found in reference CSV")
        for field in kpis:
            if field not in reference_fields:
                raise ValueError(f"KPI {field!r} is missing from reference CSV")
            if field not in sample_fields:
                raise ValueError(f"KPI {field!r} is missing from sample CSV")

        transforms = (
            [value.strip() for value in args.transforms.split(",")]
            if args.transforms is not None
            else ["identity"] * len(kpis)
        )
        if len(transforms) != len(kpis):
            raise ValueError(f"expected {len(kpis)} transforms, got {len(transforms)}")
        if any(not value for value in transforms):
            raise ValueError("transforms must be non-empty")

        X = _matrix(reference_rows, kpis)
        labels = [row["batch"] for row in reference_rows]
        reference_has_sd = any(f"{kpi}_sd" in reference_fields for kpi in kpis)
        reference_measurements = None
        if reference_has_sd:
            reference_measurements = [
                Measurement.from_sd(
                    [_float(row, kpi) for kpi in kpis],
                    _sd(row, reference_fields, kpis),
                )
                for row in reference_rows
            ]

        if len(sample_rows) == 1:
            sample_row = sample_rows[0]
            sample_has_sd = any(f"{kpi}_sd" in sample_fields for kpi in kpis)
            measurement = Measurement.from_sd(
                [_float(sample_row, kpi) for kpi in kpis],
                _sd(sample_row, sample_fields, kpis) if sample_has_sd else None,
            )
        else:
            measurement = Measurement.from_runs(_matrix(sample_rows, kpis))

        classifier = BayesianStudentTClassifier(
            kpis=kpis,
            transforms=transforms,
            covariance=args.covariance,
            prior_strength=args.prior_strength,
        ).fit(X, labels, reference_measurements=reference_measurements)
        prediction = classifier.predict(measurement)

        loo = None
        separation = None
        if args.loo:
            loo = leave_one_out(
                X,
                labels,
                measurements=reference_measurements,
                kpis=kpis,
                transforms=transforms,
                covariance=args.covariance,
                prior_strength=args.prior_strength,
            )
            separation = kpi_separation(X, labels, kpis, transforms)

        if args.json:
            output = {
                "probabilities": prediction.probabilities,
                "pvalues": prediction.typicality_pvalue,
                "flags": {
                    "outlier": prediction.outlier,
                    "ambiguous": prediction.ambiguous,
                },
                "kpi_evidence": prediction.kpi_evidence,
            }
            if args.loo:
                output["loo"] = {
                    key: loo[key]
                    for key in ("accuracy", "log_loss", "brier", "baseline_log_loss")
                }
                output["kpi_separation"] = separation
            print(json.dumps(output, indent=2))
        else:
            print(prediction.summary())
            if args.loo:
                print("\nLeave-one-out metrics:")
                for key in ("accuracy", "log_loss", "brier", "baseline_log_loss"):
                    print(f"  {key}: {loo[key]:.4f}")
                print("KPI separation:")
                for kpi, score in separation.items():
                    print(f"  {kpi}: {score:.3f}")
        return 0
    except (OSError, ValueError) as exc:
        _parser().error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
