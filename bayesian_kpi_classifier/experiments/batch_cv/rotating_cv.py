from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neura_uq import (
    BayesianStudentTClassifier,
    Measurement,
    confidence_tier,
    rank_kpi_subsets,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_KPIS = REPO_ROOT / "anode_microstructure_qc/results/kpis/per_image.csv"
DEFAULT_OUT = HERE / "results"
BASE = {
    "frac_pore": "logit",
    "graphite_crack_density": "log",
    "graphite_aspect_ratio_median": "log",
    "si_contact_graphite": "logit",
    "si_fraction_of_solids": "logit",
    "si_ecd_d50": "log",
    "si_ecd_d90": "log",
    "si_cv_w256": "log",
    "si_crack_density": "log",
    "si_dispersion_index_w512": "log",
    "cbd_fraction_of_solids": "logit",
    "graphite_alignment": "log",
}
CLASSES = ("Batch_1", "Batch_2", "Batch_3")
TIERS = ("high", "medium", "review")
CALIBRATION_BINS = ((0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 17-round rotating held-out CV on the real KPI table.")
    parser.add_argument("--kpis", type=Path, default=DEFAULT_KPIS, help="Per-image KPI CSV.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Directory for experiment outputs.")
    return parser.parse_args()


def _trust_flags(value: object) -> str:
    return "" if pd.isna(value) else str(value)


def run_experiment(kpis_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    data = pd.read_csv(kpis_path)
    required = {"image_id", "batch", "trust_flags", *BASE}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"KPI table is missing required columns: {', '.join(missing)}")

    rng0 = np.random.default_rng(0)
    order = {
        batch: rng0.permutation(sorted(group.image_id)).tolist()
        for batch, group in data.groupby("batch")
    }
    n_folds = max(len(image_ids) for image_ids in order.values())
    predictions: list[dict[str, object]] = []
    selected_subsets: list[str] = []
    kpis = list(BASE)
    transforms = list(BASE.values())

    for fold in range(n_folds):
        held_out = [order[batch][fold % len(order[batch])] for batch in order]
        training = data.loc[~data.image_id.isin(held_out)]
        test = data.loc[data.image_id.isin(held_out)]
        best = rank_kpi_subsets(
            training[kpis].values,
            training.batch.values,
            kpis,
            transforms,
            max_size=3,
            covariance="diag",
        )[0]["kpis"]
        selected_subsets.append("|".join(best))
        classifier = BayesianStudentTClassifier(
            best,
            [BASE[kpi] for kpi in best],
            covariance="diag",
        ).fit(training[best].values, training.batch.values)

        for _, row in test.iterrows():
            prediction = classifier.predict(Measurement.from_sd(row[best].values.astype(float)))
            probabilities = prediction.probabilities
            top_probability = probabilities[prediction.predicted]
            predictions.append(
                {
                    "fold": fold,
                    "image_id": row.image_id,
                    "true": row.batch,
                    "pred": prediction.predicted,
                    **{f"P_{batch}": probabilities[batch] for batch in CLASSES},
                    "typ_top": prediction.typicality_pvalue[prediction.predicted],
                    "tier": confidence_tier(prediction),
                    "trust_flags": _trust_flags(row.trust_flags),
                    "kpis": "|".join(best),
                    "P_top": top_probability,
                }
            )

    prediction_frame = pd.DataFrame(predictions)
    per_image_rows: list[dict[str, object]] = []
    for image_id, group in prediction_frame.groupby("image_id", sort=False):
        mean_probabilities = {
            batch: float(group[f"P_{batch}"].mean())
            for batch in CLASSES
        }
        true_batch = str(group["true"].iloc[0])
        predicted_batch = max(CLASSES, key=mean_probabilities.__getitem__)
        tiers_seen = [tier for tier in TIERS if tier in set(group["tier"])]
        per_image_rows.append(
            {
                "image_id": image_id,
                "true": true_batch,
                **{f"P_{batch}": mean_probabilities[batch] for batch in CLASSES},
                "pred": predicted_batch,
                "correct": predicted_batch == true_batch,
                "tiers_seen": "|".join(tiers_seen),
                "trust_flags": str(group["trust_flags"].iloc[0]),
            }
        )
    return prediction_frame, pd.DataFrame(per_image_rows), [
        {"kpis": subset} for subset in selected_subsets
    ]


def _format_confusion_matrix(predictions: pd.DataFrame) -> str:
    matrix = pd.crosstab(predictions["true"], predictions["pred"]).reindex(
        index=CLASSES, columns=CLASSES, fill_value=0
    )
    matrix.index.name = None
    matrix.columns.name = None
    return matrix.to_string()


def make_summary(
    predictions: pd.DataFrame,
    per_image: pd.DataFrame,
    selected_subsets: list[dict[str, object]],
) -> str:
    n_predictions = len(predictions)
    correct = int((predictions["true"] == predictions["pred"]).sum())
    accuracy = correct / n_predictions
    true_probabilities = np.array(
        [row[f"P_{row['true']}"] for _, row in predictions.iterrows()],
        dtype=float,
    )
    log_loss = float(-np.log(np.clip(true_probabilities, 1e-300, 1.0)).mean())
    per_image_accuracy = (
        f"{int(per_image['correct'].sum())}/{len(per_image)} ({per_image['correct'].mean():.3f})"
    )
    per_batch_accuracy = {
        batch: float(
            (
                predictions.loc[predictions["true"] == batch, "true"]
                == predictions.loc[predictions["true"] == batch, "pred"]
            ).mean()
        )
        for batch in CLASSES
    }
    tier_stats = predictions.groupby("tier", sort=False).agg(
        n=("tier", "size"),
        accuracy=("true", lambda true: float((true == predictions.loc[true.index, "pred"]).mean())),
    )
    subset_counts = Counter(item["kpis"] for item in selected_subsets)
    lines = [
        "Rotating held-out CV on the real per-image KPI table",
        f"Predictions: {n_predictions} across {len(selected_subsets)} folds",
        f"Accuracy: {correct}/{n_predictions} ({accuracy:.3f})",
        f"Mean log-loss: {log_loss:.3f} (uniform baseline log(3) = {np.log(3):.3f})",
        f"Per-image averaged-probability accuracy: {per_image_accuracy}",
        "",
        "Per-batch accuracy:",
    ]
    lines.extend(f"  {batch}: {per_batch_accuracy[batch]:.3f}" for batch in CLASSES)
    lines.extend(
        [
            "",
            "Confusion matrix (rows=true, columns=pred; counts):",
            _format_confusion_matrix(predictions),
            "",
            "Calibration bins (top probability):",
            "  interval       n   mean_confidence   accuracy",
        ]
    )
    top_probability = predictions["P_top"]
    for lower, upper in CALIBRATION_BINS:
        mask = (top_probability > lower) & (top_probability <= upper)
        subset = predictions.loc[mask]
        mean_confidence = float(subset["P_top"].mean()) if len(subset) else float("nan")
        bin_accuracy = float((subset["true"] == subset["pred"]).mean()) if len(subset) else float("nan")
        lines.append(
            f"  ({lower:.1f}, {upper:.1f}]      {len(subset):2d}      "
            f"{mean_confidence:.3f}           {bin_accuracy:.3f}"
        )
    lines.extend(["", "Accuracy by confidence tier:"])
    for tier in TIERS:
        if tier in tier_stats.index:
            stats = tier_stats.loc[tier]
            lines.append(f"  {tier}: {int(stats['n'])} predictions, accuracy {stats['accuracy']:.3f}")
        else:
            lines.append(f"  {tier}: 0 predictions")
    lines.extend(["", "Selected KPI subset frequencies:"])
    lines.extend(
        f"  {subset}: {count}/{len(selected_subsets)}"
        for subset, count in sorted(subset_counts.items(), key=lambda item: (-item[1], item[0]))
    )
    return "\n".join(lines) + "\n"


def plot_per_image_probabilities(per_image: pd.DataFrame, output_path: Path) -> None:
    ordered_groups = [
        group.sort_values(
            [f"P_{group['true'].iloc[0]}", "image_id"],
            ascending=[False, True],
        )
        for _, group in per_image.groupby("true", sort=False)
    ]
    ordered = pd.concat(ordered_groups, ignore_index=True)
    positions = np.arange(len(ordered))
    bottoms = np.zeros(len(ordered))
    figure, axis = plt.subplots(figsize=(16, 5))

    for batch in CLASSES:
        values = ordered[f"P_{batch}"].to_numpy()
        axis.bar(positions, values, bottom=bottoms, label=f"P({batch})")
        bottoms += values

    for position, row in ordered.iterrows():
        if not row["correct"]:
            axis.text(position, 1.015, "x", color="red", ha="center", va="bottom", fontsize=10)
        if row["trust_flags"]:
            axis.text(position, 1.075, "*", color="red", ha="center", va="bottom", fontsize=10)

    starts = 0
    for group in ordered_groups:
        count = len(group)
        batch = str(group["true"].iloc[0])
        center = starts + (count - 1) / 2
        axis.text(center, 1.13, f"true {batch}", ha="center", va="bottom", fontweight="bold")
        starts += count
        if starts < len(ordered):
            axis.axvline(starts - 0.5, color="black", linewidth=1)

    labels = [str(image_id).removeprefix("img_") for image_id in ordered["image_id"]]
    axis.set_xticks(positions, labels, rotation=90)
    axis.set_ylim(0, 1.2)
    axis.set_ylabel("posterior probability (held out)")
    axis.set_title("Held-out batch probabilities per image (x = wrong top batch, * = trust-flagged)")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    predictions, per_image, selected_subsets = run_experiment(args.kpis)
    args.out.mkdir(parents=True, exist_ok=True)
    predictions.drop(columns="P_top").to_csv(args.out / "predictions.csv", index=False)
    per_image.to_csv(args.out / "per_image.csv", index=False)
    summary = make_summary(predictions, per_image, selected_subsets)
    (args.out / "summary.txt").write_text(summary, encoding="utf-8")
    plot_per_image_probabilities(per_image, args.out / "per_image_probabilities.png")
    print(summary, end="")


if __name__ == "__main__":
    main()
