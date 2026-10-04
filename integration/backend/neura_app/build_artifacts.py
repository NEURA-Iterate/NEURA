from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from anode_qc.cli import load_detectors
from anode_qc.config import Config
from anode_qc.data import find_bse_images
from anode_qc.pipeline import segment_image
from neura_uq import Measurement

from . import REPO_ROOT
from .model import BASE, BATCHES, SOURCE_KPIS, DualSourceClassifier
from .pipeline import LEGEND, map_learned_labels, render_panels

DEFAULT_ARTIFACTS = REPO_ROOT / "integration" / "artifacts"
DEFAULT_CHECKPOINT = Path("/home/ubuntu/data/runs/holdout/fusion/model.pt")
DEFAULT_METRICS = Path("/home/ubuntu/data/runs/holdout/fusion/metrics.json")
DEFAULT_IMAGES = Path("/home/ubuntu/data/neura")
DEFAULT_DINO_META = Path("/home/ubuntu/data/features/meta.json")
DEFAULT_PREDICTIONS = Path("/home/ubuntu/data/runs/holdout/predict_all")
DEFAULT_RULE_TABLE = REPO_ROOT / "anode_microstructure_qc" / "results" / "kpis" / "per_image.csv"
DEFAULT_LEARNED_TABLE = (
    REPO_ROOT
    / "bayesian_kpi_classifier"
    / "experiments"
    / "learned_masks"
    / "learned_per_image.csv"
)


def _json_number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _seconds(path: Path) -> float | None:
    try:
        value = path.read_text(encoding="utf-8").strip().splitlines()[0]
        return float(value.split("=", 1)[-1])
    except (OSError, ValueError, IndexError):
        return None


def _representative_scores(
    model: DualSourceClassifier,
    rule_table: pd.DataFrame,
    learned_table: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {batch: {} for batch in BATCHES}
    tables = {"rule": rule_table.set_index("image_id"), "learned": learned_table.set_index("image_id")}
    for image_id in rule_table["image_id"].astype(str):
        batch = str(tables["rule"].loc[image_id, "batch"])
        score = 0.0
        for source in ("rule", "learned"):
            row = tables[source].loc[image_id]
            kpis = SOURCE_KPIS[source]
            measurement = Measurement.from_sd([float(row[kpi]) for kpi in kpis])
            prediction = model.models[source].predict(measurement)
            score += 0.5 * sum(prediction.kpi_log_likelihood[batch].values())
        scores[batch][image_id] = score
    return scores


def _in_sample_correct(
    model: DualSourceClassifier,
    rule_table: pd.DataFrame,
    learned_table: pd.DataFrame,
) -> int:
    learned_by_id = learned_table.set_index("image_id")
    correct = 0
    for row in rule_table.to_dict(orient="records"):
        image_id = row["image_id"]
        batch = str(row["batch"])
        rule_prediction = model._prediction(
            model.models["rule"],
            {kpi: float(row[kpi]) for kpi in SOURCE_KPIS["rule"]},
        )
        learned_row = learned_by_id.loc[image_id]
        learned_prediction = model._prediction(
            model.models["learned"],
            {kpi: float(learned_row[kpi]) for kpi in SOURCE_KPIS["learned"]},
        )
        scores = {
            candidate: np.sqrt(
                rule_prediction.probabilities[candidate]
                * learned_prediction.probabilities[candidate]
            )
            for candidate in BATCHES
        }
        correct += max(scores, key=scores.__getitem__) == batch
    return correct


def _make_representatives(
    *,
    artifacts_dir: Path,
    images_dir: Path,
    predictions_dir: Path,
    rule_table: pd.DataFrame,
    learned_table: pd.DataFrame,
    checkpoint_metrics: dict[str, Any],
) -> tuple[dict[str, str], float]:
    started = time.perf_counter()
    classifier = DualSourceClassifier(
        rule_table,
        learned_table,
        bootstrap_replicates=1,
        seed=0,
    )
    scores = _representative_scores(classifier, rule_table, learned_table)
    sample_by_id = {sample.image_id: sample for sample in find_bse_images(images_dir)}
    cfg = Config()
    selected = {}
    for batch in BATCHES:
        image_id = max(scores[batch], key=scores[batch].__getitem__)
        image = sample_by_id[image_id]
        raw, etd, inlens = load_detectors(image, cfg)
        if etd is None or inlens is None or etd.shape != raw.shape or inlens.shape != raw.shape:
            raise ValueError(f"{image_id}: BSE, ETD, and Inlens shapes must match")
        result = segment_image(raw, cfg, etd, inlens)
        prediction = np.load(predictions_dir / f"{image_id}.npy", allow_pickle=False)
        if prediction.shape != raw.shape or prediction.dtype != np.uint8:
            raise ValueError(
                f"{image_id}: expected aligned uint8 mask with shape {raw.shape}, "
                f"got {prediction.dtype} {prediction.shape}"
            )
        learned_labels = map_learned_labels(prediction, cfg)
        etd_n = result.mm.etd_n if result.mm is not None else None
        if etd_n is None:
            raise ValueError(f"{image_id}: ETD data is needed to render crack panels")
        batch_dir = artifacts_dir / "representatives" / batch
        render_panels(raw, result.labels, learned_labels, etd_n, cfg, batch_dir)
        kpis = {}
        for source, table in (("rule", rule_table), ("learned", learned_table)):
            row = table.loc[table["image_id"] == image_id].iloc[0]
            kpis[source] = {kpi: _json_number(row[kpi]) for kpi in BASE}
        metadata = {"image_id": image_id, "batch": batch, "kpis": kpis, "legend": LEGEND}
        (batch_dir / "kpis.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        selected[batch] = image_id
    return selected, time.perf_counter() - started


def build_artifacts(
    *,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    metrics_path: Path = DEFAULT_METRICS,
    images_dir: Path = DEFAULT_IMAGES,
    dino_meta_path: Path = DEFAULT_DINO_META,
    predictions_dir: Path = DEFAULT_PREDICTIONS,
    rule_table_path: Path = DEFAULT_RULE_TABLE,
    learned_table_path: Path = DEFAULT_LEARNED_TABLE,
    artifacts_dir: Path = DEFAULT_ARTIFACTS,
) -> dict[str, Any]:
    started = time.perf_counter()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if checkpoint_metrics.get("split") != "holdout":
        raise ValueError(f"Expected a validated holdout decoder, got split={checkpoint_metrics.get('split')!r}")
    if len(checkpoint_metrics.get("train_image_ids", [])) != 25:
        raise ValueError("The reused decoder must record exactly 25 training samples")
    if len(checkpoint_metrics.get("validation_image_ids", [])) != 3:
        raise ValueError("The reused decoder must record exactly 3 validation samples")
    if len(checkpoint_metrics.get("test_image_ids", [])) != 3:
        raise ValueError("The reused decoder must record exactly 3 test samples")

    rule_table = pd.read_csv(rule_table_path)
    learned_table = pd.read_csv(learned_table_path)
    if len(rule_table) != 31 or len(learned_table) != 31:
        raise ValueError("Expected 31 rule and learned KPI rows")
    if set(rule_table["image_id"]) != set(learned_table["image_id"]):
        raise ValueError("Rule and learned KPI tables must contain the same samples")

    shutil.copy2(checkpoint_path, artifacts_dir / "fusion_holdout25.pt")
    shutil.copy2(dino_meta_path, artifacts_dir / "dino_meta.json")
    shutil.copy2(rule_table_path, artifacts_dir / "rule_per_image.csv")
    shutil.copy2(learned_table_path, artifacts_dir / "learned_per_image.csv")
    representatives, representative_seconds = _make_representatives(
        artifacts_dir=artifacts_dir,
        images_dir=images_dir,
        predictions_dir=predictions_dir,
        rule_table=rule_table,
        learned_table=learned_table,
        checkpoint_metrics=checkpoint_metrics,
    )
    all_sample_classifier = DualSourceClassifier(
        rule_table,
        learned_table,
        bootstrap_replicates=1,
        seed=0,
    )
    in_sample_correct = _in_sample_correct(all_sample_classifier, rule_table, learned_table)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source_training_metrics = checkpoint_metrics.get("metrics", {})
    build_info = {
        "commit": commit,
        "decoder": {
            "artifact": "fusion_holdout25.pt",
            "source": str(checkpoint_path),
            "description": "Validated 25-sample holdout Fusion decoder, reused without retraining.",
            "train_samples": len(checkpoint_metrics["train_image_ids"]),
            "validation_samples": len(checkpoint_metrics["validation_image_ids"]),
            "test_samples": len(checkpoint_metrics["test_image_ids"]),
            "epochs": checkpoint_metrics.get("epochs"),
            "best_epoch": source_training_metrics.get("best_epoch"),
        },
        "classifiers": {
            "fit_samples": len(rule_table),
            "covariance": "diag",
            "bootstrap_replicates": 200,
            "seed": 0,
            "kpis": {source: list(SOURCE_KPIS[source]) for source in ("rule", "learned")},
            "in_sample_correct": in_sample_correct,
            "in_sample_n": len(rule_table),
        },
        "kpi_lists": {source: list(SOURCE_KPIS[source]) for source in ("rule", "learned")},
        "representatives": representatives,
        "timings_seconds": {
            "holdout_fusion_training": _seconds(
                Path("/home/ubuntu/data/runs/holdout/fusion_wall_seconds.txt")
            ),
            "feature_extraction_all_samples": _seconds(
                Path("/home/ubuntu/data/features_extraction_wall_time.txt")
            ),
            "full_resolution_predictions_all_samples": _seconds(
                predictions_dir / "wall_time_seconds.txt"
            ),
            "representative_panels_and_fit": representative_seconds,
        },
    }
    build_info["timings_seconds"]["artifact_assembly"] = time.perf_counter() - started
    (artifacts_dir / "build_info.json").write_text(
        json.dumps(build_info, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return build_info


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Assemble compact app artifacts from validated outputs.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--dino-meta", type=Path, default=DEFAULT_DINO_META)
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--rule-table", type=Path, default=DEFAULT_RULE_TABLE)
    parser.add_argument("--learned-table", type=Path, default=DEFAULT_LEARNED_TABLE)
    parser.add_argument("--out", type=Path, default=DEFAULT_ARTIFACTS)
    args = parser.parse_args()
    info = build_artifacts(
        checkpoint_path=args.checkpoint,
        metrics_path=args.metrics,
        images_dir=args.images,
        dino_meta_path=args.dino_meta,
        predictions_dir=args.pred_dir,
        rule_table_path=args.rule_table,
        learned_table_path=args.learned_table,
        artifacts_dir=args.out,
    )
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
