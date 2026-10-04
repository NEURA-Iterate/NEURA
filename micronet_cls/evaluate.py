import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from micronet_cls.data import CropDataset, detector_indices
from micronet_cls.predict import load_classifier_checkpoint, predict
from micronet_cls.training_utils import runtime_info, write_json


CLASS_NAMES = ["Batch 1", "Batch 2", "Batch 3"]


def aggregate_predictions(prediction_rows, manifest: pd.DataFrame):
    predictions = pd.DataFrame(prediction_rows)
    if predictions.empty:
        raise ValueError("No crop predictions to aggregate")
    metadata = manifest.reset_index(names="row_index")
    joined = predictions.merge(
        metadata[
            [
                "row_index",
                "source_image_id",
                "group_id",
                "detector",
                "class_id",
                "batch_name",
                "y",
                "x",
            ]
        ],
        on="row_index",
        how="left",
        validate="one_to_one",
    )
    if joined["source_image_id"].isna().any():
        raise ValueError("Prediction row index did not join to the crop manifest")
    joined["true_class"] = joined["class_id"].astype(int)
    probabilities = np.stack(joined["probabilities"].to_list())
    joined["pred_class"] = probabilities.argmax(axis=1)
    joined["crop_loss"] = -np.log(
        np.clip(
            probabilities[np.arange(len(probabilities)), joined["true_class"].to_numpy()],
            1e-12,
            1,
        )
    )
    for class_id in range(3):
        joined[f"p{class_id}"] = probabilities[:, class_id]
    return joined.sort_values("row_index").reset_index(drop=True)


def aggregate_image_predictions(crop_predictions):
    records = []
    for source_image_id, group in crop_predictions.groupby("source_image_id", sort=True):
        probabilities = group[["p0", "p1", "p2"]].to_numpy(dtype=np.float64).mean(axis=0)
        records.append(
            {
                "source_image_id": source_image_id,
                "group_id": group["group_id"].iloc[0],
                "batch_name": group["batch_name"].iloc[0],
                "detector": group["detector"].iloc[0],
                "true_class": int(group["true_class"].iloc[0]),
                "pred_class": int(probabilities.argmax()),
                "p0": float(probabilities[0]),
                "p1": float(probabilities[1]),
                "p2": float(probabilities[2]),
                "n_crops": int(len(group)),
                "image_loss": float(group["crop_loss"].mean()),
            }
        )
    return pd.DataFrame(records)


def aggregate_section_predictions(image_predictions):
    records = []
    for group_id, group in image_predictions.groupby("group_id", sort=True):
        probabilities = group[["p0", "p1", "p2"]].to_numpy(dtype=np.float64).mean(axis=0)
        records.append(
            {
                "group_id": group_id,
                "batch_name": group["batch_name"].iloc[0],
                "true_class": int(group["true_class"].iloc[0]),
                "pred_class": int(probabilities.argmax()),
                "p0": float(probabilities[0]),
                "p1": float(probabilities[1]),
                "p2": float(probabilities[2]),
                "n_detector_images": int(len(group)),
            }
        )
    return pd.DataFrame(records)


def classification_metrics(true, predicted):
    precision, recall, f1, support = precision_recall_fscore_support(
        true, predicted, labels=[0, 1, 2], zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, labels=[0, 1, 2], average="macro", zero_division=0)),
        "per_class": {
            CLASS_NAMES[class_id]: {
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "f1": float(f1[class_id]),
                "support": int(support[class_id]),
            }
            for class_id in range(3)
        },
        "confusion_matrix": confusion_matrix(true, predicted, labels=[0, 1, 2]).tolist(),
        "confusion_matrix_labels": CLASS_NAMES,
    }


def bootstrap_group_metrics(image_predictions, iterations=2000, seed=42):
    rng = np.random.default_rng(seed)
    groups_by_batch = {
        batch: sorted(group["group_id"].unique())
        for batch, group in image_predictions.groupby("batch_name", sort=True)
    }
    group_rows = {
        group_id: group
        for group_id, group in image_predictions.groupby("group_id", sort=True)
    }
    accuracies = []
    macro_f1s = []
    for _ in range(iterations):
        sampled = []
        for groups in groups_by_batch.values():
            selected = rng.choice(groups, size=len(groups), replace=True)
            sampled.extend(group_rows[group_id] for group_id in selected)
        sample = pd.concat(sampled, ignore_index=True)
        metrics = classification_metrics(sample["true_class"], sample["pred_class"])
        accuracies.append(metrics["accuracy"])
        macro_f1s.append(metrics["macro_f1"])
    return {
        "iterations": iterations,
        "seed": seed,
        "stratified_by": "batch/group",
        "groups_per_batch": {batch: len(groups) for batch, groups in groups_by_batch.items()},
        "accuracy_95_ci": np.quantile(accuracies, [0.025, 0.975]).tolist(),
        "macro_f1_95_ci": np.quantile(macro_f1s, [0.025, 0.975]).tolist(),
    }


def _save_confusion(metrics, path):
    matrix = np.asarray(metrics["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=range(3),
        yticks=range(3),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted",
        ylabel="True",
    )
    for row in range(3):
        for column in range(3):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _save_previews(crop_predictions, crop_manifest, npy_path, out_dir, limit=6):
    previews = Path(out_dir) / "previews"
    previews.mkdir(parents=True, exist_ok=True)
    correct = crop_predictions[crop_predictions["true_class"] == crop_predictions["pred_class"]]
    errors = crop_predictions[crop_predictions["true_class"] != crop_predictions["pred_class"]]
    errors = errors.assign(confidence=errors[["p0", "p1", "p2"]].max(axis=1)).sort_values(
        "confidence", ascending=False
    )
    selected = list(correct.head(limit).itertuples(index=False)) + list(errors.head(limit).itertuples(index=False))
    cache = np.load(npy_path, mmap_mode="r")
    if not selected:
        return
    columns = min(4, len(selected))
    rows = (len(selected) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows), squeeze=False)
    for axis, row in zip(axes.flat, selected):
        values = cache[int(row.row_index)]
        axis.imshow(values, cmap="gray", vmin=0, vmax=1)
        axis.set_title(
            f"{CLASS_NAMES[int(row.true_class)]} → {CLASS_NAMES[int(row.pred_class)]}\n"
            f"p={max(row.p0, row.p1, row.p2):.3f}",
            color="red" if row.true_class != row.pred_class else "green",
        )
        axis.axis("off")
    for axis in list(axes.flat)[len(selected) :]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(previews / "source_examples.png", dpi=150)
    plt.close(figure)


def evaluate_checkpoint(
    model_ckpt,
    prepared_split_dir,
    train_manifest,
    out_dir,
    split="test",
    batch_size=1,
    seed=42,
    neutral_check=True,
    detectors: tuple[str, ...] | None = None,
):
    prepared_split_dir = Path(prepared_split_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_manifest = pd.read_csv(prepared_split_dir / "crop_manifest.csv")
    selected_indices = detector_indices(crop_manifest, detectors)
    train_manifest = (
        train_manifest.copy()
        if isinstance(train_manifest, pd.DataFrame)
        else pd.read_csv(train_manifest)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, prep, checkpoint, device = load_classifier_checkpoint(model_ckpt, device)
    training_detectors = checkpoint.get("config", {}).get("detectors")
    if training_detectors is not None:
        train_manifest = train_manifest[
            train_manifest["detector"].isin(training_detectors)
        ]
    order = selected_indices[
        np.random.default_rng(seed).permutation(len(selected_indices))
    ]
    dataset = CropDataset(
        prepared_split_dir / "crops.npy",
        crop_manifest,
        order,
        load_into_memory=False,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    prediction_rows = []
    with torch.no_grad():
        for images, _, row_indices in loader:
            logits = model(prep(images.to(device, non_blocking=True)))
            probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
            for row_index, probability in zip(row_indices.tolist(), probabilities):
                prediction_rows.append(
                    {
                        "row_index": int(row_index),
                        "probabilities": probability,
                        "pred_class": int(probability.argmax()),
                    }
                )
    crop_predictions = aggregate_predictions(prediction_rows, crop_manifest)
    images = aggregate_image_predictions(crop_predictions)
    sections = aggregate_section_predictions(images)

    crop_metrics = classification_metrics(
        crop_predictions["true_class"], crop_predictions["pred_class"]
    )
    image_metrics = classification_metrics(images["true_class"], images["pred_class"])
    per_detector = {
        detector: classification_metrics(group["true_class"], group["pred_class"])
        for detector, group in images.groupby("detector", sort=True)
    }
    error_table = (
        images.assign(error=images["true_class"] != images["pred_class"])
        .groupby(["batch_name", "detector"], sort=True)
        .agg(images=("error", "size"), errors=("error", "sum"))
        .reset_index()
    )
    error_table["accuracy"] = 1 - error_table["errors"] / error_table["images"]
    detector_agreement = []
    pairwise_agreements = []
    for _, group in images.groupby("group_id", sort=True):
        predictions = group["pred_class"].tolist()
        detector_agreement.append(len(group) == 3 and len(set(predictions)) == 1)
        pairwise = [
            first == second
            for first, second in itertools.combinations(predictions, 2)
        ]
        pairwise_agreements.extend(pairwise)
    agreement = {
        "all_three_detectors_agree_rate": float(np.mean(detector_agreement)) if detector_agreement else None,
        "pairwise_detector_agreement_rate": float(np.mean(pairwise_agreements)) if pairwise_agreements else None,
        "groups_with_three_detectors": int(sum(len(group) == 3 for _, group in images.groupby("group_id"))),
    }

    majority = int(train_manifest["class_id"].value_counts().idxmax())
    baseline_predictions = np.full(len(images), majority)
    baseline_metrics = {
        "majority_class": majority,
        "majority_class_name": CLASS_NAMES[majority],
        "test_image_accuracy": float(accuracy_score(images["true_class"], baseline_predictions)),
        "test_image_macro_f1": float(
            f1_score(images["true_class"], baseline_predictions, labels=[0, 1, 2], average="macro", zero_division=0)
        ),
        "train_crop_label_counts": {
            CLASS_NAMES[int(class_id)]: int(count)
            for class_id, count in train_manifest["class_id"].value_counts().items()
        },
    }
    bootstrap = bootstrap_group_metrics(images, iterations=2000, seed=42)

    crop_columns = [
        "row_index",
        "source_image_id",
        "group_id",
        "detector",
        "y",
        "x",
        "true_class",
        "pred_class",
        "p0",
        "p1",
        "p2",
    ]
    crop_predictions[crop_columns].to_csv(out_dir / "predictions_crops.csv", index=False)
    image_columns = [
        "source_image_id",
        "group_id",
        "batch_name",
        "detector",
        "true_class",
        "pred_class",
        "p0",
        "p1",
        "p2",
        "n_crops",
    ]
    images[image_columns].to_csv(out_dir / "predictions_images.csv", index=False)
    sections.to_csv(out_dir / "predictions_sections.csv", index=False)
    error_table.to_csv(out_dir / "batch_detector_errors.csv", index=False)

    _save_confusion(image_metrics, out_dir / "confusion_matrix_source_images.png")
    for detector, metrics in per_detector.items():
        _save_confusion(metrics, out_dir / f"confusion_matrix_{detector}.png")
    _save_previews(
        crop_predictions,
        crop_manifest,
        prepared_split_dir / "crops.npy",
        out_dir,
    )

    neutral_check_result = {"performed": False}
    if neutral_check and len(crop_predictions):
        rng = np.random.default_rng(seed)
        selected_indices = rng.choice(
            len(crop_predictions), size=min(5, len(crop_predictions)), replace=False
        )
        checks = []
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory(prefix="neutral-csv-") as directory:
            for neutral_index, prediction_index in enumerate(selected_indices):
                row = crop_predictions.iloc[int(prediction_index)]
                original_path = Path(crop_manifest.iloc[int(row["row_index"])]["csv_path"])
                neutral_path = Path(directory) / f"q_{neutral_index:03d}.csv"
                shutil.copyfile(original_path, neutral_path)
                neutral = predict(model_ckpt, neutral_path, mode="csv")
                expected = row[["p0", "p1", "p2"]].to_numpy(dtype=np.float64)
                actual = np.asarray(
                    [
                        neutral["probabilities"]["Batch 1"],
                        neutral["probabilities"]["Batch 2"],
                        neutral["probabilities"]["Batch 3"],
                    ]
                )
                max_difference = float(np.max(np.abs(actual - expected)))
                argmax_identical = int(actual.argmax()) == int(expected.argmax())
                tolerance = 1e-4
                if not argmax_identical or max_difference > tolerance:
                    raise AssertionError(
                        "Neutral-filename prediction mismatch: "
                        f"argmax_identical={argmax_identical}, "
                        f"max_probability_difference={max_difference}"
                    )
                checks.append(
                    {
                        "neutral_filename": neutral_path.name,
                        "max_probability_difference": max_difference,
                        "argmax_identical": argmax_identical,
                    }
                )
        neutral_check_result = {
            "performed": True,
            "checks": checks,
            "passed": True,
            "tolerance": tolerance,
        }

    group_losses = images.groupby("group_id")["image_loss"].mean()
    metrics = {
        "run_id": checkpoint.get("run_id"),
        "split": split,
        "model_checkpoint": str(model_ckpt),
        "stage": checkpoint.get("stage"),
        "detectors": list(detectors) if detectors is not None else None,
        "input_prep_mode": checkpoint.get("input_prep_mode", "none"),
        "crop_level_supplementary": crop_metrics,
        "source_image_level_primary": image_metrics,
        "source_image_by_detector": per_detector,
        "batch_detector_error_table": error_table.to_dict(orient="records"),
        "per_section_detector_agreement": agreement,
        "majority_class_baseline": baseline_metrics,
        "bootstrap_95_ci": bootstrap,
        "n_test_groups": int(images["group_id"].nunique()),
        "n_test_source_images": int(len(images)),
        "val_loss_group_mean": float(group_losses.mean()),
        "neutral_filename_check": neutral_check_result,
        **runtime_info(),
    }
    write_json(metrics, out_dir / "metrics.json")
    return metrics


def evaluate_raw_model(
    model_ckpt,
    run_id,
    prepared_split_dir,
    raw_root,
    scale,
    out_dir,
    split="test",
    batch_size=16,
    detectors: tuple[str, ...] | None = None,
):
    from micronet_cls.raw_preprocess import _read_channel_zero, normalize_single_image, tile_whole_image

    prepared_split_dir = Path(prepared_split_dir)
    raw_root = Path(raw_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_manifest = pd.read_csv(prepared_split_dir / "crop_manifest.csv")
    if detectors is not None:
        selected_indices = detector_indices(crop_manifest, detectors)
        crop_manifest = crop_manifest.iloc[selected_indices].reset_index(drop=True)
    model, prep, checkpoint, device = load_classifier_checkpoint(model_ckpt)
    model.eval()
    prep.eval()
    predictions = []
    grouped = crop_manifest.groupby("source_image_id", sort=True)
    for source_image_id, rows in grouped:
        first = rows.iloc[0]
        raw_path = raw_root / Path(first["original_relative_path"])
        raw = _read_channel_zero(raw_path)
        whole_normalized, whole_info = normalize_single_image(
            raw, scale, return_info=True
        )
        whole_crops, whole_positions = tile_whole_image(whole_normalized, 512)
        whole_by_position = dict(zip(whole_positions, whole_crops))
        actual_positions = [
            (int(row["y"]), int(row["x"])) for _, row in rows.iterrows()
        ]
        missing = [position for position in actual_positions if position not in whole_by_position]
        if missing:
            raise ValueError(f"Raw tiling mismatch for {source_image_id}: {missing[:4]}")

        cutout_crops = []
        cutout_fallbacks = []
        for y, x in actual_positions:
            cutout, cutout_info = normalize_single_image(
                raw[y : y + 512, x : x + 512],
                scale,
                return_info=True,
            )
            cutout_crops.append(np.round(cutout, 3).astype(np.float32))
            cutout_fallbacks.append(cutout_info["fallback_used"])
        variants = {
            "whole_da": (
                [whole_by_position[position] for position in actual_positions],
                [whole_info["fallback_used"]] * len(actual_positions),
            ),
            "cutout_da": (cutout_crops, cutout_fallbacks),
        }
        for variant, (crops, fallback_flags) in variants.items():
            crop_probabilities = []
            with torch.no_grad():
                for start in range(0, len(crops), batch_size):
                    images = torch.from_numpy(
                        np.stack(crops[start : start + batch_size])
                    ).unsqueeze(1)
                    probability = torch.softmax(
                        model(prep(images.to(device))).float(), dim=1
                    ).cpu().numpy()
                    crop_probabilities.extend(probability)
            mean_probability = np.stack(crop_probabilities).mean(axis=0)
            predictions.append(
                {
                    "variant": variant,
                    "source_image_id": source_image_id,
                    "group_id": first["group_id"],
                    "detector": first["detector"],
                    "batch_name": first["batch_name"],
                    "true_class": int(first["class_id"]),
                    "pred_class": int(mean_probability.argmax()),
                    "p0": float(mean_probability[0]),
                    "p1": float(mean_probability[1]),
                    "p2": float(mean_probability[2]),
                    "n_crops": len(crops),
                    "fallback_used": any(fallback_flags),
                    "fallback_crop_count": int(sum(fallback_flags)),
                }
            )
    frame = pd.DataFrame(predictions)
    results = {}
    for variant, group in frame.groupby("variant", sort=True):
        by_detector = {
            detector: classification_metrics(rows["true_class"], rows["pred_class"])
            for detector, rows in group.groupby("detector", sort=True)
        }
        results[variant] = {
            "source_image_level_primary": classification_metrics(
                group["true_class"], group["pred_class"]
            ),
            "source_image_by_detector": by_detector,
            "n_test_source_images": int(group["source_image_id"].nunique()),
            "fallback_source_images": int(group["fallback_used"].sum()),
            "fallback_crops": int(group["fallback_crop_count"].sum()),
        }
    frame.to_csv(out_dir / "predictions.csv", index=False)
    write_json(
        {
            "run_id": run_id,
            "split": split,
            "detectors": list(detectors) if detectors is not None else None,
            "model_checkpoint": str(model_ckpt),
            "variant": "DA-v1",
            "agnostic_scale": float(scale),
            "prepared_input_prep_mode": checkpoint.get("input_prep_mode", "none"),
            "results": results,
            **runtime_info(),
        },
        out_dir / "metrics.json",
    )
    return {
        "run_id": run_id,
        "split": split,
        "agnostic_scale": float(scale),
        "results": results,
        "predictions_path": str(out_dir / "predictions.csv"),
    }
