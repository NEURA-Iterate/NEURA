from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from anode_qc.data import BSEImage, find_bse_images
from anode_qc.kpis import si_particles

from . import CLASSES
from .data import load_triplet
from .features import feature_upsample_factor, load_features, tile_starts
from .heads import build_head
from .labels import IGNORE, load_scribbles, pseudo_labels
from .train import segmentation_metrics


def sliding_window_probabilities(
    head: torch.nn.Module,
    image: np.ndarray,
    features: np.ndarray,
    tile_size: int = 224,
    overlap: int = 56,
    patch_size: int = 14,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Predict full-resolution probabilities with aligned feature tiles and overlap averaging."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected image shape (3,H,W), got {image.shape}")
    height, width = image.shape[-2:]
    upsample = feature_upsample_factor((height, width), features.shape[:2], patch_size)
    if tile_size % patch_size or overlap % patch_size:
        raise ValueError("tile_size and overlap must be multiples of patch_size")
    feature_tile_size = tile_size // patch_size * upsample
    if device is None:
        try:
            device = next(head.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device)
    probabilities = np.zeros((len(CLASSES), height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    feature_tensor = torch.from_numpy(np.ascontiguousarray(features.transpose(2, 0, 1))).float()
    for top in tile_starts(height, tile_size, overlap):
        for left in tile_starts(width, tile_size, overlap):
            actual_height = min(tile_size, height - top)
            actual_width = min(tile_size, width - left)
            image_tile = image[:, top : top + actual_height, left : left + actual_width]
            image_tensor = torch.from_numpy(np.ascontiguousarray(image_tile)).unsqueeze(0).float()
            image_tensor = torch.nn.functional.pad(
                image_tensor,
                (0, tile_size - actual_width, 0, tile_size - actual_height),
                mode="replicate",
            )

            feature_top = top // patch_size * upsample
            feature_left = left // patch_size * upsample
            feature_crop = feature_tensor[
                :,
                feature_top : feature_top + feature_tile_size,
                feature_left : feature_left + feature_tile_size,
            ]
            feature_crop = torch.nn.functional.pad(
                feature_crop,
                (
                    0,
                    feature_tile_size - feature_crop.shape[-1],
                    0,
                    feature_tile_size - feature_crop.shape[-2],
                ),
                mode="replicate",
            )
            with torch.inference_mode():
                logits = head(feature_crop.unsqueeze(0).to(device), image_tensor.to(device))
                tile_probabilities = logits.softmax(dim=1)[0, :, :actual_height, :actual_width]
            probabilities[:, top : top + actual_height, left : left + actual_width] += (
                tile_probabilities.cpu().numpy()
            )
            counts[top : top + actual_height, left : left + actual_width] += 1.0
    return probabilities / counts[None]


def compute_phase_kpis(labels: np.ndarray, scale: float = 1.0) -> dict[str, float]:
    valid = labels != IGNORE
    total = int(valid.sum())
    if total == 0:
        return {
            "si_fraction": float("nan"),
            "pore_fraction": float("nan"),
            "si_fraction_of_solids": float("nan"),
            "si_ecd_d50": float("nan"),
        }
    si = (labels == CLASSES.index("si")) & valid
    pore = (labels == CLASSES.index("pore")) & valid
    solids = valid & ~pore
    particles = si_particles(si, scale)
    d50 = float(particles["ecd"].median()) if not particles.empty else float("nan")
    return {
        "si_fraction": float(si.sum() / total),
        "pore_fraction": float(pore.sum() / total),
        "si_fraction_of_solids": float(si.sum() / max(int(solids.sum()), 1)),
        "si_ecd_d50": d50,
    }


def kpi_agreement(
    predicted: np.ndarray,
    reference: np.ndarray,
    scale: float = 1.0,
) -> dict[str, dict[str, float]]:
    predicted_kpis = compute_phase_kpis(predicted, scale)
    reference_kpis = compute_phase_kpis(reference, scale)
    result = {}
    for name, predicted_value in predicted_kpis.items():
        reference_value = reference_kpis[name]
        if np.isnan(predicted_value) and np.isnan(reference_value):
            error = 0.0
        elif np.isfinite(predicted_value) and np.isfinite(reference_value):
            error = abs(predicted_value - reference_value)
        else:
            error = float("nan")
        result[name] = {
            "predicted": predicted_value,
            "rule_based": reference_value,
            "absolute_error": float(error),
        }
    return result


def _load_labels(label_spec: str, image: BSEImage) -> np.ndarray:
    if label_spec == "pseudo":
        return pseudo_labels(image)
    label_dir = Path(label_spec)
    label_path = next(
        (label_dir / f"{image.image_id}{suffix}" for suffix in (".npy", ".png")
         if (label_dir / f"{image.image_id}{suffix}").exists()),
        None,
    )
    if label_path is None:
        raise FileNotFoundError(f"No PNG or NPY label map for {image.image_id} in {label_dir}")
    return load_scribbles(label_path)


def evaluate_checkpoint(
    images_dir: str | Path,
    features_dir: str | Path,
    labels: str,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    device: str = "auto",
    tile_size: int = 224,
    overlap: int = 56,
    scale: float = 1.0,
) -> tuple[pd.DataFrame, dict]:
    device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    head = build_head(checkpoint["head"], checkpoint["in_dim"], checkpoint["n_classes"]).to(device)
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()
    held_out = set(checkpoint.get("validation_image_ids", []))
    if not held_out:
        raise ValueError(f"{checkpoint_path} does not record held-out image IDs")
    data = {image.image_id: image for image in find_bse_images(Path(images_dir))}
    missing = sorted(held_out - set(data))
    if missing:
        raise FileNotFoundError(f"Checkpoint validation images missing from {images_dir}: {missing}")

    rows = []
    for image_id in sorted(held_out):
        image = data[image_id]
        triplet = load_triplet(image)
        features = load_features(features_dir, image_id, triplet.shape[-2:])
        target = _load_labels(labels, image)
        rule_labels = pseudo_labels(image, ignore_boundary_px=0)
        if target.shape != triplet.shape[-2:] or rule_labels.shape != triplet.shape[-2:]:
            raise ValueError(f"{image_id}: labels do not match the cropped image shape")
        probabilities = sliding_window_probabilities(
            head,
            triplet,
            features,
            tile_size=tile_size,
            overlap=overlap,
            patch_size=checkpoint.get("patch_size", 14),
            device=device,
        )
        prediction = probabilities.argmax(axis=0).astype(np.uint8)
        accuracy, mean_iou, class_ious = segmentation_metrics(prediction, target)
        kpis = kpi_agreement(prediction, rule_labels, scale)
        row = {
            "image_id": image_id,
            "batch": image.batch,
            "pixel_accuracy": accuracy,
            "mean_iou": mean_iou,
            "mean_max_probability": float(probabilities.max(axis=0).mean()),
            **{f"iou_{name}": class_ious[name] for name in CLASSES},
        }
        for name, values in kpis.items():
            row[f"pred_{name}"] = values["predicted"]
            row[f"rule_{name}"] = values["rule_based"]
            row[f"abs_error_{name}"] = values["absolute_error"]
        rows.append(row)

    frame = pd.DataFrame(rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "per_image.csv", index=False)
    summary = {
        "n_images": len(frame),
        "mean_pixel_accuracy": float(frame["pixel_accuracy"].mean()),
        "mean_iou": float(frame["mean_iou"].mean()),
        "mean_max_probability": float(frame["mean_max_probability"].mean()),
        "mean_kpi_absolute_error": {
            name: float(frame[f"abs_error_{name}"].mean())
            for name in ("si_fraction", "pore_fraction", "si_fraction_of_solids", "si_ecd_d50")
        },
    }
    lines = [
        f"Evaluated held-out images: {summary['n_images']}",
        f"Mean pixel accuracy: {summary['mean_pixel_accuracy']:.4f}",
        f"Mean IoU: {summary['mean_iou']:.4f}",
        f"Mean max-probability: {summary['mean_max_probability']:.4f}",
        "Mean absolute KPI agreement error:",
    ]
    lines.extend(f"  {name}: {value:.6g}" for name, value in summary["mean_kpi_absolute_error"].items())
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a fold checkpoint on its held-out images.")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", required=True, help="pseudo or directory of PNG/NPY label maps")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--overlap", type=int, default=56)
    parser.add_argument("--scale", type=float, default=1.0, help="pixel size used by Si particle ECD")
    args = parser.parse_args()
    _, summary = evaluate_checkpoint(
        args.images,
        args.features,
        args.labels,
        args.checkpoint,
        args.out,
        args.device,
        args.tile_size,
        args.overlap,
        args.scale,
    )
    print(summary)


if __name__ == "__main__":
    main()
