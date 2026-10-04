from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from anode_qc.config import Config
from anode_qc.data import BSEImage, find_bse_images
from sklearn.model_selection import StratifiedKFold
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from . import CLASSES
from .data import load_triplet
from .features import feature_upsample_factor, load_features
from .heads import build_head
from .labels import IGNORE, load_scribbles, pseudo_labels

PATCH_SIZE = 14


@dataclass
class ImageSample:
    image_id: str
    batch: str
    image: np.ndarray
    features: np.ndarray
    labels: np.ndarray


def holdout_image_split(
    image_ids: Sequence[str],
    batches: Sequence[str],
    seed: int = 0,
) -> dict[str, list[str]]:
    if len(image_ids) != len(batches):
        raise ValueError("image_ids and batches must have equal length")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("image_ids must be unique")
    groups: dict[str, list[str]] = {}
    for image_id, batch in zip(image_ids, batches, strict=True):
        groups.setdefault(batch, []).append(image_id)
    roles = {"train": [], "val": [], "test": []}
    rng = np.random.default_rng(seed)
    for batch in sorted(groups):
        ids = sorted(groups[batch])
        if len(ids) < 3:
            raise ValueError(f"Batch {batch} needs at least three images for train/val/test")
        shuffled = rng.permutation(ids).tolist()
        roles["test"].append(shuffled[0])
        roles["val"].append(shuffled[1])
        roles["train"].extend(shuffled[2:])
    return roles


def prepare_pseudo_label_cache(
    cache_root: str | Path,
    ignore_boundary_px: int = 2,
    cfg: Config | None = None,
) -> Path:
    cfg = Config() if cfg is None else cfg
    settings = {
        "source": "pseudo",
        "ignore_boundary_px": ignore_boundary_px,
        "classes": list(CLASSES),
        "anode_qc_config": json.loads(json.dumps(cfg.to_dict())),
    }
    encoded = json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()
    cache_dir = Path(cache_root) / hashlib.sha256(encoded).hexdigest()[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    settings_path = cache_dir / "settings.json"
    if settings_path.exists():
        if json.loads(settings_path.read_text(encoding="utf-8")) != settings:
            raise ValueError(f"Pseudo-label cache settings do not match {settings_path}")
    else:
        settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    return cache_dir


def load_or_create_pseudo_labels(
    image: BSEImage,
    cache_dir: str | Path | None = None,
    ignore_boundary_px: int = 2,
    cfg: Config | None = None,
) -> np.ndarray:
    cfg = Config() if cfg is None else cfg
    if cache_dir is None:
        return pseudo_labels(image, cfg, ignore_boundary_px)
    cache_path = Path(cache_dir) / f"{image.image_id}.npy"
    if cache_path.exists():
        labels = np.load(cache_path, allow_pickle=False)
        if labels.dtype != np.uint8:
            raise ValueError(f"Cached labels for {image.image_id} must be uint8, got {labels.dtype}")
        return labels
    labels = pseudo_labels(image, cfg, ignore_boundary_px).astype(np.uint8, copy=False)
    temporary_path = cache_path.with_name(f".{cache_path.stem}.{os.getpid()}.tmp")
    with temporary_path.open("wb") as stream:
        np.save(stream, labels, allow_pickle=False)
    os.replace(temporary_path, cache_path)
    return labels


def stratified_image_splits(
    image_ids: Sequence[str],
    batches: Sequence[str],
    folds: int = 5,
    seed: int = 0,
) -> list[tuple[list[str], list[str]]]:
    if len(image_ids) != len(batches):
        raise ValueError("image_ids and batches must have equal length")
    if folds < 2 or folds > len(image_ids):
        raise ValueError("folds must be between 2 and the number of images")
    counts = {batch: list(batches).count(batch) for batch in set(batches)}
    too_small = {batch: count for batch, count in counts.items() if count < folds}
    if too_small:
        raise ValueError(f"Each batch needs at least {folds} images for stratification; got {too_small}")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    ids = np.asarray(image_ids)
    strata = np.asarray(batches)
    return [
        (ids[train].tolist(), ids[validation].tolist())
        for train, validation in splitter.split(ids, strata)
    ]


def load_training_samples(
    images_dir: str | Path,
    features_dir: str | Path,
    labels: str,
    label_cache: str | Path | None = None,
    image_ids: set[str] | None = None,
) -> list[ImageSample]:
    all_images = find_bse_images(Path(images_dir))
    if image_ids is not None:
        missing = image_ids - {image.image_id for image in all_images}
        if missing:
            raise FileNotFoundError(f"Images not found under {images_dir}: {sorted(missing)}")
        images = [image for image in all_images if image.image_id in image_ids]
    else:
        images = all_images
    if not images:
        raise ValueError(f"No BSE images found under {images_dir}")
    label_dir = None if labels == "pseudo" else Path(labels)
    cache_dir = (
        prepare_pseudo_label_cache(label_cache)
        if labels == "pseudo" and label_cache is not None
        else None
    )
    samples = []
    for image in images:
        triplet = load_triplet(image)
        if label_dir is None:
            label_map = load_or_create_pseudo_labels(image, cache_dir)
        else:
            label_path = next(
                (label_dir / f"{image.image_id}{suffix}" for suffix in (".npy", ".png")
                 if (label_dir / f"{image.image_id}{suffix}").exists()),
                None,
            )
            if label_path is None:
                raise FileNotFoundError(f"No PNG or NPY label map for {image.image_id} in {label_dir}")
            label_map = load_scribbles(label_path)
        if label_map.shape != triplet.shape[-2:]:
            raise ValueError(
                f"{image.image_id}: label shape {label_map.shape} does not match image {triplet.shape[-2:]}"
            )
        feature_map = load_features(features_dir, image.image_id, triplet.shape[-2:]).astype(
            np.float16
        )
        samples.append(ImageSample(image.image_id, image.batch, triplet, feature_map, label_map))
    return samples


def aligned_crop(
    image: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    top: int,
    left: int,
    crop_size: int,
    patch_size: int = PATCH_SIZE,
    horizontal_flip: bool = False,
    vertical_flip: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if crop_size % patch_size:
        raise ValueError(f"crop_size must be a multiple of patch_size ({patch_size})")
    if top % patch_size or left % patch_size:
        raise ValueError("crop top and left must align to the feature patch grid")
    height, width = image.shape[-2:]
    if labels.shape != (height, width):
        raise ValueError(f"Label shape {labels.shape} does not match image {(height, width)}")
    upsample = feature_upsample_factor((height, width), features.shape[:2], patch_size)
    actual_height = min(crop_size, height - top)
    actual_width = min(crop_size, width - left)
    image_crop = image[:, top : top + actual_height, left : left + actual_width]
    label_crop = labels[top : top + actual_height, left : left + actual_width]
    feature_top, feature_left = top // patch_size * upsample, left // patch_size * upsample
    feature_height, feature_width = crop_size // patch_size * upsample, crop_size // patch_size * upsample
    feature_crop = features[
        feature_top : feature_top + feature_height,
        feature_left : feature_left + feature_width,
    ]
    image_crop = np.pad(
        image_crop,
        ((0, 0), (0, crop_size - actual_height), (0, crop_size - actual_width)),
        mode="edge",
    )
    label_crop = np.pad(
        label_crop,
        ((0, crop_size - actual_height), (0, crop_size - actual_width)),
        mode="constant",
        constant_values=IGNORE,
    )
    feature_crop = np.pad(
        feature_crop,
        (
            (0, feature_height - feature_crop.shape[0]),
            (0, feature_width - feature_crop.shape[1]),
            (0, 0),
        ),
        mode="edge",
    )
    if horizontal_flip:
        image_crop = image_crop[:, :, ::-1]
        feature_crop = feature_crop[:, ::-1]
        label_crop = label_crop[:, ::-1]
    if vertical_flip:
        image_crop = image_crop[:, ::-1, :]
        feature_crop = feature_crop[::-1, :]
        label_crop = label_crop[::-1, :]
    return (
        np.ascontiguousarray(image_crop),
        np.ascontiguousarray(feature_crop),
        np.ascontiguousarray(label_crop),
    )


class RandomCropDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[ImageSample],
        crop_size: int = 224,
        crops_per_image: int = 4,
        patch_size: int = PATCH_SIZE,
        seed: int = 0,
    ) -> None:
        if crop_size % patch_size:
            raise ValueError(f"crop_size must be a multiple of patch_size ({patch_size})")
        self.samples = list(samples)
        self.crop_size = crop_size
        self.crops_per_image = crops_per_image
        self.patch_size = patch_size
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.samples) * self.crops_per_image

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_index = index // self.crops_per_image
        sample = self.samples[sample_index]
        rng = np.random.default_rng(self.seed + self.epoch * max(len(self), 1) + index)
        height, width = sample.image.shape[-2:]
        top_limit = max(height - self.crop_size, 0) // self.patch_size
        left_limit = max(width - self.crop_size, 0) // self.patch_size
        top = int(rng.integers(top_limit + 1)) * self.patch_size
        left = int(rng.integers(left_limit + 1)) * self.patch_size
        image, features, labels = aligned_crop(
            sample.image,
            sample.features,
            sample.labels,
            top,
            left,
            self.crop_size,
            self.patch_size,
            horizontal_flip=bool(rng.integers(2)),
            vertical_flip=bool(rng.integers(2)),
        )
        return (
            torch.from_numpy(image).float(),
            torch.from_numpy(features.transpose(2, 0, 1)).float(),
            torch.from_numpy(labels.astype(np.int64)),
        )


def class_weights(label_maps: Sequence[np.ndarray], n_classes: int = len(CLASSES)) -> torch.Tensor:
    values = np.concatenate([labels.reshape(-1) for labels in label_maps])
    values = values[(values != IGNORE) & (values >= 0) & (values < n_classes)]
    counts = np.bincount(values, minlength=n_classes).astype(np.float64)
    weights = np.zeros(n_classes, dtype=np.float32)
    present = counts > 0
    weights[present] = 1 / np.sqrt(counts[present])
    if present.any():
        weights[present] /= weights[present].mean()
    return torch.from_numpy(weights)


def masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if not (target != IGNORE).any():
        return logits.sum() * 0
    return F.cross_entropy(logits, target.long(), weight=weights, ignore_index=IGNORE)


def segmentation_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[float, float, dict[str, float]]:
    valid = target != IGNORE
    if not valid.any():
        return float("nan"), float("nan"), {name: float("nan") for name in CLASSES}
    prediction = prediction[valid]
    target = target[valid]
    correct = (prediction == target).mean()
    ious: dict[str, float] = {}
    for class_index, name in enumerate(CLASSES):
        intersection = np.logical_and(prediction == class_index, target == class_index).sum()
        union = np.logical_or(prediction == class_index, target == class_index).sum()
        ious[name] = float(intersection / union) if union else float("nan")
    mean_iou = float(np.nanmean(list(ious.values())))
    return float(correct), mean_iou, ious


def _validation_crop(sample: ImageSample, crop_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = sample.image.shape[-2:]
    top = max((height - crop_size) // (2 * PATCH_SIZE), 0) * PATCH_SIZE
    left = max((width - crop_size) // (2 * PATCH_SIZE), 0) * PATCH_SIZE
    return aligned_crop(sample.image, sample.features, sample.labels, top, left, crop_size)


def train_fold(
    train_samples: Sequence[ImageSample],
    validation_samples: Sequence[ImageSample],
    head_name: str = "fusion",
    epochs: int = 5,
    crop_size: int = 224,
    crops_per_image: int = 4,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> tuple[torch.nn.Module, dict]:
    if not train_samples:
        raise ValueError("Training needs at least one image")
    torch.manual_seed(seed)
    device = torch.device(device)
    in_dim = int(train_samples[0].features.shape[-1])
    head = build_head(head_name, in_dim=in_dim).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1e-4)
    weights = class_weights([sample.labels for sample in train_samples]).to(device)
    dataset = RandomCropDataset(train_samples, crop_size, crops_per_image, seed=seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    best_loss = float("inf")
    best_state = None
    best_metrics: dict = {}

    for epoch in range(epochs):
        head.train()
        dataset.set_epoch(epoch)
        for images, features, targets in loader:
            images, features, targets = images.to(device), features.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_cross_entropy(head(features, images), targets, weights)
            loss.backward()
            optimizer.step()

        if not validation_samples:
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            best_metrics = {
                "last_epoch": epoch + 1,
                "train_loss": float(loss.detach().cpu()),
            }
            continue

        head.eval()
        losses, all_predictions, all_targets = [], [], []
        with torch.inference_mode():
            for sample in validation_samples:
                image, features, target = _validation_crop(sample, crop_size)
                image_tensor = torch.from_numpy(image).unsqueeze(0).float().to(device)
                feature_tensor = torch.from_numpy(features.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
                target_tensor = torch.from_numpy(target).unsqueeze(0).long().to(device)
                logits = head(feature_tensor, image_tensor)
                losses.append(float(masked_cross_entropy(logits, target_tensor, weights)))
                all_predictions.append(logits.argmax(dim=1).squeeze(0).cpu().numpy())
                all_targets.append(target)
        val_loss = float(np.mean(losses))
        joined_prediction = np.concatenate([p.reshape(-1) for p in all_predictions])
        joined_target = np.concatenate([t.reshape(-1) for t in all_targets])
        pixel_accuracy, mean_iou, class_ious = segmentation_metrics(joined_prediction, joined_target)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            best_metrics = {
                "best_epoch": epoch + 1,
                "val_loss": val_loss,
                "pixel_accuracy": pixel_accuracy,
                "mean_iou": mean_iou,
                "class_iou": class_ious,
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    head.load_state_dict(best_state)
    head.eval()
    return head, best_metrics


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small phase segmentation head on frozen features.")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", required=True, help="pseudo or directory of PNG/NPY label maps")
    parser.add_argument("--label-cache", type=Path)
    parser.add_argument("--head", choices=("linear", "fusion"), default="fusion")
    parser.add_argument("--split", choices=("kfold", "holdout", "all"), default="kfold")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--crops-per-image", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(os.cpu_count() or 1)
    args.out.mkdir(parents=True, exist_ok=True)
    device = _select_device(args.device)
    if args.split == "all":
        images = find_bse_images(args.images)
        split = {
            "train": [image.image_id for image in images],
            "val": [],
            "test": [],
        }
        (args.out / "split.json").write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
        samples = load_training_samples(
            args.images,
            args.features,
            args.labels,
            label_cache=args.label_cache,
        )
        head, metrics = train_fold(
            samples,
            [],
            head_name=args.head,
            epochs=args.epochs,
            crop_size=args.crop_size,
            crops_per_image=args.crops_per_image,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
        )
        checkpoint_path = args.out / "model.pt"
        torch.save(
            {
                "state_dict": head.state_dict(),
                "head": args.head,
                "in_dim": int(samples[0].features.shape[-1]),
                "n_classes": len(CLASSES),
                "classes": CLASSES,
                "patch_size": PATCH_SIZE,
                "crop_size": args.crop_size,
                "train_image_ids": split["train"],
                "validation_image_ids": split["val"],
                "test_image_ids": split["test"],
            },
            checkpoint_path,
        )
        summary = {
            "head": args.head,
            "split": "all",
            "epochs": args.epochs,
            "device": str(device),
            "seed": args.seed,
            "classes": CLASSES,
            "checkpoint": checkpoint_path.name,
            "train_image_ids": split["train"],
            "validation_image_ids": split["val"],
            "test_image_ids": split["test"],
            "metrics": metrics,
        }
        (args.out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"all samples: {metrics}")
        return

    if args.split == "holdout":
        images = find_bse_images(args.images)
        split = holdout_image_split(
            [image.image_id for image in images],
            [image.batch for image in images],
            seed=args.seed,
        )
        (args.out / "split.json").write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
        samples = load_training_samples(
            args.images,
            args.features,
            args.labels,
            label_cache=args.label_cache,
            image_ids=set(split["train"] + split["val"]),
        )
        by_id = {sample.image_id: sample for sample in samples}
        head, metrics = train_fold(
            [by_id[image_id] for image_id in split["train"]],
            [by_id[image_id] for image_id in split["val"]],
            head_name=args.head,
            epochs=args.epochs,
            crop_size=args.crop_size,
            crops_per_image=args.crops_per_image,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
        )
        checkpoint_path = args.out / "model.pt"
        torch.save(
            {
                "state_dict": head.state_dict(),
                "head": args.head,
                "in_dim": int(samples[0].features.shape[-1]),
                "n_classes": len(CLASSES),
                "classes": CLASSES,
                "patch_size": PATCH_SIZE,
                "crop_size": args.crop_size,
                "train_image_ids": split["train"],
                "validation_image_ids": split["val"],
                "test_image_ids": split["test"],
            },
            checkpoint_path,
        )
        summary = {
            "head": args.head,
            "split": "holdout",
            "epochs": args.epochs,
            "device": str(device),
            "seed": args.seed,
            "classes": CLASSES,
            "checkpoint": checkpoint_path.name,
            "train_image_ids": split["train"],
            "validation_image_ids": split["val"],
            "test_image_ids": split["test"],
            "metrics": metrics,
            "mean_validation_iou": metrics["mean_iou"],
        }
        (args.out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"holdout: {metrics}")
        return

    samples = load_training_samples(
        args.images,
        args.features,
        args.labels,
        label_cache=args.label_cache,
    )
    splits = stratified_image_splits(
        [sample.image_id for sample in samples],
        [sample.batch for sample in samples],
        folds=args.folds,
        seed=args.seed,
    )
    fold_metrics = []
    by_id = {sample.image_id: sample for sample in samples}
    for fold, (train_ids, validation_ids) in enumerate(splits):
        head, metrics = train_fold(
            [by_id[image_id] for image_id in train_ids],
            [by_id[image_id] for image_id in validation_ids],
            head_name=args.head,
            epochs=args.epochs,
            crop_size=args.crop_size,
            crops_per_image=args.crops_per_image,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed + fold,
        )
        checkpoint = {
            "state_dict": head.state_dict(),
            "head": args.head,
            "in_dim": int(samples[0].features.shape[-1]),
            "n_classes": len(CLASSES),
            "classes": CLASSES,
            "patch_size": PATCH_SIZE,
            "crop_size": args.crop_size,
            "validation_image_ids": validation_ids,
        }
        checkpoint_path = args.out / f"fold_{fold}.pt"
        torch.save(checkpoint, checkpoint_path)
        fold_metrics.append(
            {
                "fold": fold,
                "checkpoint": checkpoint_path.name,
                "train_image_ids": train_ids,
                "validation_image_ids": validation_ids,
                **metrics,
            }
        )
        print(f"fold {fold}: {metrics}")
    summary = {
        "head": args.head,
        "folds": args.folds,
        "epochs": args.epochs,
        "device": str(device),
        "seed": args.seed,
        "classes": CLASSES,
        "metrics": fold_metrics,
        "mean_validation_iou": float(np.mean([fold["mean_iou"] for fold in fold_metrics])),
    }
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
