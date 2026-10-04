import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from micronet_cls.data import (
    CropDataset,
    SpatialAug,
    balanced_sampler_weights,
    detector_indices,
)
from micronet_cls.model import Classifier, Encoder, InputPrep, freeze_bn_stats, set_trainable
from micronet_cls.training_utils import (
    append_jsonl,
    atomic_torch_save,
    autocast_context,
    capture_rng_state,
    config_sha256,
    device_for_training,
    make_grad_scaler,
    restore_rng_state,
    runtime_info,
    seed_everything,
    write_json,
)
from micronet_cls.weights import download_micronet, load_micronet_encoder


CLASS_NAMES = {0: "Batch 1", 1: "Batch 2", 2: "Batch 3"}


@dataclass
class StageBConfig:
    init: str = "stage_a"
    tag: str = ""
    detectors: tuple[str, ...] | None = None
    warmup_epochs: int = 5
    finetune_epochs: int = 25
    warmup_head_lr: float = 1e-3
    encoder_lr: float = 1e-5
    finetune_head_lr: float = 1e-4
    microbatch: int = 4
    accum: int = 4
    weight_decay: float = 1e-4
    full_encoder: bool = False
    patience: int = 7
    seed: int = 42
    amp: bool = True
    input_prep: str = "none"
    rot90: bool = False
    workers: int = 0


def _sha256(path: Path):
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_init_encoder(run_dir, init, input_prep="none"):
    encoder = Encoder()
    if init in {"stage_a", "stage_a_v2"}:
        stage_dir = Path(run_dir) / init
        export_path = stage_dir / "encoder_export.pt"
        best_path = stage_dir / "best.pt"
        export = torch.load(export_path, map_location="cpu", weights_only=False)
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        export_input_prep = export.get("metadata", {}).get("config", {}).get("input_prep")
        if export_input_prep != input_prep:
            raise ValueError(
                f"Stage A export input_prep={export_input_prep!r} "
                f"does not match Stage B input_prep={input_prep!r}"
            )
        encoder_state = export["encoder"]
        best_state = best["encoder"]
        if encoder_state.keys() != best_state.keys():
            raise ValueError("Stage A export keys do not match Stage A best checkpoint")
        mismatched = [
            key for key in encoder_state if not torch.equal(encoder_state[key], best_state[key])
        ]
        if mismatched:
            raise ValueError(f"Stage A export differs from best checkpoint tensors: {mismatched[:8]}")
        encoder.load_state_dict(encoder_state, strict=True)
        return encoder, {
            "source": str(export_path),
            "all_tensors_equal_best": True,
            "tensors_verified": len(encoder_state),
            "micronet_weight_sha256": best["micronet_weight_sha256"],
        }
    if init == "micronet":
        downloaded = download_micronet("/hf")
        encoder, report = load_micronet_encoder(downloaded["path"])
        return encoder, {
            "source": downloaded["path"],
            "micronet_weight_sha256": downloaded["sha256"],
            "load_report": report,
        }
    raise ValueError(f"Unsupported Stage B initialization: {init}")


def _validation(model, prep, dataset, manifest, batch_size, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    prep.eval()
    rows = []
    crop_losses = []
    with torch.no_grad():
        for images, class_ids, row_indices in loader:
            images = images.to(device, non_blocking=True)
            labels = torch.as_tensor(class_ids, device=device)
            logits = model(prep(images))
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
            losses = F.cross_entropy(logits.float(), labels, reduction="none").cpu().numpy()
            crop_losses.extend(losses.tolist())
            for row_index, label, probability, loss in zip(
                row_indices.tolist(), class_ids.tolist(), probs, losses.tolist()
            ):
                rows.append(
                    {
                        "row_index": int(row_index),
                        "true_class": int(label),
                        "probabilities": probability,
                        "loss": float(loss),
                    }
                )
    preds = pd.DataFrame(rows).merge(
        manifest[["source_image_id", "group_id", "detector", "class_id"]],
        left_on="row_index",
        right_index=True,
        how="left",
    )
    if preds.empty:
        raise ValueError("Validation split has no crops")
    image_rows = []
    for source_image_id, group in preds.groupby("source_image_id", sort=True):
        probability = np.stack(group["probabilities"].to_list()).mean(axis=0)
        image_rows.append(
            {
                "source_image_id": source_image_id,
                "group_id": group["group_id"].iloc[0],
                "detector": group["detector"].iloc[0],
                "true_class": int(group["true_class"].iloc[0]),
                "probabilities": probability,
                "image_loss": float(group["loss"].mean()),
            }
        )
    images = pd.DataFrame(image_rows)
    group_losses = images.groupby("group_id")["image_loss"].mean()
    image_predictions = np.stack(images["probabilities"].to_list()).argmax(axis=1)
    image_true = images["true_class"].to_numpy()
    result = {
        "val_loss_crop_mean": float(np.mean(crop_losses)),
        "val_loss_image_mean": float(images["image_loss"].mean()),
        "val_loss_group_mean": float(group_losses.mean()),
        "crop_accuracy": float(
            accuracy_score(
                [row["true_class"] for row in rows],
                [np.argmax(row["probabilities"]) for row in rows],
            )
        ),
        "source_image_accuracy": float(accuracy_score(image_true, image_predictions)),
        "source_image_macro_f1": float(
            f1_score(image_true, image_predictions, labels=[0, 1, 2], average="macro", zero_division=0)
        ),
        "source_image_per_class_recall": recall_score(
            image_true, image_predictions, labels=[0, 1, 2], average=None, zero_division=0
        ).tolist(),
        "source_image_accuracy_by_detector": {
            detector: float(
                accuracy_score(group["true_class"], np.stack(group["probabilities"].to_list()).argmax(1))
            )
            for detector, group in images.groupby("detector", sort=True)
        },
        "n_validation_groups": int(images["group_id"].nunique()),
        "n_validation_images": int(len(images)),
    }
    return result


def verify_resume_checkpoint(checkpoint, config_dict, split_sha, profile_sha):
    expected_hash = config_sha256(config_dict)
    actual_hash = checkpoint.get("config_sha256")
    legacy_config = dict(config_dict)
    legacy_config.pop("tag", None)
    legacy_default_tag = not config_dict.get("tag") and actual_hash == config_sha256(legacy_config)
    if actual_hash != expected_hash and not legacy_default_tag:
        raise ValueError("Stage B resume config hash mismatch")
    if checkpoint.get("split_sha256") != split_sha:
        raise ValueError("Stage B resume split SHA mismatch")
    if checkpoint.get("profile_sha256") != profile_sha:
        raise ValueError("Stage B resume profile SHA mismatch")
    return int(checkpoint["epoch"]) + 1


def _save_checkpoint(
    model,
    optimizer,
    scaler,
    epoch,
    phase,
    config_dict,
    config_hash,
    split_sha,
    profile_sha,
    micronet_sha,
    best_metric,
    best_epoch,
    no_improvement,
    sampler_rng_state,
):
    return {
        "stage": "stage_b",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "phase": phase,
        "config": config_dict,
        "config_sha256": config_hash,
        "split_sha256": split_sha,
        "profile_sha256": profile_sha,
        "micronet_weight_sha256": micronet_sha,
        "input_prep_mode": config_dict["input_prep"],
        "rng_state": capture_rng_state(),
        "sampler_rng_state": sampler_rng_state,
        "best_val_loss_group_mean": best_metric,
        "best_epoch": best_epoch,
        "no_improvement": no_improvement,
    }


def run_stage_b(
    run_id: str,
    prepared_root: Path,
    run_dir: Path,
    split_sha: str,
    config: StageBConfig | None = None,
    resume: bool = False,
):
    config = StageBConfig() if config is None else config
    if config.init not in {"stage_a", "stage_a_v2", "micronet"}:
        raise ValueError("Stage B init must be stage_a, stage_a_v2, or micronet")
    if config.tag and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", config.tag):
        raise ValueError(f"Invalid Stage B tag: {config.tag}")
    config_dict = asdict(config)
    config_hash = config_sha256(config_dict)
    run_dir = Path(run_dir)
    stage_name = f"stage_b_{config.init}"
    if config.tag and not stage_name.endswith(f"_{config.tag}"):
        stage_name += f"_{config.tag}"
    stage_dir = run_dir / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    profile_sha = _sha256(run_dir / "profile.json")
    device = device_for_training()
    seed_everything(config.seed)

    train_manifest = pd.read_csv(Path(prepared_root) / "train" / "crop_manifest.csv")
    val_manifest = pd.read_csv(Path(prepared_root) / "val" / "crop_manifest.csv")
    train_indices = detector_indices(train_manifest, config.detectors)
    val_indices = detector_indices(val_manifest, config.detectors)
    train_rows = train_manifest.iloc[train_indices].reset_index(drop=True)

    encoder, init_report = _load_init_encoder(run_dir, config.init, config.input_prep)
    micronet_sha = init_report["micronet_weight_sha256"]
    model = Classifier(encoder).to(device)
    prep = InputPrep(config.input_prep).to(device)
    augmentation = SpatialAug(flips=True, rot90=config.rot90)
    train_dataset = CropDataset(
        Path(prepared_root) / "train" / "crops.npy",
        Path(prepared_root) / "train" / "crop_manifest.csv",
        train_indices,
        transform=augmentation,
        load_into_memory=True,
    )
    val_dataset = CropDataset(
        Path(prepared_root) / "val" / "crops.npy",
        Path(prepared_root) / "val" / "crop_manifest.csv",
        val_indices,
        load_into_memory=False,
    )
    sampler_weights = balanced_sampler_weights(train_rows)
    sampler_generator = torch.Generator().manual_seed(config.seed)
    sampler = WeightedRandomSampler(
        sampler_weights,
        num_samples=len(train_dataset),
        replacement=True,
        generator=sampler_generator,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.microbatch,
        sampler=sampler,
        num_workers=config.workers,
        pin_memory=True,
        drop_last=False,
    )

    def configure_optimizer(phase):
        if phase == "warmup":
            set_trainable(model.encoder, set())
            return torch.optim.AdamW(
                [{"params": list(model.fc.parameters()), "lr": config.warmup_head_lr}],
                weight_decay=config.weight_decay,
            )
        parts = {"all"} if config.full_encoder else {"input_conv", "layer4"}
        set_trainable(model.encoder, parts)
        return torch.optim.AdamW(
            [
                {
                    "params": [p for p in model.encoder.parameters() if p.requires_grad],
                    "lr": config.encoder_lr,
                },
                {"params": list(model.fc.parameters()), "lr": config.finetune_head_lr},
            ],
            weight_decay=config.weight_decay,
        )

    scaler = make_grad_scaler(config.amp)
    last_path = stage_dir / "last.pt"
    best_path = stage_dir / "best.pt"
    best_metric = float("inf")
    best_epoch = None
    start_epoch = 1
    last_phase = None
    optimizer = None
    no_improvement = 0
    if resume:
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        start_epoch = verify_resume_checkpoint(checkpoint, config_dict, split_sha, profile_sha)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        restore_rng_state(checkpoint["rng_state"])
        best_metric = checkpoint["best_val_loss_group_mean"]
        best_epoch = checkpoint["best_epoch"]
        no_improvement = int(checkpoint.get("no_improvement", 0))
        last_phase = checkpoint["phase"]
        scaler.load_state_dict(checkpoint["scaler_state"])
        if checkpoint.get("sampler_rng_state") is not None:
            sampler_generator.set_state(checkpoint["sampler_rng_state"])
        optimizer = configure_optimizer(last_phase)
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    metrics_path = stage_dir / "metrics.jsonl"
    total_epochs = config.warmup_epochs + config.finetune_epochs
    for epoch in range(start_epoch, total_epochs + 1):
        phase = "warmup" if epoch <= config.warmup_epochs else "finetune"
        if optimizer is None or phase != last_phase:
            optimizer = configure_optimizer(phase)
            last_phase = phase
        model.train()
        freeze_bn_stats(model.encoder)
        prep.train()
        optimizer.zero_grad(set_to_none=True)
        train_losses = []
        pending = 0
        for batch_index, (images, labels, _) in enumerate(train_loader, start=1):
            images = prep(images.to(device, non_blocking=True))
            labels = torch.as_tensor(labels, dtype=torch.long, device=device)
            with autocast_context(config.amp):
                logits = model(images)
                loss = F.cross_entropy(logits.float(), labels)
            scaler.scale(loss / config.accum).backward()
            pending += 1
            train_losses.append(float(loss.detach().float().cpu()))
            is_last = batch_index == len(train_loader)
            if pending == config.accum or is_last:
                if is_last and pending < config.accum:
                    correction = config.accum / pending
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
        validation = _validation(
            model, prep, val_dataset, val_manifest, config.microbatch, device
        )
        metric = validation["val_loss_group_mean"]
        epoch_metrics = {
            "epoch": epoch,
            "phase": phase,
            "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
            **validation,
        }
        append_jsonl(epoch_metrics, metrics_path)
        improved = np.isfinite(metric) and metric < best_metric
        if improved:
            best_metric = metric
            best_epoch = epoch
        if phase == "finetune":
            no_improvement = 0 if improved else no_improvement + 1
        checkpoint = _save_checkpoint(
            model,
            optimizer,
            scaler,
            epoch,
            phase,
            config_dict,
            config_hash,
            split_sha,
            profile_sha,
            micronet_sha,
            best_metric,
            best_epoch,
            no_improvement,
            sampler_generator.get_state(),
        )
        checkpoint["run_id"] = run_id
        atomic_torch_save(checkpoint, last_path)
        if improved:
            atomic_torch_save(checkpoint, best_path)
        print(json.dumps(epoch_metrics, sort_keys=True), flush=True)
        if phase == "finetune" and no_improvement >= config.patience:
            break

    if best_epoch is None:
        raise RuntimeError("Stage B did not produce a finite validation selection metric")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state"], strict=True)
    best_val_metrics = next(
        json.loads(line)
        for line in reversed(metrics_path.read_text(encoding="utf-8").splitlines())
        if json.loads(line)["epoch"] == best_epoch
    )
    model_card = {
        "run_id": run_id,
        "stage": stage_name,
        "init": config.init,
        "config": config_dict,
        "selected_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "split_sha256": split_sha,
        "profile_sha256": profile_sha,
        "micronet_weight_sha256": micronet_sha,
        "input_prep_mode": config.input_prep,
        "detectors": list(config.detectors) if config.detectors is not None else None,
        "class_mapping": CLASS_NAMES,
        "init_report": init_report,
        **(
            {
                "configuration": (
                    "v2 (pre-declared after v1 validation; full-encoder fine-tuning + ImageNet input normalization)"
                )
            }
            if config.tag == "v2"
            else (
                {
                    "configuration": (
                        "inlens-only direct MicroNet fine-tuning (no Stage A, no curriculum)"
                    )
                }
                if config.tag == "inlens"
                else {}
            )
        ),
        **runtime_info(),
    }
    write_json(model_card, stage_dir / "model_card.json")
    write_json(
        {
            "best_epoch": best_epoch,
            "best_val_loss_group_mean": best_metric,
            "model_card": str(stage_dir / "model_card.json"),
            **runtime_info(),
        },
        stage_dir / "summary.json",
    )
    return {
        "model": stage_name,
        "best_epoch": best_epoch,
        "best_val_loss_group_mean": best_metric,
        "model_card": model_card,
        **runtime_info(),
    }
