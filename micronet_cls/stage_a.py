import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from micronet_cls.data import CropDataset, SpatialAug, TwoViewDataset
from micronet_cls.model import Encoder, InputPrep, Predictor, Projector, freeze_bn_stats, set_trainable
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
from micronet_cls.weights import load_micronet_encoder


@dataclass
class StageAConfig:
    tag: str = ""
    epochs: int = 10
    batch: int = 8
    encoder_lr: float = 1e-5
    head_lr: float = 1e-3
    weight_decay: float = 1e-4
    trainable_parts: tuple[str, ...] = ("input_conv", "layer4")
    flips: bool = True
    rrc_scale: tuple[float, float] | None = (0.3, 1.0)
    brightness_contrast: tuple[float, float] | None = (0.1, 0.1)
    seed: int = 42
    amp: bool = True
    input_prep: str = "none"
    workers: int = 0


def _load_prepared_manifest(prepared_root: Path, split: str):
    path = prepared_root / split / "crop_manifest.csv"
    return pd.read_csv(path)


def _profile_sha(run_dir: Path):
    import hashlib

    digest = hashlib.sha256()
    with (run_dir / "profile.json").open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _simsiam_batch(encoder, projector, predictor, prep, first, second, amp=True):
    with autocast_context(amp):
        x1 = prep(first)
        x2 = prep(second)
        z1 = projector(encoder(x1))
        z2 = projector(encoder(x2))
        p1 = predictor(z1)
        p2 = predictor(z2)
        loss = 0.5 * (
            -(torch.nn.functional.normalize(p1, dim=1) * torch.nn.functional.normalize(z2.detach(), dim=1))
            .sum(dim=1)
            .mean()
            -(torch.nn.functional.normalize(p2, dim=1) * torch.nn.functional.normalize(z1.detach(), dim=1))
            .sum(dim=1)
            .mean()
        )
    return loss, z1, z2


def _validation(encoder, projector, predictor, prep, dataset, augmentation, batch_size, config):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    encoder.eval()
    projector.eval()
    predictor.eval()
    prep.eval()
    losses = []
    embeddings = []
    with torch.no_grad():
        for crops, _, row_indices in loader:
            views_a = []
            views_b = []
            for crop, row_index in zip(crops, row_indices.tolist()):
                generator_a = torch.Generator().manual_seed(config.seed + row_index * 2)
                generator_b = torch.Generator().manual_seed(config.seed + row_index * 2 + 1)
                views_a.append(augmentation(crop.clone(), generator=generator_a))
                views_b.append(augmentation(crop.clone(), generator=generator_b))
            a = torch.stack(views_a).cuda(non_blocking=True)
            b = torch.stack(views_b).cuda(non_blocking=True)
            with autocast_context(config.amp):
                z1 = projector(encoder(prep(a)))
                z2 = projector(encoder(prep(b)))
                p1 = predictor(z1)
                p2 = predictor(z2)
                loss = 0.5 * (
                    -(torch.nn.functional.normalize(p1, dim=1) * torch.nn.functional.normalize(z2, dim=1))
                    .sum(dim=1)
                    .mean()
                    -(torch.nn.functional.normalize(p2, dim=1) * torch.nn.functional.normalize(z1, dim=1))
                    .sum(dim=1)
                    .mean()
                )
            losses.append(float(loss.float().cpu()))
            embeddings.extend(z1.float().cpu())
    z = torch.stack(embeddings)
    normalized = torch.nn.functional.normalize(z, dim=1)
    z_std = float(normalized.std(dim=0, unbiased=False).mean())
    if len(normalized) > 1:
        cosine = normalized @ normalized.T
        mask = ~torch.eye(len(cosine), dtype=torch.bool)
        offdiag_cos = float(cosine[mask].mean())
    else:
        offdiag_cos = 1.0
    return {
        "val_loss": float(np.mean(losses)) if losses else float("nan"),
        "z_std": z_std,
        "offdiag_cos": offdiag_cos,
    }


def _checkpoint(
    epoch,
    encoder,
    projector,
    predictor,
    optimizer,
    scaler,
    config,
    config_hash,
    split_sha,
    profile_sha,
    micronet_sha,
    best_loss,
    best_epoch,
    loader_rng_state,
):
    return {
        "stage": "stage_a",
        "epoch": epoch,
        "encoder": encoder.state_dict(),
        "projector": projector.state_dict(),
        "predictor": predictor.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "rng_state": capture_rng_state(),
        "loader_rng_state": loader_rng_state,
        "config": config,
        "config_sha256": config_hash,
        "split_sha256": split_sha,
        "profile_sha256": profile_sha,
        "micronet_weight_sha256": micronet_sha,
        "best_val_loss": best_loss,
        "best_epoch": best_epoch,
    }


def run_stage_a(
    run_id: str,
    prepared_root: Path,
    run_dir: Path,
    split_sha: str,
    micronet_path: Path,
    micronet_sha: str,
    config: StageAConfig | None = None,
    resume: bool = False,
):
    config = StageAConfig() if config is None else config
    config_dict = asdict(config)
    config_hash = config_sha256(config_dict)
    run_root = Path(run_dir)
    stage_name = f"stage_a_{config.tag}" if config.tag else "stage_a"
    run_dir = run_root / stage_name
    run_dir.mkdir(parents=True, exist_ok=True)
    profile_sha = _profile_sha(run_root)
    device = device_for_training()
    seed_everything(config.seed)

    train_manifest = _load_prepared_manifest(prepared_root, "train")
    val_manifest = _load_prepared_manifest(prepared_root, "val")
    train_indices = np.flatnonzero(
        (train_manifest["class_id"].to_numpy() == 2)
        & (train_manifest["split"].to_numpy() == "train")
    )
    val_indices = np.flatnonzero(
        (val_manifest["class_id"].to_numpy() == 2)
        & (val_manifest["split"].to_numpy() == "val")
    )
    if not len(train_indices) or not len(val_indices):
        raise ValueError("Stage A requires non-empty Batch 3 train and validation crops")

    encoder, load_report = load_micronet_encoder(micronet_path)
    encoder.to(device)
    projector = Projector().to(device)
    predictor = Predictor().to(device)
    prep = InputPrep(config.input_prep).to(device)
    parameter_report = set_trainable(encoder, set(config.trainable_parts))
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in encoder.parameters() if p.requires_grad],
                "lr": config.encoder_lr,
            },
            {
                "params": list(projector.parameters()) + list(predictor.parameters()),
                "lr": config.head_lr,
            },
        ],
        weight_decay=config.weight_decay,
    )
    scaler = make_grad_scaler(config.amp)

    train_base = CropDataset(
        Path(prepared_root) / "train" / "crops.npy",
        Path(prepared_root) / "train" / "crop_manifest.csv",
        train_indices,
        load_into_memory=True,
    )
    augmentation = SpatialAug(
        flips=config.flips,
        rrc_scale=config.rrc_scale,
        brightness_contrast=config.brightness_contrast,
    )
    train_dataset = TwoViewDataset(train_base, augmentation)
    train_generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch,
        shuffle=True,
        drop_last=True,
        num_workers=config.workers,
        pin_memory=True,
        generator=train_generator,
    )
    val_base = CropDataset(
        Path(prepared_root) / "val" / "crops.npy",
        Path(prepared_root) / "val" / "crop_manifest.csv",
        val_indices,
        load_into_memory=False,
    )
    val_dataset = val_base

    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"
    best_loss = float("inf")
    best_epoch = None
    start_epoch = 1
    if resume:
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if checkpoint["config_sha256"] != config_hash:
            raise ValueError("Stage A resume config hash mismatch")
        if checkpoint["split_sha256"] != split_sha or checkpoint["profile_sha256"] != profile_sha:
            raise ValueError("Stage A resume split/profile SHA mismatch")
        if checkpoint["micronet_weight_sha256"] != micronet_sha:
            raise ValueError("Stage A resume MicroNet weight SHA mismatch")
        encoder.load_state_dict(checkpoint["encoder"], strict=True)
        projector.load_state_dict(checkpoint["projector"], strict=True)
        predictor.load_state_dict(checkpoint["predictor"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng_state(checkpoint["rng_state"])
        if checkpoint.get("loader_rng_state") is not None:
            train_generator.set_state(checkpoint["loader_rng_state"])
        best_loss = checkpoint["best_val_loss"]
        best_epoch = checkpoint["best_epoch"]
        start_epoch = checkpoint["epoch"] + 1

    metrics_path = run_dir / "metrics.jsonl"
    for epoch in range(start_epoch, config.epochs + 1):
        encoder.train()
        freeze_bn_stats(encoder)
        projector.train()
        predictor.train()
        prep.train()
        train_losses = []
        for first, second, _, _ in train_loader:
            first = first.cuda(non_blocking=True)
            second = second.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(config.amp):
                loss, _, _ = _simsiam_batch(
                    encoder, projector, predictor, prep, first, second, config.amp
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().float().cpu()))
        validation = _validation(
            encoder, projector, predictor, prep, val_dataset, augmentation, config.batch, config
        )
        loss_value = validation["val_loss"]
        collapsed = (
            not math.isfinite(loss_value)
            or not math.isfinite(validation["z_std"])
            or not math.isfinite(validation["offdiag_cos"])
            or validation["z_std"] < 0.5 / math.sqrt(2048)
            or validation["offdiag_cos"] > 0.9
        )
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
            **validation,
            "collapsed": bool(collapsed),
        }
        append_jsonl(epoch_metrics, metrics_path)
        improved = not collapsed and loss_value < best_loss
        if improved:
            best_loss = loss_value
            best_epoch = epoch
        checkpoint = _checkpoint(
            epoch,
            encoder,
            projector,
            predictor,
            optimizer,
            scaler,
            config_dict,
            config_hash,
            split_sha,
            profile_sha,
            micronet_sha,
            best_loss,
            best_epoch,
            train_generator.get_state(),
        )
        atomic_torch_save(checkpoint, last_path)
        if improved:
            atomic_torch_save(checkpoint, best_path)
        print(json.dumps(epoch_metrics, sort_keys=True), flush=True)

    if best_epoch is None or not best_path.is_file():
        write_json(
            {"run_id": run_id, "status": "failed", "reason": "all epochs collapsed"},
            run_dir / "FAILED.json",
        )
        raise RuntimeError("Every Stage A epoch collapsed; no best checkpoint is available")

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    encoder.load_state_dict(best["encoder"], strict=True)
    export = {
        "encoder": encoder.state_dict(),
        "metadata": {
            "stage": "stage_a",
            "best_epoch": best_epoch,
            "config": config_dict,
            "split_sha256": split_sha,
            "profile_sha256": profile_sha,
            "micronet_weight_sha256": micronet_sha,
            "load_report": load_report,
        },
    }
    atomic_torch_save(export, run_dir / "encoder_export.pt")
    write_json(
        {
            "run_id": run_id,
            "best_epoch": best_epoch,
            "best_val_loss": best_loss,
            "config": config_dict,
            "encoder_trainable_parameters": parameter_report,
            "micronet_load_report": load_report,
            **runtime_info(),
        },
        run_dir / "summary.json",
    )
    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "encoder_export": str(run_dir / "encoder_export.pt"),
        "load_report": load_report,
        **runtime_info(),
    }
