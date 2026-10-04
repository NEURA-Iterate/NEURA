import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from micronet_cls.data import SpatialAug
from micronet_cls.model import (
    Classifier,
    Encoder,
    InputPrep,
    Predictor,
    Projector,
    freeze_bn_stats,
    set_trainable,
)
from micronet_cls.stage_b import StageBConfig, verify_resume_checkpoint
from micronet_cls.training_utils import (
    atomic_torch_save,
    autocast_context,
    config_sha256,
    make_grad_scaler,
    seed_everything,
)
from micronet_cls.weights import load_micronet_encoder


def run_smoke_test(run_id, prepared_root, run_dir, micronet_path, micronet_sha, tag=""):
    if not torch.cuda.is_available():
        raise RuntimeError("Smoke test requires a CUDA GPU")
    device = torch.device("cuda")
    seed_everything(42)
    torch.cuda.reset_peak_memory_stats(device)
    gpu_name = torch.cuda.get_device_name(device)
    cuda_capability = torch.cuda.get_device_capability(device)
    cuda_version = torch.version.cuda
    if not cuda_version or not cuda_version.startswith("12.8"):
        raise RuntimeError(f"Expected CUDA 12.8 PyTorch wheels, got {cuda_version!r}")
    if "B200" in gpu_name.upper() and cuda_capability != (10, 0):
        raise RuntimeError(f"Unexpected B200 compute capability: {cuda_capability}")
    smoke_name = f"smoke_{tag}" if tag else "smoke"
    run_dir = Path(run_dir) / smoke_name
    run_dir.mkdir(parents=True, exist_ok=True)
    prepared_root = Path(prepared_root)

    manifest = pd.read_csv(prepared_root / "train" / "crop_manifest.csv")
    candidate_indices = np.flatnonzero(
        (manifest["class_id"].to_numpy() == 2)
        & (manifest["split"].to_numpy() == "train")
    )
    if len(candidate_indices) < 16:
        raise ValueError("GPU smoke test requires 16 Batch 3 train crops")
    crops = np.load(prepared_root / "train" / "crops.npy", mmap_mode="r")
    batch = torch.from_numpy(
        np.array(crops[candidate_indices[:16]], dtype=np.float32, copy=True)
    ).unsqueeze(1)

    encoder, weight_report = load_micronet_encoder(micronet_path)
    if weight_report["missing_keys"] or weight_report["unexpected_keys"]:
        raise AssertionError(f"MicroNet load had non-fc mismatches: {weight_report}")
    encoder = encoder.to(device)
    projector = Projector().to(device)
    predictor = Predictor().to(device)
    prep = InputPrep("none").to(device)
    set_trainable(encoder, {"input_conv", "layer4"})
    freeze_bn_stats(encoder)
    encoder.train()
    freeze_bn_stats(encoder)
    projector.train()
    predictor.train()
    stage_a_optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in encoder.parameters() if p.requires_grad], "lr": 1e-5},
            {
                "params": list(projector.parameters()) + list(predictor.parameters()),
                "lr": 1e-3,
            },
        ],
        weight_decay=1e-4,
    )
    scaler = make_grad_scaler(True)
    if not scaler.is_enabled():
        raise RuntimeError("CUDA GradScaler did not enable")
    augmentation = SpatialAug(
        flips=True,
        rrc_scale=(0.3, 1.0),
        brightness_contrast=(0.1, 0.1),
    )
    stage_a_losses = []
    projector_grad_seen = False
    target_detached = False
    for step in range(3):
        first = torch.stack(
            [augmentation(image.clone()) for image in batch[step * 4 : step * 4 + 4]]
        ).to(device)
        second = torch.stack(
            [augmentation(image.clone()) for image in batch[step * 4 : step * 4 + 4]]
        ).to(device)
        stage_a_optimizer.zero_grad(set_to_none=True)
        with autocast_context(True):
            autocast_enabled = torch.is_autocast_enabled("cuda")
            autocast_fp16 = torch.get_autocast_dtype("cuda") == torch.float16
            z1 = projector(encoder(prep(first)))
            z2 = projector(encoder(prep(second)))
            p1 = predictor(z1)
            p2 = predictor(z2)
            target1 = z1.detach()
            target2 = z2.detach()
            target_detached = not target1.requires_grad and not target2.requires_grad
            loss = 0.5 * (
                -(F.normalize(p1, dim=1) * F.normalize(target2, dim=1)).sum(1).mean()
                -(F.normalize(p2, dim=1) * F.normalize(target1, dim=1)).sum(1).mean()
            )
        if not autocast_enabled or not autocast_fp16:
            raise AssertionError("Stage A smoke did not activate CUDA fp16 autocast")
        if not torch.isfinite(loss):
            raise AssertionError("Stage A smoke loss is not finite")
        scaler.scale(loss).backward()
        projector_grad_seen = projector_grad_seen or any(
            parameter.grad is not None for parameter in projector.parameters()
        )
        if not target_detached or not projector_grad_seen:
            raise AssertionError("SimSiam stop-gradient/projector-gradient assertion failed")
        scaler.step(stage_a_optimizer)
        scaler.update()
        stage_a_losses.append(float(loss.detach().float().cpu()))

    encoder_cpu = {key: value.detach().cpu() for key, value in encoder.state_dict().items()}
    stage_a_best = {"encoder": encoder_cpu, "stage": "stage_a", "epoch": 1}
    encoder_export = {"encoder": encoder_cpu, "metadata": {"stage": "stage_a"}}
    atomic_torch_save(stage_a_best, run_dir / "stage_a" / "best.pt")
    atomic_torch_save(encoder_export, run_dir / "stage_a" / "encoder_export.pt")
    exported = torch.load(run_dir / "stage_a" / "encoder_export.pt", map_location="cpu", weights_only=False)
    stage_b_encoder = Encoder().cpu()
    stage_b_encoder.load_state_dict(exported["encoder"], strict=True)
    equal_tensors = [
        torch.equal(stage_a_best["encoder"][key], stage_b_encoder.state_dict()[key])
        for key in stage_a_best["encoder"]
    ]
    if not all(equal_tensors):
        raise AssertionError("Stage B did not load every Stage A export tensor identically")

    model = Classifier(stage_b_encoder).to(device)
    input_prep = InputPrep("none").to(device)
    labels = torch.as_tensor(
        manifest.iloc[candidate_indices[:16]]["class_id"].to_numpy(), device=device
    )
    warmup_parameter_report = set_trainable(model.encoder, set())
    model.train()
    freeze_bn_stats(model.encoder)
    warmup_optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
    warmup_losses = []
    for step in range(3):
        images = batch[step * 4 : step * 4 + 4].to(device)
        target = labels[step * 4 : step * 4 + 4]
        warmup_optimizer.zero_grad(set_to_none=True)
        with autocast_context(True):
            loss = F.cross_entropy(model(input_prep(images)).float(), target)
        if not torch.isfinite(loss):
            raise AssertionError("Stage B warmup smoke loss is not finite")
        scaler.scale(loss).backward()
        if any(parameter.grad is not None for parameter in model.encoder.parameters()):
            raise AssertionError("Frozen encoder received gradients during warmup")
        scaler.step(warmup_optimizer)
        scaler.update()
        warmup_losses.append(float(loss.detach().float().cpu()))

    set_trainable(model.encoder, {"input_conv", "layer4"})
    model.train()
    freeze_bn_stats(model.encoder)
    finetune_optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in model.encoder.parameters() if p.requires_grad],
                "lr": 1e-5,
            },
            {"params": model.fc.parameters(), "lr": 1e-4},
        ],
        weight_decay=1e-4,
    )
    finetune_losses = []
    layer4_gradient = False
    layer1_frozen = False
    for step in range(3):
        images = batch[step * 4 : step * 4 + 4].to(device)
        target = labels[step * 4 : step * 4 + 4]
        finetune_optimizer.zero_grad(set_to_none=True)
        with autocast_context(True):
            loss = F.cross_entropy(model(input_prep(images)).float(), target)
        if not torch.isfinite(loss):
            raise AssertionError("Stage B fine-tune smoke loss is not finite")
        scaler.scale(loss).backward()
        layer4_gradient = any(p.grad is not None for p in model.encoder.layer4.parameters())
        layer1_frozen = all(p.grad is None for p in model.encoder.layer1.parameters())
        if not layer4_gradient or not layer1_frozen:
            raise AssertionError("Fine-tune gradient boundary assertion failed")
        scaler.step(finetune_optimizer)
        scaler.update()
        finetune_losses.append(float(loss.detach().float().cpu()))

    model.eval()
    input_prep.eval()
    smoke_checkpoint = {
        "stage": "stage_b",
        "model_state": model.state_dict(),
        "input_prep_mode": "none",
        "epoch": 1,
    }
    checkpoint_path = run_dir / "checkpoint.pt"
    atomic_torch_save(smoke_checkpoint, checkpoint_path)
    sample = input_prep(batch[:2].to(device))
    with torch.no_grad():
        expected = model(sample)
    reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reloaded_model = Classifier(Encoder()).to(device)
    reloaded_model.load_state_dict(reloaded["model_state"], strict=True)
    reloaded_model.eval()
    with torch.no_grad():
        actual = reloaded_model(sample)
    if not torch.equal(expected, actual):
        raise AssertionError("Checkpoint reload changed model output")

    resume_config = asdict(StageBConfig())
    split_sha = "smoke-split-sha"
    profile_sha = "smoke-profile-sha"
    resume_checkpoint = {
        "epoch": 1,
        "config_sha256": config_sha256(resume_config),
        "split_sha256": split_sha,
        "profile_sha256": profile_sha,
    }
    resume_epoch = verify_resume_checkpoint(
        resume_checkpoint, resume_config, split_sha, profile_sha
    )
    if resume_epoch != 2:
        raise AssertionError(f"Expected resume to continue at epoch 2, got {resume_epoch}")
    modified_config = dict(resume_config)
    modified_config["warmup_epochs"] += 1
    config_rejected = False
    try:
        verify_resume_checkpoint(
            resume_checkpoint, modified_config, split_sha, profile_sha
        )
    except ValueError:
        config_rejected = True
    if not config_rejected:
        raise AssertionError("Resume accepted a modified config")

    report = {
        "run_id": run_id,
        "tag": tag,
        "device": str(device),
        "cuda_name": gpu_name,
        "cuda_version": cuda_version,
        "cuda_capability": list(cuda_capability),
        "torch_version": torch.__version__,
        "amp": True,
        "amp_fp16_observed": autocast_fp16,
        "grad_scaler_enabled": scaler.is_enabled(),
        "grad_scaler_scale": scaler.get_scale(),
        "micronet_weight_sha256": micronet_sha,
        "micronet_load_report": weight_report,
        "stage_a": {
            "steps": len(stage_a_losses),
            "losses": stage_a_losses,
            "losses_finite": bool(np.isfinite(stage_a_losses).all()),
            "target_detach_assertion": target_detached,
            "projector_gradients_seen": projector_grad_seen,
        },
        "stage_a_export": {
            "tensors_equal": all(equal_tensors),
            "tensors_verified": len(equal_tensors),
        },
        "stage_b_warmup": {
            "steps": len(warmup_losses),
            "losses": warmup_losses,
            "encoder_frozen": True,
            "trainable_parameter_report": warmup_parameter_report,
        },
        "stage_b_finetune": {
            "steps": len(finetune_losses),
            "losses": finetune_losses,
            "layer4_gradient": layer4_gradient,
            "layer1_frozen": layer1_frozen,
        },
        "checkpoint_reload_identical": True,
        "resume_epoch": resume_epoch,
        "modified_config_rejected": config_rejected,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "status": "passed",
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report
