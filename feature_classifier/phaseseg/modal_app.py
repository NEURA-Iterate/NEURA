from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2]
PHASESEG_DIR = Path(__file__).resolve().parent
ANODE_QC_DIR = ROOT / "anode_microstructure_qc" / "src" / "anode_qc"

app = modal.App("neura-phaseseg")
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "transformers",
        "numpy",
        "scipy",
        "scikit-image",
        "scikit-learn",
        "tifffile",
        "pillow",
        "pandas",
        "pyyaml",
        "huggingface_hub",
    )
    .add_local_dir(str(PHASESEG_DIR), remote_path="/root/phaseseg", copy=True)
    .add_local_dir(str(ANODE_QC_DIR), remote_path="/root/anode_qc", copy=True)
)
data_volume = modal.Volume.from_name("neura-data", create_if_missing=True)
huggingface_secret = modal.Secret.from_name("huggingface")


@app.function(
    image=image,
    volumes={"/data": data_volume},
    secrets=[huggingface_secret],
    timeout=86400,
)
def download_dataset() -> None:
    from anode_qc.data import download_dataset as qc_download_dataset

    qc_download_dataset(Path("/data/images"), token=os.environ["HF_TOKEN"])
    data_volume.commit()


@app.function(image=image, gpu="T4", volumes={"/data": data_volume}, timeout=86400)
def extract_features(batch_size: int = 8) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "phaseseg.features",
            "--images",
            "/data/images",
            "--out",
            "/data/features",
            "--device",
            "cuda",
            "--batch-size",
            str(batch_size),
        ],
        check=True,
    )
    data_volume.commit()


@app.function(image=image, gpu="T4", volumes={"/data": data_volume}, timeout=86400)
def train(epochs: int = 10, folds: int = 5, head: str = "fusion") -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "phaseseg.train",
            "--images",
            "/data/images",
            "--features",
            "/data/features",
            "--labels",
            "pseudo",
            "--head",
            head,
            "--folds",
            str(folds),
            "--out",
            "/data/runs",
            "--epochs",
            str(epochs),
            "--device",
            "cuda",
        ],
        check=True,
    )
    data_volume.commit()


@app.function(image=image, gpu="T4", timeout=1800)
def smoke() -> dict[str, object]:
    import torch

    from phaseseg.backbone import Dinov2Backbone

    device = torch.device("cuda")
    backbone = Dinov2Backbone(device=device)
    batch = torch.rand((8, 3, 518, 518), device=device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for _ in range(2):
            features = backbone(batch)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(20):
            features = backbone(batch)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = {
        "feature_shape": list(features.shape),
        "device": str(features.device),
        "warmup_forwards": 2,
        "timed_forwards": 20,
        "batch_size": 8,
        "seconds_per_forward": elapsed / 20,
        "seconds_per_tile": elapsed / (20 * 8),
        "tiles_per_second": (20 * 8) / elapsed,
    }
    print(result)
    return result
