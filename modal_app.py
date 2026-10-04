import json
import hashlib
import os
import random
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import modal

from micronet_cls.crop_manifest import (
    build_crop_manifest,
    cache_crops_npy,
    expected_crop_counts,
    expected_external_crop_counts,
    pixel_area_summary,
    verify_crop_counts,
)
from micronet_cls.inventory import (
    build_external_source_manifest,
    build_source_manifest,
    inventory_summary,
)
from micronet_cls.prepare_wrapper import prepare_heldout, prepare_train
from micronet_cls.prepare_wrapper import run_prepare
from micronet_cls.splits import (
    assign_group_splits,
    check_split_integrity,
    load_split_manifest,
    save_split_manifest,
    split_manifest_sha256,
)
from micronet_cls.staging import stage_split


app = modal.App("neura-micronet")
data_volume = modal.Volume.from_name("neura-data", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("neura-hf-cache", create_if_missing=True)
runs_volume = modal.Volume.from_name("neura-runs", create_if_missing=True)
# NEURA_GPU sets the default at import time; local --gpu options override it and are
# forwarded by the pipeline to every GPU child function.
DEFAULT_GPU = os.environ.get("NEURA_GPU", "L4")


def _local_git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==2.2.6",
        "scipy==1.15.3",
        "pillow==11.3.0",
        "pandas==2.3.2",
        "tifffile==2025.5.10",
        "torch==2.8.0",
        "torchvision==0.23.0",
        "huggingface_hub==0.30.2",
        "scikit-learn==1.6.1",
        "matplotlib==3.10.1",
    )
    .env({"HF_HOME": "/hf", "NEURA_GIT_HEAD": _local_git_head()})
    .add_local_python_source("micronet_cls")
    .add_local_file("prepare_dataset.py", "/root/prepare_dataset.py")
    .add_local_file("normalize.py", "/root/normalize.py")
)

MOUNTS = {"/data": data_volume, "/hf": hf_cache_volume, "/runs": runs_volume}
GPU_FUNCTION_OPTIONS = {
    "image": image,
    "volumes": MOUNTS,
    "gpu": DEFAULT_GPU,
    "memory": 32768,
    "timeout": 14400,
}


def _remote_files(volume, prefix: str):
    try:
        return {
            entry.path.lstrip("/"): entry.size
            for entry in volume.listdir(prefix, recursive=True)
            if entry.type == modal.volume.FileEntryType.FILE
        }
    except (FileNotFoundError, modal.exception.NotFoundError):
        return {}


def _sha256(path: Path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch_runs_subdir(run_id: str, subdir: str, destination: Path):
    destination = Path(destination)
    entries = runs_volume.listdir(f"/{run_id}/{subdir}", recursive=True)
    downloaded = []
    prefix = Path(run_id) / Path(subdir)
    for entry in entries:
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        remote_relative = Path(entry.path.lstrip("/"))
        relative = remote_relative.relative_to(prefix)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as stream:
            for block in runs_volume.read_file(
                f"/{run_id}/{subdir}/{relative.as_posix()}"
            ):
                stream.write(block)
        downloaded.append(str(target))
    return downloaded


def _model_checkpoint(run_id, model):
    for init in ("stage_a", "micronet"):
        base = f"stage_b_{init}"
        if model == base:
            return Path("/runs") / run_id / base / "best.pt"
        if model.startswith(base + "_"):
            tag = model[len(base) + 1 :]
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", tag):
                return Path("/runs") / run_id / model / "best.pt"
    raise ValueError(f"Unknown model name: {model}")


def _parse_detectors(value: str | None):
    if not value:
        return None
    return tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))


def _split_sha():
    return split_manifest_sha256(Path("/data/splits/split_manifest_seed42.csv"))


def _profile_sha(run_id):
    import hashlib

    digest = hashlib.sha256()
    path = Path("/runs") / run_id / "profile.json"
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_staging_map(staged_root: Path, split_name: str):
    path = staged_root / split_name / "staging_map.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _split_crop_summary(crop_manifest):
    result = {}
    if crop_manifest.empty:
        return result
    for (batch_name, detector), group in crop_manifest.groupby(
        ["batch_name", "detector"], sort=True
    ):
        result.setdefault(batch_name, {})[detector] = {
            "groups": int(group["group_id"].nunique()),
            "images": int(group["source_image_id"].nunique()),
            "crops": int(len(group)),
        }
    return result


def _finalize_completion_marker(prepared_split_dir: Path, crop_count: int, profile_sha: str):
    from datetime import datetime, timezone

    marker_path = prepared_split_dir / "_COMPLETE.json"
    marker = {
        "csv_count": sum(1 for _ in prepared_split_dir.glob("*.csv")),
        "crop_count": crop_count,
        "profile_sha256": profile_sha,
        "tile": 512,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = marker_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(marker, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(marker_path)


def _spawn_wait(function, *args, **kwargs):
    call = function.spawn(*args, **kwargs)
    print(f"function_call_id={call.object_id}", flush=True)
    return call.get()


@app.local_entrypoint()
def upload(data_root: str = "."):
    root = Path(data_root).resolve()
    artifacts = Path("artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)
    manifest = build_source_manifest(root)
    manifest_path = artifacts / "source_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    existing = _remote_files(data_volume, "/raw")
    uploads = []
    with data_volume.batch_upload() as batch:
        for relative_path in sorted(manifest["relative_path"].unique()):
            source = root / Path(relative_path)
            remote_path = f"/raw/{relative_path}"
            if existing.get(remote_path.lstrip("/")) == source.stat().st_size:
                continue
            batch.put_file(str(source), remote_path)
            uploads.append(relative_path)
        remote_manifest = "/raw/source_manifest.csv"
        if existing.get(remote_manifest.lstrip("/")) != manifest_path.stat().st_size:
            batch.put_file(str(manifest_path), remote_manifest)
            uploads.append("source_manifest.csv")
    print(
        json.dumps(
            {
                "source_manifest": str(manifest_path),
                "source_images": int(manifest["relative_path"].nunique()),
                "uploaded_files": len(uploads),
                "skipped_matching_size": int(manifest["relative_path"].nunique())
                - len([path for path in uploads if path != "source_manifest.csv"]),
                "inventory": inventory_summary(manifest),
            },
            indent=2,
        )
    )


@app.local_entrypoint()
def predict_folder(
    run_id: str,
    folder: str,
    detectors: str = "Inlens",
    name: str = "external",
    model: str = "stage_b_micronet_inlens",
    gpu: str = "B200",
):
    root = Path(folder).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise ValueError(f"Invalid external dataset name: {name}")
    detector_filter = _parse_detectors(detectors)
    if detector_filter is None:
        raise ValueError("Select at least one detector with --detectors")

    manifest = build_external_source_manifest(root)
    if manifest.empty:
        raise ValueError(f"No detector TIFFs found in {root}")
    output_dir = Path("artifacts") / run_id / "external" / name
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "source_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    inventory = []
    for _, row in manifest.sort_values("relative_path").iterrows():
        inventory.append(
            {
                "relative_path": row["relative_path"],
                "cross_section_id": row["cross_section_id"],
                "detector": row["detector"],
                "height": int(row["height"]),
                "width": int(row["width"]),
                "dtype": row["dtype"],
                "n_channels": int(row["n_channels"]),
                "channels_identical": bool(row["channels_identical"]),
                "n_mismatched_pixels": int(row["n_mismatched_pixels"]),
                "sha256": row["sha256"],
                "complete_set": bool(row["complete_set"]),
                "missing_detectors": row["missing_detectors"],
                "crops": (int(row["height"]) // 512) * (int(row["width"]) // 512),
            }
        )
    incomplete_groups = [
        {
            "group_id": group_id,
            "missing_detectors": rows["missing_detectors"].iloc[0],
        }
        for group_id, rows in manifest.groupby("group_id", sort=True)
        if not bool(rows["complete_set"].iloc[0])
    ]
    inventory_summary_local = {
        "name": name,
        "folder": str(root),
        "ground_truth": "not available",
        "source_manifest": str(manifest_path),
        "source_images": int(manifest["source_image_id"].nunique()),
        "groups": int(manifest["group_id"].nunique()),
        "complete_groups": int(
            manifest.loc[manifest["complete_set"], "group_id"].nunique()
        ),
        "incomplete_groups": incomplete_groups,
        "images": inventory,
        "zero_crop_images": [
            row["source_image_id"]
            for _, row in manifest.iterrows()
            if int(row["height"]) < 512 or int(row["width"]) < 512
        ],
    }

    remote_raw = f"/external/{name}/raw"
    existing = _remote_files(data_volume, f"/external/{name}")
    uploaded_files = 0
    skipped_files = 0
    with data_volume.batch_upload() as batch:
        for relative_path in sorted(manifest["relative_path"].unique()):
            source = root / Path(relative_path)
            remote_path = f"{remote_raw}/{relative_path}"
            if existing.get(remote_path.lstrip("/")) == source.stat().st_size:
                skipped_files += 1
            else:
                batch.put_file(str(source), remote_path)
                uploaded_files += 1
        remote_manifest = f"/external/{name}/source_manifest.csv"
        if existing.get(remote_manifest.lstrip("/")) == manifest_path.stat().st_size:
            skipped_files += 1
        else:
            batch.put_file(str(manifest_path), remote_manifest)
            uploaded_files += 1
    inventory_summary_local["uploaded_files"] = uploaded_files
    inventory_summary_local["skipped_matching_size"] = skipped_files
    print(json.dumps(inventory_summary_local, indent=2), flush=True)

    prepare_call = prepare_external.spawn(run_id, name)
    print(f"prepare_function_call_id={prepare_call.object_id}", flush=True)
    preparation = prepare_call.get()
    prediction_call = predict_external.with_options(gpu=gpu).spawn(
        run_id,
        name,
        detectors=detector_filter,
        model=model,
    )
    print(f"predict_function_call_id={prediction_call.object_id}", flush=True)
    predictions = prediction_call.get()
    fetched_files = _fetch_runs_subdir(
        run_id,
        f"external/{name}",
        output_dir,
    )
    print(
        json.dumps(
            {
                "prepare_summary": preparation,
                "prediction_summary": predictions,
                "fetched_files": fetched_files,
                "artifact_dir": str(output_dir),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


@app.local_entrypoint()
def split():
    source_path = Path("artifacts/source_manifest.csv")
    if not source_path.is_file():
        raise FileNotFoundError(f"Run upload first; source manifest missing: {source_path}")
    source_manifest = load_split_manifest(source_path)
    split_manifest = assign_group_splits(source_manifest, seed=42)
    summary = check_split_integrity(split_manifest)
    output_path = Path("artifacts/split_manifest.csv")
    save_split_manifest(split_manifest, output_path)
    with data_volume.batch_upload() as batch:
        batch.put_file(str(output_path), "/splits/split_manifest_seed42.csv")
    print(json.dumps(summary, indent=2))
    print(f"split_manifest_sha256={split_manifest_sha256(output_path)}")


@app.function(
    image=image,
    volumes=MOUNTS,
    cpu=4,
    memory=16384,
    timeout=7200,
)
def prepare(run_id: str, split_manifest_rel: str = "splits/split_manifest_seed42.csv"):
    data_root = Path("/data")
    staged_root = data_root / "staged" / run_id
    prepared_root = data_root / "prepared" / run_id
    run_dir = Path("/runs") / run_id
    split_path = data_root / split_manifest_rel.lstrip("/")
    split_manifest = load_split_manifest(split_path)
    excluded_groups = inventory_summary(split_manifest)["excluded_groups"]

    for split_name in ("train", "val", "test"):
        stage_split(split_manifest, data_root / "raw", staged_root, split_name)
        data_volume.commit()

    train_result = prepare_train(run_dir, staged_root, prepared_root)
    data_volume.commit()
    runs_volume.commit()
    profile_sha = train_result["profile_sha256"]
    split_summaries = {}
    all_areas = {}

    for split_name in ("train", "val", "test"):
        if split_name != "train":
            prepare_heldout(run_dir, staged_root, prepared_root, split_name)
            data_volume.commit()
        prepared_split_dir = prepared_root / split_name
        selected_manifest = split_manifest[split_manifest["split"] == split_name]
        crop_manifest = build_crop_manifest(
            prepared_split_dir,
            selected_manifest,
            _load_staging_map(staged_root, split_name),
            profile_sha,
            split_name,
        )
        expected = expected_crop_counts(selected_manifest)
        verified = verify_crop_counts(crop_manifest, expected)
        cache_crops_npy(crop_manifest, prepared_split_dir / "crops.npy")
        _finalize_completion_marker(prepared_split_dir, verified["actual"], profile_sha)
        data_volume.commit()
        split_summaries[split_name] = {
            "counts": _split_crop_summary(crop_manifest),
            "crop_count": verified["actual"],
            "crop_manifest": str(prepared_split_dir / "crop_manifest.csv"),
            "crops_npy": str(prepared_split_dir / "crops.npy"),
        }
        all_areas[split_name] = pixel_area_summary(split_manifest, split_name)

    summary = {
        "run_id": run_id,
        "split_manifest_sha256": split_manifest_sha256(split_path),
        "profile": train_result["profile"],
        "profile_sha256": profile_sha,
        "calibration_image_name": train_result["calibration_image_name"],
        "retained_discarded_pixel_area": all_areas,
        "excluded_groups": excluded_groups,
        "splits": split_summaries,
    }
    (run_dir / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    runs_volume.commit()
    return summary


@app.local_entrypoint()
def prepare_cli(run_id: str = ""):
    if not run_id:
        run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    summary = _spawn_wait(prepare, run_id, "splits/split_manifest_seed42.csv")
    output_dir = Path("artifacts") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "prepare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"summary_path={summary_path}")


@app.function(
    image=image,
    volumes=MOUNTS,
    cpu=4,
    memory=16384,
    timeout=7200,
)
def prepare_external(run_id: str, name: str):
    import pandas as pd

    from micronet_cls.training_utils import write_json

    external_root = Path("/data/external") / name
    source_manifest = pd.read_csv(external_root / "source_manifest.csv")
    run_dir = Path("/runs") / run_id
    provenance_path = run_dir / "profile_provenance.json"
    profile_path = run_dir / "profile.json"
    if not profile_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError("The reviewed training profile and provenance are required")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    profile_sha = _sha256(profile_path)
    if profile_sha != provenance.get("profile_sha256"):
        raise ValueError("Training profile SHA256 does not match profile provenance")

    stage_base = external_root / "staged"
    staged_folder = stage_base / "External"
    staged_folder.mkdir(parents=True, exist_ok=True)
    selected = source_manifest[
        source_manifest["complete_set"].astype(bool)
        & source_manifest["detector"].isin(("BSE", "ETD", "Inlens"))
    ]
    mapping = {}
    destinations = {}
    for _, row in selected.iterrows():
        detector = row["detector"]
        image_name = row["cross_section_id"]
        filename = f"{image_name}_{detector}.tif"
        source = external_root / "raw" / Path(row["relative_path"])
        target = staged_folder / filename
        target_key = f"External/{filename}"
        previous = destinations.get(target_key)
        if previous is not None and previous != row["relative_path"]:
            raise ValueError(f"External staging collision for {target_key}")
        destinations[target_key] = row["relative_path"]
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copyfile(source, target)
        source_sha = row["sha256"]
        if _sha256(target) != source_sha:
            raise ValueError(f"SHA256 mismatch after staging {source} -> {target}")
        mapping[target_key] = {
            "original_relative_path": row["relative_path"],
            "sha256": source_sha,
        }

    map_path = stage_base / "staging_map.json"
    map_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_map = map_path.with_suffix(".json.tmp")
    temporary_map.write_text(
        json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary_map.replace(map_path)
    data_volume.commit()

    prepared_dir = external_root / "prepared"
    external_run_dir = run_dir / "external" / name
    external_run_dir.mkdir(parents=True, exist_ok=True)
    run_prepare(
        prepared_dir,
        [str(staged_folder)],
        profile_path,
        tile=512,
        decimals=3,
        log_path=external_run_dir / "prepare.log",
    )
    if _sha256(profile_path) != profile_sha:
        raise ValueError("External preparation changed the training profile")
    crop_manifest = build_crop_manifest(
        prepared_dir,
        source_manifest,
        mapping,
        profile_sha,
        "external",
        external=True,
    )
    expected = expected_external_crop_counts(source_manifest)
    verified = verify_crop_counts(crop_manifest, expected)
    cache_crops_npy(crop_manifest, prepared_dir / "crops.npy")
    _finalize_completion_marker(prepared_dir, verified["actual"], profile_sha)
    data_volume.commit()

    (external_run_dir / "crop_manifest.csv").write_text(
        crop_manifest.to_csv(index=False), encoding="utf-8"
    )
    shutil.copyfile(map_path, external_run_dir / "staging_map.json")
    image_rows = source_manifest.drop_duplicates("source_image_id")
    inventory = []
    for _, row in image_rows.sort_values("source_image_id").iterrows():
        crops = (int(row["height"]) // 512) * (int(row["width"]) // 512)
        inventory.append(
            {
                "source_image_id": row["source_image_id"],
                "cross_section_id": row["cross_section_id"],
                "detector": row["detector"],
                "relative_path": row["relative_path"],
                "sha256": row["sha256"],
                "height": int(row["height"]),
                "width": int(row["width"]),
                "dtype": row["dtype"],
                "n_channels": int(row["n_channels"]),
                "channels_identical": bool(row["channels_identical"]),
                "n_mismatched_pixels": int(row["n_mismatched_pixels"]),
                "crops": crops,
            }
        )
    summary = {
        "run_id": run_id,
        "name": name,
        "status": "complete",
        "ground_truth": "not available",
        "source_images": int(source_manifest["source_image_id"].nunique()),
        "complete_groups": int(
            source_manifest.loc[source_manifest["complete_set"], "group_id"].nunique()
        ),
        "incomplete_groups": [
            {
                "group_id": group_id,
                "missing_detectors": rows["missing_detectors"].iloc[0],
            }
            for group_id, rows in source_manifest.groupby("group_id", sort=True)
            if not bool(rows["complete_set"].iloc[0])
        ],
        "images": inventory,
        "crop_count": verified["actual"],
        "expected_crop_count": verified["expected"],
        "zero_crop_groups": expected["zero_crop_groups"],
        "zero_crop_images": [
            row["source_image_id"] for row in inventory if row["crops"] == 0
        ],
        "profile_sha256": profile_sha,
        "profile_provenance": str(provenance_path),
        "prepare_dataset_sha256": _sha256(Path("/root/prepare_dataset.py")),
        "prepared_dir": str(prepared_dir),
        "crop_manifest": str(prepared_dir / "crop_manifest.csv"),
        "crops_npy": str(prepared_dir / "crops.npy"),
    }
    write_json(summary, external_run_dir / "prepare_summary.json")
    runs_volume.commit()
    return summary


@app.function(**GPU_FUNCTION_OPTIONS)
def smoke_test(run_id: str, tag: str = ""):
    from micronet_cls.smoke import run_smoke_test
    from micronet_cls.weights import download_micronet

    weight = download_micronet("/hf")
    result = run_smoke_test(
        run_id,
        Path("/data/prepared") / run_id,
        Path("/runs") / run_id,
        Path(weight["path"]),
        weight["sha256"],
        tag=tag,
    )
    runs_volume.commit()
    return result


@app.function(**GPU_FUNCTION_OPTIONS)
def adapt_batch3(run_id: str, resume: bool = False, tag: str = ""):
    from micronet_cls.stage_a import StageAConfig, run_stage_a
    from micronet_cls.weights import download_micronet

    if tag == "v2":
        config = StageAConfig(
            tag=tag,
            epochs=20,
            batch=32,
            encoder_lr=5e-5,
            head_lr=1e-3,
            weight_decay=1e-4,
            trainable_parts=("all",),
            flips=True,
            rrc_scale=(0.3, 1.0),
            brightness_contrast=(0.1, 0.1),
            seed=42,
            amp=True,
            input_prep="imagenet",
        )
    else:
        config = StageAConfig(tag=tag)
    weight = download_micronet("/hf")
    result = run_stage_a(
        run_id,
        Path("/data/prepared") / run_id,
        Path("/runs") / run_id,
        _split_sha(),
        Path(weight["path"]),
        weight["sha256"],
        config=config,
        resume=resume,
    )
    runs_volume.commit()
    return result


@app.function(**GPU_FUNCTION_OPTIONS)
def train_classifier(
    run_id: str,
    init: str = "stage_a",
    resume: bool = False,
    microbatch: int = 4,
    accum: int = 4,
    full_encoder: bool = False,
    input_prep: str = "none",
    tag: str = "",
    warmup_epochs: int = 5,
    finetune_epochs: int = 25,
    encoder_lr: float = 1e-5,
    patience: int = 7,
    detectors: tuple[str, ...] | None = None,
):
    from micronet_cls.stage_b import StageBConfig, run_stage_b

    if tag == "inlens":
        if init != "micronet" or detectors != ("Inlens",):
            raise ValueError("The inlens tag requires init='micronet' and detectors=('Inlens',)")
        microbatch = 16
        accum = 1
        full_encoder = True
        input_prep = "imagenet"
        warmup_epochs = 5
        finetune_epochs = 40
        encoder_lr = 3e-5
        patience = 10
    result = run_stage_b(
        run_id,
        Path("/data/prepared") / run_id,
        Path("/runs") / run_id,
        _split_sha(),
        config=StageBConfig(
            init=init,
            tag=tag,
            microbatch=microbatch,
            accum=accum,
            full_encoder=full_encoder,
            input_prep=input_prep,
            warmup_epochs=warmup_epochs,
            finetune_epochs=finetune_epochs,
            encoder_lr=encoder_lr,
            patience=patience,
            detectors=detectors,
        ),
        resume=resume,
    )
    runs_volume.commit()
    return result


@app.function(**GPU_FUNCTION_OPTIONS)
def evaluate(
    run_id: str,
    model: str = "stage_b_stage_a",
    split: str = "test",
    detectors: tuple[str, ...] | None = None,
):
    from micronet_cls.evaluate import evaluate_checkpoint

    run_dir = Path("/runs") / run_id
    prepared_root = Path("/data/prepared") / run_id
    output_split = (
        "test_all_detectors"
        if model.endswith("_inlens") and split == "test" and detectors is None
        else split
    )
    result = evaluate_checkpoint(
        _model_checkpoint(run_id, model),
        prepared_root / split,
        prepared_root / "train" / "crop_manifest.csv",
        run_dir / "evaluation" / model / output_split,
        split=split,
        detectors=detectors,
    )
    runs_volume.commit()
    return result


def _ensure_agnostic_profile(run_id):
    from micronet_cls.raw_preprocess import fit_agnostic_scale
    from micronet_cls.training_utils import write_json

    run_dir = Path("/runs") / run_id
    profile_path = run_dir / "profile_agnostic.json"
    provenance_path = run_dir / "profile_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    calibration = provenance["calibration"]
    calibration_paths = {
        detector: Path("/data/raw") / Path(source["original_relative_path"])
        for detector, source in calibration["sources"].items()
    }
    normalize_path = Path("/root/normalize.py")
    import hashlib

    digest = hashlib.sha256(normalize_path.read_bytes()).hexdigest()
    if profile_path.is_file():
        existing = json.loads(profile_path.read_text(encoding="utf-8"))
        if existing.get("normalize_sha256") == digest and existing.get("variant") == "DA-v1":
            return existing
    scale = fit_agnostic_scale(calibration_paths)
    profile = {
        "scale": scale,
        "calibration_image_name": calibration["image_name"],
        "calibration_sources": {
            detector: {
                "original_relative_path": source["original_relative_path"],
                "sha256": source["sha256"],
            }
            for detector, source in calibration["sources"].items()
        },
        "normalize_sha256": digest,
        "variant": "DA-v1",
    }
    write_json(profile, profile_path)
    runs_volume.commit()
    return profile


@app.function(**GPU_FUNCTION_OPTIONS)
def evaluate_raw(
    run_id: str,
    model: str,
    split: str = "test",
    tag: str = "",
    detectors: tuple[str, ...] | None = None,
):
    from micronet_cls.evaluate import evaluate_raw_model

    profile = _ensure_agnostic_profile(run_id)
    raw_eval_name = (
        f"raw_eval_{tag}"
        if tag
        else (
            f"raw_eval_{'_'.join(detector.lower() for detector in detectors)}"
            if detectors
            else "raw_eval"
        )
    )
    result = evaluate_raw_model(
        _model_checkpoint(run_id, model),
        run_id,
        Path("/data/prepared") / run_id / split,
        Path("/data/raw"),
        profile["scale"],
        Path("/runs") / run_id / raw_eval_name,
        split=split,
        detectors=detectors,
    )
    runs_volume.commit()
    return result


@app.function(
    image=image,
    volumes=MOUNTS,
    cpu=4,
    memory=32768,
    timeout=14400,
)
def raw_parity(run_id: str, split: str = "test", tag: str = ""):
    from micronet_cls.raw_preprocess import parity_report

    profile = _ensure_agnostic_profile(run_id)
    result = parity_report(
        run_id,
        split=split,
        prepared_root=Path("/data/prepared") / run_id,
        raw_root=Path("/data/raw"),
        run_dir=Path("/runs") / run_id,
        tag=tag,
    )
    result["agnostic_scale"] = profile["scale"]
    runs_volume.commit()
    return result


@app.function(**GPU_FUNCTION_OPTIONS)
def predict_remote(
    run_id: str,
    model: str,
    file_bytes: bytes,
    filename: str,
    mode: str = "csv",
):
    from micronet_cls.predict import predict_bytes
    from micronet_cls.training_utils import write_json

    checkpoint = _model_checkpoint(run_id, model)
    scale = _ensure_agnostic_profile(run_id)["scale"] if mode == "raw" else None
    result = predict_bytes(
        checkpoint,
        file_bytes,
        filename,
        mode=mode,
        agnostic_scale=scale,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    result["filename"] = Path(filename).name
    result["model"] = model
    write_json(result, Path("/runs") / run_id / "remote_predictions" / f"{stamp}.json")
    runs_volume.commit()
    return result


@app.function(**GPU_FUNCTION_OPTIONS)
def predict_external(
    run_id: str,
    name: str,
    detectors: tuple[str, ...] = ("Inlens",),
    model: str = "stage_b_micronet_inlens",
):
    import numpy as np
    import pandas as pd

    from micronet_cls.predict import load_classifier_checkpoint, predict_with_model
    from micronet_cls.training_utils import runtime_info, write_json

    detectors = tuple(dict.fromkeys(detectors))
    if not detectors or set(detectors) - {"BSE", "ETD", "Inlens"}:
        raise ValueError(f"Invalid external detector filter: {detectors}")
    external_root = Path("/data/external") / name
    prepared_dir = external_root / "prepared"
    source_manifest = pd.read_csv(external_root / "source_manifest.csv")
    crop_manifest = pd.read_csv(prepared_dir / "crop_manifest.csv")
    source_rows = source_manifest[
        source_manifest["complete_set"].astype(bool)
        & source_manifest["detector"].isin(detectors)
    ].drop_duplicates("source_image_id")
    source_rows = source_rows.sort_values("source_image_id")
    checkpoint_path = _model_checkpoint(run_id, model)
    network, prep, checkpoint, device = load_classifier_checkpoint(checkpoint_path)
    profile_sha = _profile_sha(run_id)
    scale_data = json.loads(
        (Path("/runs") / run_id / "profile_agnostic.json").read_text(encoding="utf-8")
    )
    scale = float(scale_data["scale"])
    prediction_rows = []
    crop_prediction_rows = []

    for _, source in source_rows.iterrows():
        source_image_id = source["source_image_id"]
        rows = crop_manifest[
            (crop_manifest["source_image_id"] == source_image_id)
            & (crop_manifest["detector"].isin(detectors))
        ].sort_values(["y", "x"])
        probabilities = []
        for crop_index, crop_row in rows.iterrows():
            result = predict_with_model(
                network,
                prep,
                crop_row["csv_path"],
                mode="csv",
                device=device,
            )
            prob = np.asarray(
                [
                    result["probabilities"]["Batch 1"],
                    result["probabilities"]["Batch 2"],
                    result["probabilities"]["Batch 3"],
                ],
                dtype=np.float64,
            )
            probabilities.append(prob)
            crop_prediction_rows.append(
                {
                    "row_index": int(crop_index),
                    "source_image_id": source_image_id,
                    "group_id": source["group_id"],
                    "detector": source["detector"],
                    "y": int(crop_row["y"]),
                    "x": int(crop_row["x"]),
                    "pred_class": int(prob.argmax()),
                    "predicted_batch": result["predicted_batch"],
                    "p0": float(prob[0]),
                    "p1": float(prob[1]),
                    "p2": float(prob[2]),
                    "crop_confidence": float(prob.max()),
                }
            )

        raw_path = external_root / "raw" / Path(source["relative_path"])
        try:
            raw_result = predict_with_model(
                network,
                prep,
                raw_path,
                mode="raw",
                device=device,
                agnostic_scale=scale,
            )
        except ValueError as exc:
            if "yields zero crops under the edge policy" not in str(exc):
                raise
            raw_result = {
                "predicted_batch": None,
                "probabilities": None,
                "n_crops": 0,
                "warning": str(exc),
                "fallback_used": False,
            }

        if probabilities:
            crop_probs = np.stack(probabilities)
            mean_probability = crop_probs.mean(axis=0)
            vote_counts = np.bincount(crop_probs.argmax(axis=1), minlength=3)
            confidence = crop_probs.max(axis=1)
            prepared_prediction = f"Batch {int(mean_probability.argmax()) + 1}"
            mean_probabilities = {
                f"Batch {index + 1}": float(value)
                for index, value in enumerate(mean_probability)
            }
            min_confidence = float(confidence.min())
            max_confidence = float(confidence.max())
        else:
            vote_counts = np.zeros(3, dtype=np.int64)
            prepared_prediction = None
            mean_probabilities = None
            min_confidence = None
            max_confidence = None
        prediction_rows.append(
            {
                "source_image_id": source_image_id,
                "cross_section_id": source["cross_section_id"],
                "detector": source["detector"],
                "relative_path": source["relative_path"],
                "height": int(source["height"]),
                "width": int(source["width"]),
                "dtype": source["dtype"],
                "sha256": source["sha256"],
                "predicted_batch": prepared_prediction,
                "probabilities": mean_probabilities,
                "n_crops": len(probabilities),
                "crop_vote_counts": {
                    f"Batch {index + 1}": int(vote_counts[index])
                    for index in range(3)
                },
                "min_crop_confidence": min_confidence,
                "max_crop_confidence": max_confidence,
                "raw_da_v1_predicted_batch": raw_result["predicted_batch"],
                "raw_da_v1_probabilities": raw_result["probabilities"],
                "raw_da_v1_n_crops": raw_result["n_crops"],
                "raw_da_v1_fallback_used": raw_result.get("fallback_used", False),
                "raw_da_v1_warning": raw_result["warning"],
            }
        )

    output_dir = Path("/runs") / run_id / "external" / name
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "predictions_images.csv"
    crop_path = output_dir / "predictions_crops.csv"
    pd.DataFrame(prediction_rows).to_csv(image_path, index=False)
    pd.DataFrame(crop_prediction_rows).to_csv(crop_path, index=False)
    summary = {
        "run_id": run_id,
        "name": name,
        "ground_truth": "not available",
        "raw_mode_note": (
            "DA-v1 raw predictions are secondary and lower-parity; they are not "
            "equivalent to prepared-CSV predictions."
        ),
        "model": model,
        "model_checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "profile_sha256": profile_sha,
        "prepare_dataset_sha256": _sha256(Path("/root/prepare_dataset.py")),
        "detectors": list(detectors),
        "n_source_images": len(prediction_rows),
        "n_crops": len(crop_prediction_rows),
        "predictions_images_csv": str(image_path),
        "predictions_crops_csv": str(crop_path),
        "images": prediction_rows,
        **runtime_info(),
    }
    write_json(summary, output_dir / "predictions.json")
    runs_volume.commit()
    return summary


@app.function(**GPU_FUNCTION_OPTIONS)
def live_demo(
    run_id: str,
    model: str,
    n: int = 12,
    seed: int = 42,
    tag: str = "",
    detectors: tuple[str, ...] | None = None,
):
    import numpy as np
    import pandas as pd

    from micronet_cls.predict import load_classifier_checkpoint, predict_with_model
    from micronet_cls.training_utils import runtime_info, write_json

    prepared_root = Path("/data/prepared") / run_id
    test_manifest = pd.read_csv(prepared_root / "test" / "crop_manifest.csv")
    source_rows = test_manifest.drop_duplicates("source_image_id").reset_index(drop=True)
    if detectors is not None:
        source_rows = source_rows[source_rows["detector"].isin(detectors)].reset_index(drop=True)
    rng = random.Random(seed)
    selected = rng.sample(
        list(range(len(source_rows))), k=min(max(0, n), len(source_rows))
    )
    checkpoint = _model_checkpoint(run_id, model)
    network, prep, _, device = load_classifier_checkpoint(checkpoint)
    scale = _ensure_agnostic_profile(run_id)["scale"]
    predictions = []
    with tempfile.TemporaryDirectory(prefix="micronet-live-demo-") as directory:
        for query_index, row_index in enumerate(selected, start=1):
            source = source_rows.iloc[row_index]
            raw_path = Path("/data/raw") / Path(source["original_relative_path"])
            neutral_path = Path(directory) / f"query_{query_index:02d}.tif"
            shutil.copyfile(raw_path, neutral_path)
            raw_result = predict_with_model(
                network,
                prep,
                neutral_path,
                mode="raw",
                device=device,
                agnostic_scale=scale,
            )
            crop_rows = test_manifest[
                test_manifest["source_image_id"] == source["source_image_id"]
            ]
            csv_probabilities = []
            for _, crop_row in crop_rows.iterrows():
                output = predict_with_model(
                    network,
                    prep,
                    crop_row["csv_path"],
                    mode="csv",
                    device=device,
                )
                csv_probabilities.append(
                    [
                        output["probabilities"]["Batch 1"],
                        output["probabilities"]["Batch 2"],
                        output["probabilities"]["Batch 3"],
                    ]
                )
            mean_csv = np.asarray(csv_probabilities).mean(axis=0)
            predictions.append(
                {
                    "query": neutral_path.name,
                    "source_image_id": source["source_image_id"],
                    "predicted_raw": raw_result["predicted_batch"],
                    "predicted_csv": f"Batch {int(mean_csv.argmax()) + 1}",
                    "raw_probabilities": raw_result["probabilities"],
                    "csv_probabilities": {
                        f"Batch {index + 1}": float(probability)
                        for index, probability in enumerate(mean_csv)
                    },
                    "warning": raw_result["warning"],
                    "fallback_used": raw_result["fallback_used"],
                }
            )

    labels = (
        test_manifest.drop_duplicates("source_image_id")
        .set_index("source_image_id")["class_id"]
        .to_dict()
    )
    for row in predictions:
        truth_id = int(labels[row["source_image_id"]])
        row["truth"] = f"Batch {truth_id + 1}"
        row["raw_correct"] = row["predicted_raw"] == row["truth"]
        row["csv_correct"] = row["predicted_csv"] == row["truth"]
    result = {
        "run_id": run_id,
        "model": model,
        "tag": tag,
        "detectors": list(detectors) if detectors is not None else None,
        "n": len(predictions),
        "seed": seed,
        "queries": predictions,
        "raw_accuracy": float(np.mean([row["raw_correct"] for row in predictions])) if predictions else None,
        "csv_accuracy": float(np.mean([row["csv_correct"] for row in predictions])) if predictions else None,
        **runtime_info(),
    }
    live_demo_name = f"live_demo_{tag}.json" if tag else "live_demo.json"
    if detectors == ("Inlens",):
        live_demo_name = "live_demo_inlens.json"
    write_json(result, Path("/runs") / run_id / live_demo_name)
    runs_volume.commit()
    print(
        pd.DataFrame(predictions)[
            ["query", "predicted_raw", "predicted_csv", "truth"]
        ].to_string(index=False),
        flush=True,
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.function(
    image=image,
    volumes=MOUNTS,
    cpu=4,
    memory=32768,
    timeout=14400,
)
def pipeline(run_id: str, gpu: str = DEFAULT_GPU, tag: str = ""):
    from micronet_cls.training_utils import write_json

    if tag not in {"", "v2"}:
        raise ValueError("pipeline tag must be empty or 'v2'")
    suffix = f"_{tag}" if tag else ""
    is_v2 = tag == "v2"
    gpu_adapt = adapt_batch3.with_options(gpu=gpu)
    gpu_train = train_classifier.with_options(gpu=gpu)
    gpu_evaluate = evaluate.with_options(gpu=gpu)
    gpu_evaluate_raw = evaluate_raw.with_options(gpu=gpu)
    gpu_live_demo = live_demo.with_options(gpu=gpu)
    run_dir = Path("/runs") / run_id
    pipeline_summary_path = run_dir / f"pipeline_summary{suffix}.json"
    if pipeline_summary_path.is_file():
        return json.loads(pipeline_summary_path.read_text(encoding="utf-8"))

    stage_a_name = f"stage_a{suffix}"
    stage_a_dir = run_dir / stage_a_name
    stage_a_summary_path = stage_a_dir / "summary.json"
    if (
        stage_a_summary_path.is_file()
        and (stage_a_dir / "best.pt").is_file()
        and (stage_a_dir / "encoder_export.pt").is_file()
    ):
        stage_a = json.loads(stage_a_summary_path.read_text(encoding="utf-8"))
        print("Stage A already complete; reusing saved outputs", flush=True)
    else:
        print("Stage A: Batch 3 SimSiam adaptation", flush=True)
        stage_a = gpu_adapt.remote(
            run_id,
            resume=(stage_a_dir / "last.pt").is_file(),
            tag=tag,
        )
    data_volume.commit()
    runs_volume.commit()
    stage_a_init = "stage_a_v2" if is_v2 else "stage_a"
    models = (
        (f"stage_b_stage_a{suffix}", stage_a_init),
        (f"stage_b_micronet{suffix}", "micronet"),
    )
    stage_b_results = {}
    for model, init in models:
        stage_dir = run_dir / model
        stage_summary = stage_dir / "summary.json"
        if (
            stage_summary.is_file()
            and (stage_dir / "model_card.json").is_file()
            and (stage_dir / "best.pt").is_file()
        ):
            stage_b_results[model] = json.loads(
                stage_summary.read_text(encoding="utf-8")
            )
            print(f"{model} already complete; reusing saved outputs", flush=True)
        else:
            stage_b_kwargs = {}
            if is_v2:
                stage_b_kwargs = {
                    "microbatch": 16,
                    "accum": 1,
                    "full_encoder": True,
                    "input_prep": "imagenet",
                    "warmup_epochs": 5,
                    "finetune_epochs": 40,
                    "encoder_lr": 3e-5,
                    "patience": 10,
                }
            print(f"Training {model}", flush=True)
            stage_b_results[model] = gpu_train.remote(
                run_id,
                init=init,
                resume=(stage_dir / "last.pt").is_file(),
                tag=tag,
                **stage_b_kwargs,
            )
        data_volume.commit()
        runs_volume.commit()

    val_results = {}
    for model, _ in models:
        metrics_path = run_dir / "evaluation" / model / "val" / "metrics.json"
        if metrics_path.is_file():
            val_results[model] = json.loads(metrics_path.read_text(encoding="utf-8"))
            print(f"Reusing saved validation evaluation for {model}", flush=True)
        else:
            val_results[model] = gpu_evaluate.remote(run_id, model=model, split="val")
        runs_volume.commit()
    model_selection_path = run_dir / f"model_selection{suffix}.json"
    if model_selection_path.is_file():
        model_selection = json.loads(model_selection_path.read_text(encoding="utf-8"))
        selection = model_selection["selected_model"]
        print("Reusing frozen validation model selection", flush=True)
    else:
        selection = min(
            val_results,
            key=lambda model: val_results[model]["val_loss_group_mean"],
        )
        model_selection = {
            "run_id": run_id,
            "selection_split": "val",
            "selection_metric": "val_loss_group_mean",
            "selected_model": selection,
            "validation": {
                model: {
                    "val_loss_group_mean": result["val_loss_group_mean"],
                    "source_image_level_primary": result["source_image_level_primary"],
                    "source_image_by_detector": result["source_image_by_detector"],
                }
                for model, result in val_results.items()
            },
            "frozen_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(model_selection, model_selection_path)
        runs_volume.commit()

    test_results = {}
    for model, _ in models:
        metrics_path = run_dir / "evaluation" / model / "test" / "metrics.json"
        if metrics_path.is_file():
            test_results[model] = json.loads(metrics_path.read_text(encoding="utf-8"))
            print(f"Reusing saved test evaluation for {model}", flush=True)
        else:
            test_results[model] = gpu_evaluate.remote(run_id, model=model, split="test")
        runs_volume.commit()
    raw_eval_name = f"raw_eval{suffix}"
    raw_metrics_path = run_dir / raw_eval_name / "metrics.json"
    if raw_metrics_path.is_file():
        raw_result = json.loads(raw_metrics_path.read_text(encoding="utf-8"))
        print("Reusing saved raw evaluation", flush=True)
    else:
        raw_result = gpu_evaluate_raw.remote(
            run_id, model=selection, split="test", tag=tag
        )
    runs_volume.commit()
    parity_path = run_dir / f"raw_parity{suffix}.json"
    if parity_path.is_file():
        parity = json.loads(parity_path.read_text(encoding="utf-8"))
        print("Reusing saved raw parity report", flush=True)
    else:
        parity = raw_parity.remote(run_id, split="test", tag=tag)
    runs_volume.commit()
    live_demo_path = run_dir / (f"live_demo_{tag}.json" if tag else "live_demo.json")
    if live_demo_path.is_file():
        live_demo_result = json.loads(live_demo_path.read_text(encoding="utf-8"))
        print("Reusing saved live demo", flush=True)
    else:
        live_demo_result = gpu_live_demo.remote(
            run_id,
            model=selection,
            n=12,
            seed=43 if is_v2 else 42,
            tag=tag,
        )
    runs_volume.commit()
    result = {
        "run_id": run_id,
        "gpu_requested": gpu,
        "gpu_actual_by_stage": {
            "stage_a": stage_a.get("gpu_name"),
            **{
                model: (
                    value.get("gpu_name")
                    or value.get("model_card", {}).get("gpu_name")
                )
                for model, value in stage_b_results.items()
            },
        },
        "stage_a": stage_a,
        **stage_b_results,
        "model_selection": model_selection,
        "test_results": {
            model: {
                "metrics_path": str(
                    Path("/runs") / run_id / "evaluation" / model / "test" / "metrics.json"
                ),
                "source_image_level_primary": value["source_image_level_primary"],
                "source_image_by_detector": value["source_image_by_detector"],
                "bootstrap_95_ci": value["bootstrap_95_ci"],
            }
            for model, value in test_results.items()
        },
        "raw_evaluation": raw_result,
        "raw_parity": parity,
        "live_demo": live_demo_result,
    }
    write_json(result, pipeline_summary_path)
    runs_volume.commit()
    return result


@app.local_entrypoint()
def smoke(run_id: str, gpu: str = DEFAULT_GPU, tag: str = ""):
    result = _spawn_wait(smoke_test.with_options(gpu=gpu), run_id, tag=tag)
    smoke_name = f"smoke_{tag}" if tag else "smoke"
    report_path = Path("artifacts") / run_id / smoke_name / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"smoke_report={report_path}")


@app.local_entrypoint()
def adapt(
    run_id: str,
    resume: bool = False,
    tag: str = "",
    gpu: str = DEFAULT_GPU,
):
    result = _spawn_wait(
        adapt_batch3.with_options(gpu=gpu), run_id, resume=resume, tag=tag
    )
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def train(
    run_id: str,
    init: str = "stage_a",
    resume: bool = False,
    microbatch: int = 4,
    accum: int = 4,
    full_encoder: bool = False,
    input_prep: str = "none",
    tag: str = "",
    gpu: str = DEFAULT_GPU,
    detectors: str = "",
):
    result = _spawn_wait(
        train_classifier.with_options(gpu=gpu),
        run_id,
        init=init,
        resume=resume,
        microbatch=microbatch,
        accum=accum,
        full_encoder=full_encoder,
        input_prep=input_prep,
        tag=tag,
        detectors=_parse_detectors(detectors),
    )
    print(json.dumps(result, indent=2))


@app.local_entrypoint(name="eval")
def eval_cli(
    run_id: str,
    model: str = "stage_b_stage_a",
    split: str = "test",
    gpu: str = DEFAULT_GPU,
    detectors: str = "",
):
    result = _spawn_wait(
        evaluate.with_options(gpu=gpu),
        run_id,
        model=model,
        split=split,
        detectors=_parse_detectors(detectors),
    )
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def eval_raw(
    run_id: str,
    model: str,
    tag: str = "",
    gpu: str = DEFAULT_GPU,
    detectors: str = "",
):
    detector_filter = _parse_detectors(detectors)
    result = _spawn_wait(
        evaluate_raw.with_options(gpu=gpu),
        run_id,
        model=model,
        split="test",
        tag=tag,
        detectors=detector_filter,
    )
    parity = (
        _spawn_wait(raw_parity, run_id, split="test", tag=tag)
        if detector_filter is None
        else None
    )
    print(json.dumps({"raw_eval": result, "raw_parity": parity}, indent=2))


@app.local_entrypoint()
def predict(
    run_id: str,
    model: str,
    path: str,
    mode: str = "csv",
    gpu: str = DEFAULT_GPU,
):
    file_path = Path(path)
    result = _spawn_wait(
        predict_remote.with_options(gpu=gpu),
        run_id,
        model,
        file_path.read_bytes(),
        file_path.name,
        mode,
    )
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def pipeline_cli(run_id: str, gpu: str = DEFAULT_GPU, tag: str = ""):
    result = _spawn_wait(pipeline, run_id, gpu=gpu, tag=tag)
    print(json.dumps(result, indent=2))


@app.local_entrypoint(name="pipeline")
def pipeline_entrypoint(run_id: str, gpu: str = DEFAULT_GPU, tag: str = ""):
    result = _spawn_wait(pipeline, run_id, gpu=gpu, tag=tag)
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def live_demo_cli(
    run_id: str,
    model: str,
    n: int = 12,
    seed: int = 42,
    tag: str = "",
    gpu: str = DEFAULT_GPU,
    detectors: str = "",
):
    result = _spawn_wait(
        live_demo.with_options(gpu=gpu),
        run_id,
        model=model,
        n=n,
        seed=seed,
        tag=tag,
        detectors=_parse_detectors(detectors),
    )
    print(json.dumps(result, indent=2))


@app.local_entrypoint(name="live_demo")
def live_demo_entrypoint(
    run_id: str,
    model: str,
    n: int = 12,
    seed: int = 42,
    tag: str = "",
    gpu: str = DEFAULT_GPU,
    detectors: str = "",
):
    result = _spawn_wait(
        live_demo.with_options(gpu=gpu),
        run_id,
        model=model,
        n=n,
        seed=seed,
        tag=tag,
        detectors=_parse_detectors(detectors),
    )
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def fetch(run_id: str, with_checkpoints: bool = False):
    root = Path("artifacts") / run_id
    root.mkdir(parents=True, exist_ok=True)
    entries = runs_volume.listdir(f"/{run_id}", recursive=True)
    downloaded = []
    for entry in entries:
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        relative = Path(entry.path.lstrip("/"))
        try:
            relative = relative.relative_to(run_id)
        except ValueError:
            relative = Path(entry.path).relative_to(f"/{run_id}")
        if not with_checkpoints and relative.suffix == ".pt":
            continue
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as stream:
            for block in runs_volume.read_file(f"/{run_id}/{relative.as_posix()}"):
                stream.write(block)
        downloaded.append(str(destination))
    print(
        json.dumps(
            {
                "run_id": run_id,
                "destination": str(root),
                "with_checkpoints": with_checkpoints,
                "files_downloaded": len(downloaded),
                "files": downloaded,
            },
            indent=2,
        )
    )
