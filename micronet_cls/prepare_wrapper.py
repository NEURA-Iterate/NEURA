import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from micronet_cls.staging import calibration_set_for


REPO_ROOT = Path(__file__).resolve().parent.parent
PREPARE_SCRIPT = REPO_ROOT / "prepare_dataset.py"
NORMALIZE_SCRIPT = REPO_ROOT / "normalize.py"
PREP_ORDER = ("Batch 3", "Batch 1", "Batch 2")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    if os.environ.get("NEURA_GIT_HEAD"):
        return os.environ["NEURA_GIT_HEAD"]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def run_prepare(
    out_dir: Path,
    folders: list[str],
    profile_path: Path,
    tile: int = 512,
    decimals: int = 3,
    log_path: Path | None = None,
):
    out_dir = Path(out_dir)
    profile_path = Path(profile_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path is None:
        log_path = out_dir / "prepare.log"
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(PREPARE_SCRIPT),
        str(out_dir),
        *[str(folder) for folder in folders],
        "--profile",
        str(profile_path),
        "--tile",
        str(tile),
        "--decimals",
        str(decimals),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return command


def _write_completion_marker(prepared_dir: Path, profile_sha: str, tile: int):
    csv_count = sum(1 for _ in prepared_dir.glob("*.csv"))
    marker = {
        "csv_count": csv_count,
        "profile_sha256": profile_sha,
        "tile": tile,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    marker_path = prepared_dir / "_COMPLETE.json"
    temporary = marker_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(marker, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(marker_path)
    return marker


def _folders(staged_root: Path, split: str):
    return [str(Path(staged_root) / split / batch_name) for batch_name in PREP_ORDER]


def prepare_train(run_dir: Path, staged_root: Path, prepared_root: Path):
    run_dir = Path(run_dir)
    staged_root = Path(staged_root)
    prepared_root = Path(prepared_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    profile_path = run_dir / "profile.json"
    if profile_path.exists():
        raise FileExistsError(f"Training profile already exists: {profile_path}")

    calibration_dir = staged_root / "train" / "Batch 3"
    calibration_image = calibration_set_for(calibration_dir)
    stage_map_path = staged_root / "train" / "staging_map.json"
    staging_map = json.loads(stage_map_path.read_text(encoding="utf-8"))
    calibration_sources = {}
    for detector in ("BSE", "ETD", "Inlens"):
        staged_path = f"Batch 3/{calibration_image}_{detector}.tif"
        source = staging_map[staged_path]
        calibration_sources[detector] = {
            "original_relative_path": source["original_relative_path"],
            "sha256": source["sha256"],
            "staged_path": str(calibration_dir / f"{calibration_image}_{detector}.tif"),
        }

    split_out = prepared_root / "train"
    run_prepare(
        split_out,
        _folders(staged_root, "train"),
        profile_path,
        tile=512,
        decimals=3,
        log_path=run_dir / "prepare_train.log",
    )
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile_sha = _sha256(profile_path)
    provenance = {
        "profile": profile,
        "profile_sha256": profile_sha,
        "calibration": {
            "image_name": calibration_image,
            "sources": calibration_sources,
        },
        "prepare_dataset_sha256": _sha256(PREPARE_SCRIPT),
        "normalize_sha256": _sha256(NORMALIZE_SCRIPT),
        "git_head": _git_head(),
        "tile": 512,
        "decimals": 3,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "profile_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    marker = _write_completion_marker(split_out, profile_sha, 512)
    return {"profile": profile, "profile_sha256": profile_sha, "calibration_image_name": calibration_image, **marker}


def prepare_heldout(run_dir: Path, staged_root: Path, prepared_root: Path, split: str):
    if split not in {"val", "test"}:
        raise ValueError(f"Held-out split must be val or test, got {split}")
    run_dir = Path(run_dir)
    profile_path = run_dir / "profile.json"
    provenance_path = run_dir / "profile_provenance.json"
    if not profile_path.is_file():
        raise FileNotFoundError(f"Training profile does not exist: {profile_path}")
    if not provenance_path.is_file():
        raise FileNotFoundError(f"Profile provenance does not exist: {provenance_path}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    before_sha = _sha256(profile_path)
    if before_sha != provenance["profile_sha256"]:
        raise ValueError("Training profile SHA256 does not match profile provenance")

    split_out = Path(prepared_root) / split
    run_prepare(
        split_out,
        _folders(Path(staged_root), split),
        profile_path,
        tile=int(provenance["tile"]),
        decimals=int(provenance["decimals"]),
        log_path=run_dir / f"prepare_{split}.log",
    )
    after_sha = _sha256(profile_path)
    if after_sha != before_sha:
        raise ValueError("Held-out preparation changed the training profile")
    return _write_completion_marker(split_out, before_sha, int(provenance["tile"]))
