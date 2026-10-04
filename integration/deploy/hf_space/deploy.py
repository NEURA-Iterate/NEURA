"""Stage the files the classifier app needs and (optionally) push them to a Hugging Face Docker Space.

    python integration/deploy/hf_space/deploy.py --stage-only /tmp/neura_space
    HF_WRITE_TOKEN=... HF_TOKEN=... python integration/deploy/hf_space/deploy.py --space user/neura-classifier

HF_WRITE_TOKEN creates/updates the Space; HF_TOKEN (read access to the dataset) is stored as a Space secret.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
INCLUDE = ("integration", "anode_microstructure_qc/src", "feature_classifier", "bayesian_kpi_classifier")
SKIP = ("integration/frontend/node_modules/", "integration/frontend/dist/", "integration/deploy/")


def stage(dest: Path) -> Path:
    files = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", *INCLUDE],
        cwd=REPO, check=True, capture_output=True, text=True,
    ).stdout.split("\0")
    if dest.exists():
        shutil.rmtree(dest)
    for rel in filter(None, files):
        src = REPO / rel
        if rel.startswith(SKIP) or not src.is_file():
            continue
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest / rel)
    (dest / "integration/deploy/hf_space").mkdir(parents=True, exist_ok=True)
    shutil.copy2(HERE / "start.sh", dest / "integration/deploy/hf_space/start.sh")
    for name in ("Dockerfile", "README.md"):
        shutil.copy2(HERE / name, dest / name)
    (dest / ".dockerignore").write_text("**/node_modules\n**/__pycache__\n")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", help="e.g. user/neura-classifier")
    ap.add_argument("--hardware", default="t4-medium")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--stage-only", type=Path)
    a = ap.parse_args()
    if a.stage_only:
        print(stage(a.stage_only))
        return
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_WRITE_TOKEN"])
    api.create_repo(a.space, repo_type="space", space_sdk="docker", private=not a.public,
                    space_hardware=a.hardware, exist_ok=True)
    if os.environ.get("HF_TOKEN"):
        api.add_space_secret(a.space, "HF_TOKEN", os.environ["HF_TOKEN"])
    with tempfile.TemporaryDirectory() as tmp:
        api.upload_folder(repo_id=a.space, repo_type="space", folder_path=stage(Path(tmp) / "space"),
                          commit_message="Deploy NEURA classifier", delete_patterns="*")
    api.request_space_hardware(a.space, a.hardware)
    print(f"https://huggingface.co/spaces/{a.space}")


if __name__ == "__main__":
    main()
