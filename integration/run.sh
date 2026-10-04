#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
NEURA_DATA_DIR="${NEURA_DATA_DIR:-/home/ubuntu/data/neura}"
LOCAL_NODE_BIN="$HOME/.local/node/bin"
PORT="${PORT:-8000}"

export NEURA_DATA_DIR
cd "$REPO_ROOT"

has_batch_dir() {
    local batch_dir
    for batch_dir in "$NEURA_DATA_DIR"/Hackathon-Polaron/Batch_*; do
        if [[ -d "$batch_dir" ]]; then
            return 0
        fi
    done
    return 1
}

if ! has_batch_dir; then
    if [[ -z "${HF_TOKEN:-}" ]]; then
        printf 'No samples found under %s/Hackathon-Polaron/Batch_*; set HF_TOKEN to download the private dataset.\n' \
            "$NEURA_DATA_DIR" >&2
        exit 1
    fi
    PYTHONPATH="$REPO_ROOT/anode_microstructure_qc/src${PYTHONPATH:+:$PYTHONPATH}" \
        python3 - <<'PY'
import os
from pathlib import Path

from anode_qc.data import download_dataset

download_dataset(
    Path(os.environ["NEURA_DATA_DIR"]),
    token=os.environ["HF_TOKEN"],
)
PY
    if ! has_batch_dir; then
        printf 'Dataset download completed, but no Batch_* directories were found under %s.\n' \
            "$NEURA_DATA_DIR" >&2
        exit 1
    fi
fi

if ! python3 - <<'PY' >/dev/null 2>&1
import fastapi
import multipart
import skimage
import tifffile
import torch
import transformers
import uvicorn
PY
then
    python3 -m pip install -e "$BACKEND_DIR"
fi

if ! command -v npm >/dev/null 2>&1; then
    if [[ ! -x "$LOCAL_NODE_BIN/npm" ]]; then
        mkdir -p "$HOME/.local/node"
        node_archive="$(mktemp)"
        trap 'rm -f "$node_archive"' EXIT
        curl --fail --location --silent --show-error \
            https://nodejs.org/dist/v20.18.1/node-v20.18.1-linux-x64.tar.xz \
            --output "$node_archive"
        tar -xJf "$node_archive" --strip-components=1 -C "$HOME/.local/node"
        rm -f "$node_archive"
        trap - EXIT
    fi
    export PATH="$LOCAL_NODE_BIN:$PATH"
fi

cd "$FRONTEND_DIR"
if [[ ! -d node_modules ]]; then
    npm ci
fi
npm run build

cd "$BACKEND_DIR"
exec python3 -m uvicorn neura_app.api:app --host 0.0.0.0 --port "$PORT"
