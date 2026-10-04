#!/usr/bin/env bash
set -euo pipefail
# Labelled samples back the "Re-run" demo dropdown; uploads work without them.
if [[ "${NEURA_DEMO_SAMPLES:-1}" != "0" && -n "${HF_TOKEN:-}" ]] && ! compgen -G "$NEURA_DATA_DIR/Hackathon-Polaron/Batch_*" >/dev/null; then
    python - <<'PY' || echo "Dataset download failed; demo samples will be unavailable." >&2
import os
from pathlib import Path
import neura_app  # noqa: F401  (puts anode_qc on sys.path)
from anode_qc.data import download_dataset
download_dataset(Path(os.environ["NEURA_DATA_DIR"]), token=os.environ["HF_TOKEN"])
PY
fi
cd /app/integration/backend
exec python -m uvicorn neura_app.api:app --host 0.0.0.0 --port "${PORT:-7860}"
