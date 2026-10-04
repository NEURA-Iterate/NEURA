from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
for source in (
    REPO_ROOT / "feature_classifier",
    REPO_ROOT / "anode_microstructure_qc" / "src",
    REPO_ROOT / "bayesian_kpi_classifier",
):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
