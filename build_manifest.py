"""Build the sample manifest from a directory of ``Batch_<id>/img_<sample>_<view>.tif`` files.

Detector names are normalised: ``ETD`` and ``SE`` both become the ``SE`` view.
Output columns: sample_id, batch_id, view_BSE_path, view_SE_path, view_InLens_path.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

VIEW_ALIASES = {"BSE": "BSE", "ETD": "SE", "SE": "SE", "INLENS": "InLens"}
VIEWS = ["BSE", "SE", "InLens"]
_FILE_RE = re.compile(r"^(?P<sample>img_[a-z0-9]+)_(?P<view>[A-Za-z]+)\.tiff?$")


def build_manifest(data_dir: Path) -> pd.DataFrame:
    rows: dict[str, dict] = {}
    for path in sorted(data_dir.rglob("*.tif*")):
        m = _FILE_RE.match(path.name)
        if not m:
            continue
        view = VIEW_ALIASES.get(m.group("view").upper())
        if view is None:
            continue
        batch = re.sub(r"^Batch_", "", path.parent.name) if path.parent.name.lower().startswith("batch") else ""
        row = rows.setdefault(m.group("sample"), {"sample_id": m.group("sample"), "batch_id": batch})
        row[f"view_{view}_path"] = str(path)
        row[f"view_{view}_detector"] = m.group("view")
    df = pd.DataFrame(rows.values())
    for v in VIEWS:
        df[f"view_{v}_path"] = df.get(f"view_{v}_path", pd.Series(dtype=str))
    cols = ["sample_id", "batch_id"] + [f"view_{v}_path" for v in VIEWS] + [f"view_{v}_detector" for v in VIEWS]
    return df.reindex(columns=cols).sort_values(["batch_id", "sample_id"]).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", default="sample_manifest.csv")
    args = p.parse_args()
    df = build_manifest(Path(args.data_dir))
    df.to_csv(args.out, index=False)
    missing = df[[f"view_{v}_path" for v in VIEWS]].isna().sum().to_dict()
    print(f"{len(df)} samples; per batch {df['batch_id'].value_counts().sort_index().to_dict()}; missing views {missing}")


if __name__ == "__main__":
    main()
