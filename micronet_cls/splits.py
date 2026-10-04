import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_COUNTS = {
    "Batch 1": (4, 1, 2),
    "Batch 2": (4, 1, 1),
    "Batch 3": (8, 3, 3),
}
SPLITS = ("train", "val", "test")


def assign_group_splits(
    manifest: pd.DataFrame,
    seed: int = 42,
    counts: dict | None = None,
) -> pd.DataFrame:
    counts = DEFAULT_COUNTS if counts is None else counts
    result = manifest.copy()
    result["split"] = "excluded"
    complete = result[result["complete_set"].astype(bool)]
    rng = np.random.default_rng(seed)

    for batch_name in sorted(counts):
        batch_groups = sorted(complete.loc[complete["batch_name"] == batch_name, "group_id"].unique())
        requested = tuple(counts[batch_name])
        if len(requested) != 3:
            raise ValueError(f"{batch_name} must have train/val/test counts, got {requested}")
        if sum(requested) != len(batch_groups):
            raise ValueError(
                f"{batch_name} split counts {requested} sum to {sum(requested)}, "
                f"but inventory has {len(batch_groups)} complete groups"
            )
        shuffled = list(np.asarray(batch_groups, dtype=object)[rng.permutation(len(batch_groups))])
        start = 0
        for split_name, count in zip(SPLITS, requested):
            selected = shuffled[start : start + count]
            result.loc[result["group_id"].isin(selected), "split"] = split_name
            start += count

    unconfigured = sorted(set(complete["batch_name"]) - set(counts))
    if unconfigured:
        raise ValueError(f"No split counts provided for batches: {unconfigured}")
    return result


def save_split_manifest(df: pd.DataFrame, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_split_manifest(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def split_manifest_sha256(source: Path | pd.DataFrame) -> str:
    content = (
        source.to_csv(index=False).encode("utf-8")
        if isinstance(source, pd.DataFrame)
        else Path(source).read_bytes()
    )
    return hashlib.sha256(content).hexdigest()


def check_split_integrity(df: pd.DataFrame) -> dict:
    included = df[df["split"] != "excluded"].copy()
    group_splits = included.groupby("group_id")["split"].nunique()
    if (group_splits > 1).any():
        raise ValueError(f"Groups occur in multiple splits: {group_splits[group_splits > 1].index.tolist()}")

    hash_splits = included.groupby("sha256")["split"].nunique()
    if (hash_splits > 1).any():
        raise ValueError(f"Source hashes occur in multiple splits: {hash_splits[hash_splits > 1].index.tolist()}")

    class_ids = set(included["class_id"].dropna().astype(int))
    for split_name in SPLITS:
        split_classes = set(
            included.loc[included["split"] == split_name, "class_id"].dropna().astype(int)
        )
        if split_classes != {0, 1, 2}:
            raise ValueError(
                f"Split {split_name} must contain all three classes; found {sorted(split_classes)}"
            )
    if class_ids != {0, 1, 2}:
        raise ValueError(f"Manifest must contain classes 0, 1, and 2; found {sorted(class_ids)}")

    summary = {}
    for split_name in SPLITS:
        summary[split_name] = {}
        for batch_name in sorted(df["batch_name"].dropna().unique()):
            summary[split_name][batch_name] = {}
            for detector in sorted(df["detector"].dropna().unique()):
                selected = included[
                    (included["split"] == split_name)
                    & (included["batch_name"] == batch_name)
                    & (included["detector"] == detector)
                ]
                summary[split_name][batch_name][detector] = {
                    "groups": int(selected["group_id"].nunique()),
                    "images": int(selected["source_image_id"].nunique()),
                }
    return summary
