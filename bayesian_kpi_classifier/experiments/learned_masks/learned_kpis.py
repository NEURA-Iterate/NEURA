"""Recompute anode_qc KPIs from learned 4-class masks (or the rule masks, for a sanity check)."""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.morphology import remove_small_objects

from anode_qc.cli import load_detectors
from anode_qc.config import Config
from anode_qc.data import find_bse_images
from anode_qc.kpis import kpis_from_labels
from anode_qc.pipeline import segment_image
from anode_qc.segment import CBD, GRAPHITE, PORE, SI

LEARNED_TO_ANODE = np.array([PORE, GRAPHITE, SI, CBD], dtype=np.uint8)  # phaseseg order: pore, graphite, si, binder


def one(job):
    im, mode, pred_dir = job
    cfg = Config()
    raw, etd, inl = load_detectors(im, cfg)
    res = segment_image(raw, cfg, etd, inl)
    etd_n = res.mm.etd_n if res.mm is not None else None
    if mode == "rule":
        labels = res.labels
    else:
        pred = np.load(Path(pred_dir) / f"{im.image_id}.npy")
        if pred.shape != res.labels.shape:
            raise ValueError(f"{im.image_id}: prediction {pred.shape} vs image {res.labels.shape}")
        labels = LEARNED_TO_ANODE[pred]
        si = labels == SI
        specks = si & ~remove_small_objects(si, min_size=cfg.cleanup.si_min_area_px)
        labels = labels.copy()
        labels[specks] = CBD
    k, _, _ = kpis_from_labels(labels, res.n, cfg.kpis, etd_n, cfg.multimodal.crack_k)
    print(im.image_id, "done", flush=True)
    return {"batch": im.batch, "image_id": im.image_id, **k}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--mode", choices=("rule", "learned"), required=True)
    p.add_argument("--pred-dir", type=Path)
    p.add_argument("--base", type=Path, required=True, help="rule-based per_image.csv")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ids", nargs="*")
    p.add_argument("--procs", type=int, default=3)
    a = p.parse_args()
    ims = [im for im in find_bse_images(a.images) if not a.ids or im.image_id in a.ids]
    with Pool(a.procs) as pool:
        rows = pool.map(one, [(im, a.mode, a.pred_dir) for im in ims])
    new = pd.DataFrame(rows).set_index("image_id")
    base = pd.read_csv(a.base).set_index("image_id").loc[new.index]
    out = base.copy()
    shared = [c for c in new.columns if c in base.columns and c not in ("batch",)]
    out[shared] = new[shared]
    out.reset_index().to_csv(a.out, index=False)
    print(f"replaced {len(shared)} columns for {len(new)} images", file=sys.stderr)


if __name__ == "__main__":
    main()
