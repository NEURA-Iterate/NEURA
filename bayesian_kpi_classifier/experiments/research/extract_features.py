"""Per-image crack representations and tile-level KPIs for rule-based and learned masks.

Writes crack_features.csv (one row per image x source), tile_kpis.csv (one row per
image x source x tile) and downsampled label/crack masks for plotting.
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from anode_qc.cli import load_detectors
from anode_qc.config import Config
from anode_qc.data import find_bse_images
from anode_qc.kpis import crack_mask, kpis_from_labels
from anode_qc.pipeline import segment_image
from anode_qc.segment import CBD, GRAPHITE, PORE, SI
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects

LEARNED_TO_ANODE = np.array([PORE, GRAPHITE, SI, CBD], dtype=np.uint8)
DS = 2
N_TILES = 4
WINDOW = 256
CRACK_KS = (4.0, 6.0, 8.0)


def learned_labels(pred: np.ndarray, cfg: Config) -> np.ndarray:
    labels = LEARNED_TO_ANODE[pred]
    si = labels == SI
    labels[si & ~remove_small_objects(si, min_size=cfg.cleanup.si_min_area_px)] = CBD
    return labels


def crack_features(cracks: np.ndarray, labels: np.ndarray, cfg: Config) -> dict:
    """Crack statistics on the DS-downsampled skeleton; densities in crack px per 1e4 phase px."""
    out = {}
    lab_ds = labels[::DS, ::DS]
    for name, code in (("graphite", GRAPHITE), ("si", SI)):
        m = lab_ds == code
        area = float((labels == code).sum())
        c = cracks & m
        comp, n = ndi.label(c, structure=np.ones((3, 3)))
        length = float(c.sum() * DS)
        out[f"{name}_crack_len_density"] = length / area * 1e4 if area else np.nan
        out[f"{name}_crack_count_density"] = n / area * 1e6 if area else np.nan
        sizes = np.bincount(comp.ravel())[1:] * DS
        out[f"{name}_crack_mean_len"] = float(sizes.mean()) if n else np.nan
        out[f"{name}_crack_p90_len"] = float(np.percentile(sizes, 90)) if n else np.nan
        if name != "graphite":
            continue
        parts, npart = ndi.label(m)
        part_area = np.bincount(parts.ravel())[1:]
        keep = part_area >= cfg.kpis.graphite_min_area_px / DS**2
        crack_px = np.bincount(parts[c], minlength=npart + 1)[1:]
        cracked = (crack_px > 0) & keep
        out["graphite_frac_particles_cracked"] = float(cracked.sum() / max(keep.sum(), 1))
        out["graphite_frac_area_cracked"] = float(part_area[cracked].sum() / max(part_area[keep].sum(), 1))
        big = keep & (part_area * DS**2 >= np.median(part_area[keep] * DS**2)) if keep.any() else keep
        out["graphite_crack_len_per_large_particle"] = float(crack_px[big].sum() * DS / max(big.sum(), 1))
        horiz = total = 0.0
        for r in regionprops(comp):
            if r.area < 5:
                continue
            w = r.area
            total += w
            horiz += w * (abs(r.orientation) > np.pi / 3)
        out["graphite_crack_frac_horizontal"] = horiz / total if total else np.nan
        win = WINDOW // DS
        h, wd = (m.shape[0] // win) * win, (m.shape[1] // win) * win
        cs = c[:h, :wd].reshape(h // win, win, wd // win, win).sum((1, 3))
        ms = m[:h, :wd].reshape(h // win, win, wd // win, win).sum((1, 3))
        ok = ms >= 0.2 * win * win
        dens = cs[ok] / ms[ok]
        out["graphite_crack_cv_w256"] = float(dens.std() / dens.mean()) if ok.sum() > 2 and dens.mean() > 0 else np.nan
    return out


def one(job):
    im, pred_dir, mask_dir = job
    cfg = Config()
    raw, etd, inl = load_detectors(im, cfg)
    res = segment_image(raw, cfg, etd, inl)
    etd_n = res.mm.etd_n
    sources = {"rule": res.labels, "learned": learned_labels(np.load(Path(pred_dir) / f"{im.image_id}.npy"), cfg)}
    crack_rows, tile_rows, save = [], [], {}
    for src, labels in sources.items():
        phases = (labels == SI) | (labels == GRAPHITE)
        row = {"batch": im.batch, "image_id": im.image_id, "source": src}
        for k in CRACK_KS:
            cracks = crack_mask(etd_n, phases, k, DS)
            feats = crack_features(cracks, labels, cfg)
            if k == cfg.multimodal.crack_k:
                row.update(feats)
                save[f"{src}_cracks"] = np.packbits(cracks)
                save[f"{src}_cracks_shape"] = np.array(cracks.shape)
            else:
                row[f"graphite_crack_len_density_k{int(k)}"] = feats["graphite_crack_len_density"]
        crack_rows.append(row)
        save[f"{src}_labels"] = labels[::4, ::4]
        w = labels.shape[1]
        for t in range(N_TILES):
            sl = (slice(None), slice(t * w // N_TILES, (t + 1) * w // N_TILES))
            k, _, _ = kpis_from_labels(labels[sl], res.n[sl], cfg.kpis, etd_n[sl], cfg.multimodal.crack_k)
            tile_rows.append({"batch": im.batch, "image_id": im.image_id, "source": src, "tile": t, **k})
    save["bse"] = (res.n[::4, ::4] * 255).clip(0, 255).astype(np.uint8)
    np.savez_compressed(Path(mask_dir) / f"{im.image_id}.npz", **save)
    print(im.image_id, "done", flush=True)
    return crack_rows, tile_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--pred-dir", type=Path, required=True)
    p.add_argument("--mask-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ids", nargs="*")
    p.add_argument("--procs", type=int, default=4)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    a.mask_dir.mkdir(parents=True, exist_ok=True)
    ims = [im for im in find_bse_images(a.images) if not a.ids or im.image_id in a.ids]
    with Pool(a.procs) as pool:
        results = pool.map(one, [(im, a.pred_dir, a.mask_dir) for im in ims])
    pd.DataFrame([r for c, _ in results for r in c]).to_csv(a.out / "crack_features.csv", index=False)
    pd.DataFrame([r for _, t in results for r in t]).to_csv(a.out / "tile_kpis.csv", index=False)


if __name__ == "__main__":
    main()
