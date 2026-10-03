"""Command-line entry point: ``anode-qc download | run | report``."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from .config import Config
from .data import BSEImage, download_dataset, find_bse_images, load_bse
from .kpis import compute_kpis, kpis_from_labels
from .montecarlo import apply_settings, batch_robustness, draw_settings, summarise
from .pipeline import segment_image
from .qc import save_qc
from .report import build_report
from .uncertainty import KEY_COLUMNS, pixel_intervals

MC_SHUFFLES = 30


def load_detectors(im: BSEImage, cfg: Config) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    raw = load_bse(im.path, cfg.preprocess.border_px)
    if not cfg.multimodal.enabled:
        return raw, None, None
    paths = [im.sibling("etd"), im.sibling("inlens")]
    etd, inl = (load_bse(p, cfg.preprocess.border_px) if p else None for p in paths)
    return raw, etd, inl


def process_image(
    im: BSEImage, cfg: Config, out_dir: Path, qc: bool
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    raw, etd, inl = load_detectors(im, cfg)
    res = segment_image(raw, cfg, etd, inl)
    name = f"{im.batch}_{im.image_id}"
    mask_dir = out_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(res.labels).save(mask_dir / f"{name}_labels.png")
    if qc:
        save_qc(res, out_dir / "qc", name)
    kpis, particles, profile = compute_kpis(res, cfg.kpis, cfg.multimodal.crack_k)
    if cfg.uncertainty.pixel:
        etd_n = res.mm.etd_n if res.mm is not None else None
        kpis.update(
            pixel_intervals(
                res,
                kpis,
                lambda lab: kpis_from_labels(lab, res.n, cfg.kpis, etd_n, cfg.multimodal.crack_k)[0],
                cfg.uncertainty,
                cfg.cleanup.si_min_area_px,
            )
        )
    meta = {"batch": im.batch, "image_id": im.image_id, "height": raw.shape[0], "width": raw.shape[1]}
    particles.insert(0, "image_id", im.image_id)
    particles.insert(0, "batch", im.batch)
    return {**meta, **kpis}, particles, profile


def mc_image(im: BSEImage, cfg: Config, settings: list[dict]) -> list[dict]:
    """KPIs for every algorithm setting (one row per run)."""
    raw, etd, inl = load_detectors(im, cfg)
    rows = []
    for run_id, setting in enumerate(settings):
        c = apply_settings(cfg, setting)
        c.kpis.n_shuffles = MC_SHUFFLES
        k, _, _ = compute_kpis(segment_image(raw, c, etd, inl), c.kpis, c.multimodal.crack_k)
        rows.append(
            {
                "batch": im.batch,
                "image_id": im.image_id,
                "run": run_id,
                **setting,
                **{key: k.get(key, np.nan) for key in KEY_COLUMNS},
            }
        )
    return rows


def run_mc(images: list[BSEImage], cfg: Config, out_dir: Path, workers: int, n_runs: int) -> None:
    settings = draw_settings(n_runs, cfg.uncertainty.mc_seed)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for im, fut in zip(images, [ex.submit(mc_image, im, cfg, settings) for im in images], strict=True):
            rows.extend(fut.result())
            print(f"MC {im.batch}/{im.image_id}: {n_runs} runs", flush=True)
    kdir = out_dir / "kpis"
    runs = pd.DataFrame(rows)
    runs.to_csv(kdir / "mc_runs.csv", index=False)
    per_image = pd.read_csv(kdir / "per_image.csv")
    alg_cols = [c for c in per_image if "_alg_" in c]
    per_image = per_image.drop(columns=alg_cols).merge(
        summarise(runs, KEY_COLUMNS), on=["batch", "image_id"], how="left"
    )
    per_image.to_csv(kdir / "per_image.csv", index=False)
    batch_robustness(runs, KEY_COLUMNS).to_csv(kdir / "batch_robustness.csv", index=False)


def run(
    data: Path,
    out_dir: Path,
    cfg: Config,
    workers: int,
    limit: int | None,
    qc: bool,
    mc_runs: int = 0,
) -> None:
    images = find_bse_images(data)
    if not images:
        raise SystemExit(f"No *_BSE.tif files under {data}")
    if limit:
        images = images[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    rows, parts, profiles = [], [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process_image, im, cfg, out_dir, qc) for im in images]
        for im, fut in zip(images, futures, strict=False):
            k, p, prof = fut.result()
            print(
                f"{im.batch}/{im.image_id}: Si/solids={k['si_fraction_of_solids']:.3f} "
                f"Si particles={k['si_count']} ({k['qc_threshold_method']})",
                flush=True,
            )
            rows.append(k)
            parts.append(p)
            profiles.append(
                {
                    "batch": im.batch,
                    "image_id": im.image_id,
                    **{f"band_{i:02d}": v for i, v in enumerate(prof)},
                }
            )
    kdir = out_dir / "kpis"
    kdir.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(kdir / "per_image.csv", index=False)
    pd.concat(parts, ignore_index=True).to_csv(kdir / "si_particles.csv", index=False)
    pd.DataFrame(profiles).to_csv(kdir / "si_profiles.csv", index=False)
    if mc_runs > 0:
        run_mc(images, cfg, out_dir, workers, mc_runs)
    print(f"Report: {build_report(out_dir)}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="anode-qc", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser(
        "download", help="download BSE / ETD / Inlens images from Hugging Face (needs HF_TOKEN)"
    )
    d.add_argument("--dest", type=Path, default=Path("data"))
    r = sub.add_parser("run", help="segment all BSE images, compute KPIs and write the report")
    r.add_argument("--data", type=Path, default=Path("data"))
    r.add_argument("--out", type=Path, default=Path("outputs"))
    r.add_argument("--config", type=Path, default=None)
    r.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    r.add_argument("--limit", type=int, default=None, help="process only the first N images")
    r.add_argument("--no-qc", action="store_true", help="skip QC overlay/histogram images")
    r.add_argument("--mc-runs", type=int, default=None, help="algorithm-uncertainty runs per image (0 = off)")
    r.add_argument("--no-pixel-uncertainty", action="store_true", help="skip pixel-ambiguity intervals")
    r.add_argument("--bse-only", action="store_true", help="ignore ETD / Inlens images")
    rp = sub.add_parser("report", help="rebuild the report from existing KPI CSVs")
    rp.add_argument("--out", type=Path, default=Path("outputs"))
    args = ap.parse_args(argv)

    if args.cmd == "download":
        print(download_dataset(args.dest, token=os.environ.get("HF_TOKEN")))
    elif args.cmd == "run":
        cfg = Config.from_yaml(args.config)
        cfg.uncertainty.pixel &= not args.no_pixel_uncertainty
        cfg.multimodal.enabled &= not args.bse_only
        mc = cfg.uncertainty.mc_runs if args.mc_runs is None else args.mc_runs
        run(args.data, args.out, cfg, args.workers, args.limit, not args.no_qc, mc)
    else:
        print(build_report(args.out))


if __name__ == "__main__":
    main()
