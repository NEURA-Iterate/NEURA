"""Command-line entry point: ``sgb download | run | report``."""

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
from .kpis import compute_kpis
from .pipeline import segment_image
from .qc import save_qc
from .report import build_report


def process_image(
    im: BSEImage, cfg: Config, out_dir: Path, qc: bool
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    raw = load_bse(im.path, cfg.preprocess.border_px)
    res = segment_image(raw, cfg)
    name = f"{im.batch}_{im.image_id}"
    mask_dir = out_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(res.labels).save(mask_dir / f"{name}_labels.png")
    if qc:
        save_qc(res, out_dir / "qc", name)
    kpis, particles, profile = compute_kpis(res, cfg.kpis)
    meta = {"batch": im.batch, "image_id": im.image_id, "height": raw.shape[0], "width": raw.shape[1]}
    particles.insert(0, "image_id", im.image_id)
    particles.insert(0, "batch", im.batch)
    return {**meta, **kpis}, particles, profile


def run(data: Path, out_dir: Path, cfg: Config, workers: int, limit: int | None, qc: bool) -> None:
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
    print(f"Report: {build_report(out_dir)}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="sgb", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="download the BSE images from Hugging Face (needs HF_TOKEN)")
    d.add_argument("--dest", type=Path, default=Path("data"))
    r = sub.add_parser("run", help="segment all BSE images, compute KPIs and write the report")
    r.add_argument("--data", type=Path, default=Path("data"))
    r.add_argument("--out", type=Path, default=Path("outputs"))
    r.add_argument("--config", type=Path, default=None)
    r.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    r.add_argument("--limit", type=int, default=None, help="process only the first N images")
    r.add_argument("--no-qc", action="store_true", help="skip QC overlay/histogram images")
    rp = sub.add_parser("report", help="rebuild the report from existing KPI CSVs")
    rp.add_argument("--out", type=Path, default=Path("outputs"))
    args = ap.parse_args(argv)

    if args.cmd == "download":
        print(download_dataset(args.dest, token=os.environ.get("HF_TOKEN")))
    elif args.cmd == "run":
        run(args.data, args.out, Config.from_yaml(args.config), args.workers, args.limit, not args.no_qc)
    else:
        print(build_report(args.out))


if __name__ == "__main__":
    main()
