"""
Prepare a dataset from folders of SEM image sets.

For every folder:
  1. find all image sets: <image_name>_BSE.tif, <image_name>_ETD.tif, <image_name>_Inlens.tif
  2. normalize each whole image set with normalize.py
  3. cut it into 512 x 512 squares (left to right, top to bottom; leftover edge pixels are skipped)
  4. save every square of every detector as a CSV (512 rows x 512 columns, values 0-1):
        {folder_name}-{image_name}_y{Y}_x{X}-{filter_type}.csv
     where Y, X = top-left pixel of the square in the full image

The fixed scale (profile.json) is learned from the FIRST image set if the profile file does not exist yet,
so put the baseline folder first. Later runs reuse the same profile.json.

Usage:
  python prepare_dataset.py OUT_DIR BASELINE_FOLDER [MORE_FOLDERS ...]
  python prepare_dataset.py OUT_DIR FOLDER --profile baseline_profile.json --tile 512 --decimals 3
"""
import argparse
import json
import os
import re

import numpy as np
from PIL import Image

from normalize import DETECTORS, normalize_images

Image.MAX_IMAGE_PIXELS = None
FILE_PATTERN = re.compile(r"^(.*)_(BSE|ETD|In-?lens)\.tiff?$", re.IGNORECASE)


def load_image(path):
    """Read a TIFF as a 2D grey image (the files are stored as RGB with 3 identical copies -> keep one)."""
    a = np.array(Image.open(path))
    return a[..., 0] if a.ndim == 3 else a


def find_image_sets(folder):
    """{image_name: {"BSE": path, "ETD": path, "Inlens": path}} for every complete set in the folder."""
    sets = {}
    for f in sorted(os.listdir(folder)):
        m = FILE_PATTERN.match(f)
        if m:
            name, det = m.group(1), m.group(2).lower()
            det = {"bse": "BSE", "etd": "ETD"}.get(det, "Inlens")
            sets.setdefault(name, {})[det] = os.path.join(folder, f)
    complete = {}
    for name, files in sets.items():
        missing = [d for d in DETECTORS if d not in files]
        if missing:
            print(f"  skipping {name}: missing {missing}")
        else:
            complete[name] = files
    return complete


def tile_positions(height, width, size):
    """Top-left corners of non-overlapping size x size squares."""
    return [(y, x) for y in range(0, height - size + 1, size) for x in range(0, width - size + 1, size)]


def process_folder(folder, out_dir, profile, profile_path, tile=512, decimals=3):
    folder_name = os.path.basename(os.path.normpath(folder))
    image_sets = find_image_sets(folder)
    print(f"{folder_name}: {len(image_sets)} image set(s)")
    for image_name, files in image_sets.items():
        images = {d: load_image(files[d]) for d in DETECTORS}
        norm, used_profile, info = normalize_images(images, profile)
        if profile is None:                                                 # first set ever -> it defines the scale
            profile = used_profile
            with open(profile_path, "w") as f:
                json.dump(profile, f, indent=2)
            print(f"  scale learned from {folder_name}/{image_name}, saved to {profile_path}: "
                  f"{ {d: round(v, 4) for d, v in profile.items()} }")

        h, w = norm["BSE"].shape
        positions = tile_positions(h, w, tile)
        for y, x in positions:
            for d in DETECTORS:
                out = os.path.join(out_dir, f"{folder_name}-{image_name}_y{y}_x{x}-{d}.csv")
                np.savetxt(out, norm[d][y:y + tile, x:x + tile], delimiter=",", fmt=f"%.{decimals}f")
        lighting = ", ".join(f"{d} {info[d]['uneven lighting (max/min)']:.2f}" for d in DETECTORS)
        stuck = ", ".join(f"{d} {info[d]['% stuck at max in raw']:.1f}%" for d in DETECTORS)
        print(f"  {image_name}: {w}x{h} px -> {len(positions)} squares x 3 detectors = {3 * len(positions)} CSVs "
              f"| uneven lighting {lighting} | stuck pixels {stuck}")
    return profile


def main():
    p = argparse.ArgumentParser(description="Normalize SEM image sets and save 512x512 squares as CSV.")
    p.add_argument("out_dir", help="where the CSV files are written")
    p.add_argument("folders", nargs="+", help="folders with *_BSE.tif, *_ETD.tif, *_Inlens.tif (baseline first)")
    p.add_argument("--profile", help="profile.json with the fixed scale (default: OUT_DIR/profile.json)")
    p.add_argument("--tile", type=int, default=512, help="square size in pixels (default 512)")
    p.add_argument("--decimals", type=int, default=3, help="decimals written to CSV (default 3)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    profile_path = args.profile or os.path.join(args.out_dir, "profile.json")
    profile = None
    if os.path.exists(profile_path):
        with open(profile_path) as f:
            profile = json.load(f)
        print(f"using scale from {profile_path}: {profile}")

    for folder in args.folders:
        profile = process_folder(folder, args.out_dir, profile, profile_path, args.tile, args.decimals)


if __name__ == "__main__":
    main()
