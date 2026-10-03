"""Plot normalised-intensity histograms of all BSE images (threshold tuning aid)."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from si_graphite_bse.data import find_bse_images, load_bse
from si_graphite_bse.preprocess import denoise, estimate_normalisation, normalise, smoothed_histogram

root, out = Path(sys.argv[1]), Path(sys.argv[2])
imgs = find_bse_images(root)
fig, axes = plt.subplots(len(imgs) // 4 + 1, 4, figsize=(20, 4 * (len(imgs) // 4 + 1)))
for ax, im in zip(axes.ravel(), imgs, strict=False):
    n_img = None
    den = denoise(load_bse(im.path), 1.5)
    norm = estimate_normalisation(den)
    n_img = normalise(den, norm)
    c, h = smoothed_histogram(n_img.ravel()[::5], -0.5, 3.5, 0.01, 2)
    ax.semilogy(c, h + 1)
    ax.set_title(
        f"{im.batch} {im.image_id}\nmode={norm.graphite_mode:.0f} floor={norm.floor:.0f} sn={norm.graphite_sigma_n:.3f}",
        fontsize=8,
    )
    ax.axvline(1, color="k", lw=0.5)
    for q in (1.5, 2.0):
        ax.axvline(q, color="r", lw=0.5)
    print(
        im.batch,
        im.image_id,
        round(norm.floor, 1),
        round(norm.graphite_mode, 1),
        round(norm.graphite_sigma_n, 4),
        "frac>1.4 {:.3f} >1.5 {:.3f} >1.6 {:.3f} >1.8 {:.3f}".format(
            *tuple((n_img > t).mean() for t in (1.4, 1.5, 1.6, 1.8))
        ),
        flush=True,
    )
fig.tight_layout()
fig.savefig(out, dpi=70)
