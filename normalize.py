"""
Graphite-anchored normalization for one SEM image set (BSE, ETD, In-lens).

    norm, profile, info = normalize_images({"BSE": bse, "ETD": etd, "Inlens": inlens}, profile=None)

Input
    images  : dict with three 2D raw images (uint8 or uint16), all the same size
    profile : fixed scale per detector, e.g. {"BSE": 0.39, "ETD": 0.41, "Inlens": 0.43}.
              None -> learned from this image set (use this for the baseline only).
              For every other image set pass the baseline's profile, so all batches share one scale.
Output
    norm    : dict with three separate 2D float32 images, values 0-1 (graphite ~ 0.4)
    profile : the scale that was used (save it and reuse it)
    info    : per-detector numbers for QC (black level, uneven lighting, stuck pixels, ...)
"""
import numpy as np
from scipy import ndimage as ndi

DETECTORS = ("BSE", "ETD", "Inlens")


def graphite_mask(bse):
    """3a. Yes/no map of graphite pixels, found from BSE relative to the typical brightness of each row."""
    b0 = np.percentile(bse, 0.1)
    s = ndi.median_filter(bse, 5) - b0                                     # light denoise
    row_typical = ndi.uniform_filter1d(np.median(s, axis=1), 101, mode="nearest")[:, None]
    m = (s > 0.75 * row_typical) & (s < 1.30 * row_typical)                 # not pore (dark), not silicon (bright)
    return ndi.binary_erosion(m, iterations=3)                              # drop particle edges / thin gaps


def black_level(img):
    """3b. Darkest 0.1% of pixels = detector black level (removes the brightness knob)."""
    return float(np.percentile(img, 0.1))


def graphite_surface(img, mask, block=96, min_frac=0.3):
    """3c. Graphite brightness at every pixel: block medians -> smooth 2nd-order surface."""
    h, w = img.shape
    ys, xs, vs = [], [], []
    for y in range(0, h - block + 1, block):
        for x in range(0, w - block + 1, block):
            m = mask[y:y + block, x:x + block]
            if m.mean() >= min_frac:                                        # only squares with enough graphite
                ys.append(y + block / 2); xs.append(x + block / 2)
                vs.append(np.median(img[y:y + block, x:x + block][m]))
    Y, X, V = np.array(ys) / h - 0.5, np.array(xs) / w - 0.5, np.array(vs)
    A = np.stack([np.ones_like(Y), Y, X, Y * Y, X * Y, X * X], 1)
    keep = np.ones(len(V), bool)
    for _ in range(2):                                                      # ignore odd squares (outliers)
        coef, *_ = np.linalg.lstsq(A[keep], V[keep], rcond=None)
        res = V - A @ coef
        keep = np.abs(res) < 3 * 1.4826 * np.median(np.abs(res[keep])) + 1e-6
    gy = (np.arange(h, dtype=np.float32) / h - 0.5)[:, None]
    gx = (np.arange(w, dtype=np.float32) / w - 0.5)[None, :]
    surf = coef[0] + coef[1] * gy + coef[2] * gx + coef[3] * gy * gy + coef[4] * gx * gy + coef[5] * gx * gx
    return np.maximum(surf.astype(np.float32), 1e-3)


def normalize_images(images, profile=None):
    """Normalize one image set. Returns (norm, profile, info); norm = {"BSE": 2D, "ETD": 2D, "Inlens": 2D}."""
    raw_int = {d: images[d] for d in DETECTORS}
    shape = raw_int["BSE"].shape
    if any(raw_int[d].shape != shape for d in DETECTORS):
        raise ValueError("BSE, ETD and Inlens images must have the same size")
    raw = {d: raw_int[d].astype(np.float32) for d in DETECTORS}
    # highest possible value of the raw file (255 for 8-bit, 65535 for 16-bit); pixels there are "stuck"
    maxval = {d: np.iinfo(raw_int[d].dtype).max if np.issubdtype(raw_int[d].dtype, np.integer) else np.inf
              for d in DETECTORS}

    # --- run 3a-3d on the whole image
    gmask = graphite_mask(raw["BSE"])
    anchored, info = {}, {}
    for d in DETECTORS:
        b = black_level(raw[d])
        surf = graphite_surface(raw[d] - b, gmask)
        a = (raw[d] - b) / surf                                             # 3d: divide by graphite brightness here
        anchored[d] = a / np.median(a[gmask])                               #     fine-tune so graphite = exactly 1.0
        info[d] = {"black level": b, "graphite (min)": float(surf.min()), "graphite (max)": float(surf.max()),
                   "uneven lighting (max/min)": float(surf.max() / surf.min())}

    # --- 3e. fixed scale: learn from this set if no baseline profile was given
    stuck = {d: raw_int[d] >= maxval[d] for d in DETECTORS}                # pixels maxed out in the raw file
    if profile is None:
        # scale so the baseline's brightest 0.1% (ignoring stuck pixels) lands at 1.0
        profile = {d: float(1.0 / np.percentile(anchored[d][::4, ::4][~stuck[d][::4, ::4]], 99.9))
                   for d in DETECTORS}

    norm = {}
    for d in DETECTORS:
        x = anchored[d] * profile[d]
        x[stuck[d]] = 1.0                                                   # 3f: true value unknown -> "at least 1"
        norm[d] = np.clip(x, 0, 1).astype(np.float32)
        info[d]["fixed scale"] = profile[d]
        info[d]["% stuck at max in raw"] = 100 * float(stuck[d].mean())
        info[d]["% pixels at 1 after"] = 100 * float((norm[d] >= 1).mean())
    info["graphite reference area %"] = 100 * float(gmask.mean())
    return norm, profile, info
