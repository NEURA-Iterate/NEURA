import numpy as np
import pytest
from skimage.draw import disk, rectangle


def make_synthetic(shape=(600, 900), floor=0, graphite=55, si=115, noise=6.0, seed=0, comb=True, rims=False):
    """Synthetic BSE-like image with known pore / Si masks."""
    rng = np.random.default_rng(seed)
    img = np.full(shape, graphite, dtype=np.float32)
    pore = np.zeros(shape, bool)
    si_mask = np.zeros(shape, bool)
    for _ in range(8):
        r0, c0 = rng.integers(0, shape[0] - 60), rng.integers(0, shape[1] - 120)
        rr, cc = rectangle((r0, c0), extent=(rng.integers(15, 50), rng.integers(40, 110)), shape=shape)
        pore[rr, cc] = True
    for _ in range(25):
        rr, cc = disk(
            (rng.integers(20, shape[0] - 20), rng.integers(20, shape[1] - 20)),
            rng.integers(6, 18),
            shape=shape,
        )
        si_mask[rr, cc] = True
    si_mask &= ~pore
    img[pore] = floor
    img[si_mask] = si
    if rims:
        edge = np.zeros(shape, bool)
        edge[:-1] |= pore[1:] != pore[:-1]
        edge[:, :-1] |= pore[:, 1:] != pore[:, :-1]
        img[edge & ~pore] = si
    img += rng.normal(0, noise, shape)
    img = np.clip(img, 0, 255)
    if comb:
        img = np.clip(np.round(np.round(img / 3) * 3), 0, 255)
    return img.astype(np.uint8), pore, si_mask


@pytest.fixture
def synthetic():
    return make_synthetic()
