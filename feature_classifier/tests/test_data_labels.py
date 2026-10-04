from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import tifffile
from anode_qc.config import Config
from anode_qc.data import BSEImage
from anode_qc.segment import CBD, GAP, GRAPHITE, PORE, SI
from PIL import Image

from phaseseg import CLASSES
from phaseseg import labels as labels_module
from phaseseg.data import load_triplet
from phaseseg.labels import IGNORE, load_scribbles, map_anode_labels, pseudo_labels


def test_load_triplet_crops_border_normalizes_and_repeats_missing_detectors(tmp_path) -> None:
    image_id = "sem"
    bse_path = tmp_path / f"{image_id}_BSE.tif"
    source = np.arange(40 * 48, dtype=np.uint16).reshape(40, 48)
    bse = (source % 256).astype(np.uint8)
    etd = np.flipud(bse)
    inlens = np.fliplr(bse)
    tifffile.imwrite(bse_path, bse)
    tifffile.imwrite(tmp_path / f"{image_id}_ETD.tif", etd)
    tifffile.imwrite(tmp_path / f"{image_id}_Inlens.tif", inlens)

    triplet = load_triplet(BSEImage(image_id, "Batch_1", bse_path))
    assert triplet.shape == (3, 36, 44)
    assert triplet.dtype == np.float32
    assert np.allclose(triplet.min(axis=(1, 2)), 0)
    assert np.allclose(triplet.max(axis=(1, 2)), 1)

    for path in tmp_path.glob("*_ETD.tif"):
        path.unlink()
    for path in tmp_path.glob("*_Inlens.tif"):
        path.unlink()
    repeated = load_triplet(BSEImage(image_id, "Batch_1", bse_path))
    assert np.array_equal(repeated[0], repeated[1])
    assert np.array_equal(repeated[0], repeated[2])


def test_rule_label_mapping_boundary_ignore_and_scribble_formats(tmp_path, monkeypatch) -> None:
    raw_labels = np.full((28, 28), GRAPHITE, dtype=np.uint8)
    raw_labels[:, 14:] = SI
    raw_labels[3:8, 3:8] = PORE
    raw_labels[18:22, 3:8] = CBD
    raw_labels[18:22, 8:12] = GAP
    mapped = map_anode_labels(raw_labels)
    assert mapped[0, 0] == CLASSES.index("graphite")
    assert mapped[0, 20] == CLASSES.index("si")
    assert mapped[4, 4] == CLASSES.index("pore")
    assert mapped[19, 4] == CLASSES.index("binder")
    assert mapped[19, 9] == CLASSES.index("pore")

    bse_path = tmp_path / "sem_BSE.tif"
    tifffile.imwrite(bse_path, np.full((32, 32), 128, dtype=np.uint8))
    monkeypatch.setattr(
        labels_module,
        "segment_image",
        lambda raw, cfg, etd=None, inlens=None: SimpleNamespace(labels=raw_labels),
    )
    cfg = Config()
    pseudo = pseudo_labels(BSEImage("sem", "Batch_1", bse_path), cfg, ignore_boundary_px=2)
    assert pseudo.shape == (28, 28)
    assert pseudo[10, 1] == CLASSES.index("graphite")
    assert pseudo[2, 24] == CLASSES.index("si")
    assert pseudo[10, 13] == IGNORE
    assert set(np.unique(pseudo)).issubset({0, 1, 2, 3, IGNORE})

    scribbles = np.full((8, 9), IGNORE, dtype=np.uint8)
    scribbles[2:4, 3:6] = CLASSES.index("si")
    npy_path = tmp_path / "expert.npy"
    png_path = tmp_path / "expert.png"
    np.save(npy_path, scribbles)
    Image.fromarray(scribbles).save(png_path)
    assert np.array_equal(load_scribbles(npy_path), scribbles)
    assert np.array_equal(load_scribbles(png_path), scribbles)
