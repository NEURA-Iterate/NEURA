import numpy as np
import pandas as pd
import pytest
from conftest import make_synthetic
from skimage.draw import disk, line

from anode_qc.config import Config
from anode_qc.data import BSEImage
from anode_qc.kpis import clustering_index, dispersion_index, graphite_alignment, kpis_from_labels
from anode_qc.montecarlo import (
    PARAM_RANGES,
    apply_settings,
    batch_robustness,
    draw_settings,
    param_name,
)
from anode_qc.multimodal import veto_porous_si
from anode_qc.pipeline import segment_image
from anode_qc.segment import CBD, GAP, PORE, SI
from anode_qc.uncertainty import pixel_intervals


def make_multimodal(seed=0):
    """BSE + ETD + Inlens with known pore / Si / thin-gap / porous-mesh regions."""
    bse, pore, si = make_synthetic(seed=seed, comb=False)
    rng = np.random.default_rng(seed)
    shape = bse.shape
    gap = np.zeros(shape, bool)
    for r in (100, 300, 500):
        rr, cc = line(r, 50, r + 20, 400)
        for d in (0, 1):
            gap[np.clip(rr + d, 0, shape[0] - 1), cc] = True
    gap &= ~pore & ~si
    mesh = np.zeros(shape, bool)
    mesh[420:560, 600:860] = True
    mesh &= ~pore & ~si
    holes = mesh & (rng.random(shape) < 0.04)
    holes = (
        np.asarray(np.lib.stride_tricks.sliding_window_view(np.pad(holes, 1), (3, 3)).any(axis=(2, 3)), bool)
        & mesh
    )
    etd = np.full(shape, 90.0)
    etd[si] = 140
    etd[pore | gap | holes] = 6
    etd += rng.normal(0, 5, shape)
    inl = np.full(shape, 60.0)
    inl[si] = 90
    inl[pore | gap | holes] = 10
    inl += rng.normal(0, 5, shape)
    bse = bse.astype(float)
    bse[gap | holes] = 0
    as8 = lambda a: np.clip(a, 0, 255).astype(np.uint8)  # noqa: E731
    return as8(bse), as8(etd), as8(inl), {"pore": pore, "si": si, "gap": gap, "mesh": mesh}


@pytest.fixture(scope="module")
def mm_result():
    bse, etd, inl, truth = make_multimodal()
    return segment_image(bse, Config(), etd, inl), truth


def test_etd_pore_and_gaps(mm_result):
    res, truth = mm_result
    pore = res.labels == PORE
    iou = (pore & truth["pore"]).sum() / (pore | truth["pore"]).sum()
    assert iou > 0.85
    assert (res.labels[truth["gap"]] == GAP).mean() > 0.5


def test_si_kept_and_mesh_found(mm_result):
    res, truth = mm_result
    si = res.labels == SI
    assert (si & truth["si"]).sum() / (si | truth["si"]).sum() > 0.8
    cbd = res.labels == CBD
    assert cbd[truth["mesh"]].mean() > 0.5
    assert cbd[~truth["mesh"]].mean() < 0.02


def test_veto_porous_si():
    si = np.zeros((200, 200), bool)
    rr, cc = disk((50, 50), 30)
    si[rr, cc] = True
    rr, cc = disk((150, 150), 30)
    si[rr, cc] = True
    dark = np.zeros_like(si)
    dark[130:170:4, 130:170:4] = True
    si &= ~dark
    kept, vetoed = veto_porous_si(si, dark, 0.005)
    assert kept[50, 50] and not kept[150, 151] and vetoed[150, 151]


def test_graphite_alignment_direction():
    y, x = np.mgrid[:400, :400]
    horiz = (y // 20) % 2 == 0
    vert = (x // 20) % 2 == 0
    diag = ((x + y) // 28) % 2 == 0  # bands rising to the left as displayed -> tangents at -45 deg
    assert graphite_alignment(horiz, 4)[0] > 0.9
    assert graphite_alignment(vert, 4)[0] < -0.9
    order, coherence, angle = graphite_alignment(diag, 4)
    assert abs(order) < 0.2 and coherence > 0.8 and abs(abs(angle) - 45) < 5


def test_dispersion_and_clustering_indices():
    rng = np.random.default_rng(1)
    shape = (2048, 2048)
    random_pts = rng.uniform(0, 2048, size=(300, 2))
    clumped = np.concatenate([rng.normal(c, 30, size=(60, 2)) for c in rng.uniform(200, 1800, size=(5, 2))])
    clumped = np.clip(clumped, 0, 2047)
    areas = np.full(300, 100.0)
    d_rand, sd = dispersion_index(random_pts, areas, shape, 256, 100, rng)
    d_clump, _ = dispersion_index(clumped, areas, shape, 256, 100, rng)
    assert abs(d_rand - 1) < 3 * sd + 0.1
    assert d_clump > 2
    assert abs(clustering_index(random_pts, shape, 50, rng) - 1) < 0.15
    assert clustering_index(clumped, shape, 50, rng) > 2


def test_mc_settings_within_ranges():
    settings = draw_settings(20, seed=0)
    for sec, f, lo, hi, is_int in PARAM_RANGES:
        v = np.array([s[param_name(sec, f)] for s in settings])
        assert v.min() >= lo and v.max() <= hi
        if is_int:
            assert set(np.unique(v)) <= set(range(int(lo), int(hi) + 1))
    c = apply_settings(Config(), settings[0])
    assert c.kpis.contact_ring_px == (
        settings[0]["kpis.contact_ring_inner"],
        settings[0]["kpis.contact_ring_outer"],
    )
    assert c.cleanup.si_min_area_px == settings[0]["cleanup.si_min_area_px"]


def test_pixel_intervals_bracket_nominal(mm_result):
    res, _ = mm_result
    cfg = Config()

    def fn(lab):
        return kpis_from_labels(lab, res.n, cfg.kpis, res.mm.etd_n)[0]

    nominal = fn(res.labels)
    out = pixel_intervals(res, nominal, fn, cfg.uncertainty)
    for k in ("si_fraction_of_solids", "frac_pore", "si_ecd_d50"):
        assert out[f"{k}_pix_lo"] <= nominal[k] <= out[f"{k}_pix_hi"]
        assert out[f"{k}_pix_hi"] > out[f"{k}_pix_lo"]
    assert 0 < out["qc_edge_blur_px"] < 10


def test_batch_robustness_verdicts():
    rows = []
    rng = np.random.default_rng(0)
    for run in range(20):
        for b, mu in (("A", 0.05), ("B", 0.10)):
            for i in range(8):
                rows.append(
                    {
                        "run": run,
                        "batch": b,
                        "image_id": f"{b}{i}",
                        "x": mu + rng.normal(0, 0.005),
                        "y": rng.normal(),
                    }
                )
    out = batch_robustness(pd.DataFrame(rows), ["x", "y"]).set_index("kpi")
    assert out.loc["x", "verdict"] == "robust difference"
    assert out.loc["y", "share_significant"] < 0.3


def test_detector_sibling_lookup_and_etd_fallback(tmp_path):
    bse = tmp_path / "img_example_BSE.tif"
    bse.touch()
    im = BSEImage("img_example", "Batch_1", bse)
    assert im.sibling("etd") is None
    se = tmp_path / "img_example_SE.tif"
    se.touch()
    assert im.sibling("etd") == se
    etd = tmp_path / "img_example_ETD.tif"
    etd.touch()
    assert im.sibling("etd") == etd
    inlens = tmp_path / "img_example_Inlens.tif"
    inlens.touch()
    assert im.sibling("inlens") == inlens


def test_detector_shape_mismatch_rejected():
    bse, etd, inl, _ = make_multimodal()
    with pytest.raises(ValueError, match="does not match BSE"):
        segment_image(bse, Config(), etd[:-1], inl)
    with pytest.raises(ValueError, match="does not match BSE"):
        segment_image(bse, Config(), etd, inl[:, :-1])
