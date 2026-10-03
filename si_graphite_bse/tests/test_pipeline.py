import numpy as np
import pytest
from conftest import make_synthetic

from si_graphite_bse.config import Config
from si_graphite_bse.kpis import compute_kpis, homogeneity_cv
from si_graphite_bse.pipeline import segment_image
from si_graphite_bse.segment import PORE, SI


def _iou(a, b):
    return (a & b).sum() / (a | b).sum()


def test_normalisation_finds_graphite_peak(synthetic):
    img, _, _ = synthetic
    res = segment_image(img, Config())
    assert res.norm.graphite_mode == pytest.approx(55, abs=2)
    assert res.thresholds.method == "valley"


@pytest.mark.parametrize("floor", [0, 24])
def test_si_and_pore_recovered(floor):
    img, pore, si = make_synthetic(floor=floor)
    res = segment_image(img, Config())
    assert _iou(res.labels == SI, si) > 0.9
    assert _iou(res.labels == PORE, pore) > 0.85


def test_contrast_invariance():
    img_a, _, si = make_synthetic(graphite=50, si=105)
    img_b, _, _ = make_synthetic(graphite=62, si=125)
    fa = (segment_image(img_a, Config()).labels == SI).mean()
    fb = (segment_image(img_b, Config()).labels == SI).mean()
    assert fa == pytest.approx(si.mean(), rel=0.1)
    assert fb == pytest.approx(fa, rel=0.05)


def test_edge_rims_not_counted_as_si():
    img, _, si = make_synthetic(rims=True)
    res = segment_image(img, Config())
    assert res.si_raw.sum() > res.si.sum()
    assert (res.labels == SI).mean() == pytest.approx(si.mean(), rel=0.1)


def test_kpis(synthetic):
    img, pore, si = synthetic
    res = segment_image(img, Config())
    k, parts, profile = compute_kpis(res, Config().kpis)
    assert k["frac_si"] + k["frac_graphite"] + k["frac_pore"] == pytest.approx(1)
    assert k["si_fraction_of_solids"] == pytest.approx(si.sum() / (~pore).sum(), rel=0.1)
    assert k["size_unit"] == "px"
    assert 10 < k["si_ecd_d50"] < 40
    assert len(profile) == Config().kpis.profile_bands
    assert k["qc_threshold_method"] == "valley"


def test_homogeneity_cv_uniform_vs_clumped():
    solids = np.ones((512, 512), bool)
    uniform = np.zeros_like(solids)
    uniform[::8, ::8] = True
    clumped = np.zeros_like(solids)
    clumped[:64, :64] = True
    assert homogeneity_cv(uniform, solids, 128, 0.5) < 0.01
    assert homogeneity_cv(clumped, solids, 128, 0.5) > 1


def test_config_roundtrip(tmp_path):
    import yaml

    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"segment": {"si_k_low": 5.0}}))
    cfg = Config.from_yaml(p)
    assert cfg.segment.si_k_low == 5.0
    assert cfg.cleanup.si_open_radius == 1
    p.write_text(yaml.safe_dump({"bogus": {}}))
    with pytest.raises(ValueError):
        Config.from_yaml(p)


def test_batch_stats_handles_constant_kpi():
    import pandas as pd

    from si_graphite_bse.report import batch_stats

    df = pd.DataFrame(
        {
            "batch": ["A"] * 3 + ["B"] * 3,
            "image_id": list("abcdef"),
            "frac_si": [0.05, 0.06, 0.055, 0.08, 0.09, 0.085],
            "si_fraction_of_solids": [0.1] * 6,
        }
    )
    st = batch_stats(df).set_index(["kpi", "comparison"])
    assert st.loc[("frac_si", "all (Kruskal-Wallis)"), "p_value"] < 0.1
    assert st.loc[("si_fraction_of_solids", "all (Kruskal-Wallis)"), "p_value"] == 1.0
