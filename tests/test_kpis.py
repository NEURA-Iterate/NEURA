import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kpis  # noqa: E402
from build_manifest import build_manifest  # noqa: E402


def synthetic_image(rng, h=400, w=600):
    img = np.full((h, w), 60, dtype=np.float32)
    img[50:150, 50:250] = 5          # pore
    img[200:300, 300:380] = 120      # bright particle
    img += rng.normal(0, 3, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def write_sample(tmp_path, batch, sample, se_name="ETD"):
    rng = np.random.default_rng(0)
    d = tmp_path / f"Batch_{batch}"
    d.mkdir(exist_ok=True)
    for view in ("BSE", se_name, "Inlens"):
        tifffile.imwrite(d / f"img_{sample}_{view}.tif", np.stack([synthetic_image(rng)] * 3, -1))


def test_manifest_normalises_detector_names(tmp_path):
    write_sample(tmp_path, 1, "aaa", "ETD")
    write_sample(tmp_path, 2, "bbb", "SE")
    m = build_manifest(tmp_path)
    assert list(m["sample_id"]) == ["img_aaa", "img_bbb"]
    assert list(m["batch_id"]) == ["1", "2"]
    assert m["view_SE_path"].notna().all()
    assert list(m["view_SE_detector"]) == ["ETD", "SE"]


def test_extract_sample_returns_unit_bearing_kpis(tmp_path):
    write_sample(tmp_path, 1, "aaa")
    row = build_manifest(tmp_path).iloc[0].to_dict()
    out = kpis.extract_sample(row, downsample=1)
    assert out["sample_id"] == "img_aaa" and out["batch_id"] == "1"
    kpi_cols = [c for c in out if c.startswith("kpi__")]
    assert all(len(c.split("__")) == 4 for c in kpi_cols)
    assert {c.split("__")[1] for c in kpi_cols} == {"BSE", "SE", "InLens"}
    assert 0.05 < out["kpi__BSE__pore_fraction__frac"] < 0.12   # 100x200 of 400x600 = 0.083
    assert 0.02 < out["kpi__BSE__bright_phase_fraction__frac"] < 0.05  # 100x80 = 0.033
    assert out["qa__BSE__height__px"] == 400
    assert pd.notna(out["kpi__BSE__pore_eq_diameter_d50__px"])


def test_missing_view_is_skipped(tmp_path):
    write_sample(tmp_path, 1, "aaa")
    row = build_manifest(tmp_path).iloc[0].to_dict()
    row["view_InLens_path"] = float("nan")
    out = kpis.extract_sample(row, downsample=1)
    assert not any(c.startswith("kpi__InLens") for c in out)
    assert any(c.startswith("kpi__BSE") for c in out)
