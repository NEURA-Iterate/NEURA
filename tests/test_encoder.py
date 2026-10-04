import numpy as np
import pandas as pd

import evaluation as ev
from encoder import tiles


def test_tiles_cover_image_without_remainder():
    img = np.random.rand(500, 1000).astype(np.float32)
    t = tiles(img, 224, 224)
    assert t.shape == (2 * 4, 224, 224)


def test_tiles_small_image_returns_single_crop():
    img = np.random.rand(100, 100).astype(np.float32)
    assert tiles(img, 224, 224).shape[0] == 1


def test_view_of_parses_embedding_and_qa_columns():
    assert ev.view_of("emb__BSE__dinov2_vits14__0") == "BSE"
    assert ev.view_of("qa__InLens__noise_sigma__8bit") == "InLens"
    assert ev.view_of("kpi__SE__dark_fraction__frac") == "SE"


def test_cosine_scorer_separates_direction_not_magnitude():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(12):
        b = "1" if i < 6 else "2"
        direction = np.array([1.0, 0.0, 0.0]) if b == "1" else np.array([0.0, 1.0, 0.0])
        scale = rng.uniform(0.7, 1.5)  # magnitude is uninformative
        vec = scale * direction + rng.normal(0, 0.05, 3)
        rows.append({"sample_id": f"s{i}", "batch_id": b, **{f"emb__BSE__m__{k}": v for k, v in enumerate(vec)}})
    df = pd.DataFrame(rows)
    cols = ev.feature_columns(df, "emb__")
    res = ev.leave_one_sample_out(df, cols, ev.CosineCentroidScorer)
    assert res.accuracy() == 1.0
    assert all(p["score_batch_1"] >= 0 for _, p in res.predictions.iterrows())
