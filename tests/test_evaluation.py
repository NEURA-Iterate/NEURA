import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluation as ev  # noqa: E402
from make_fake_features import make_features  # noqa: E402


def separable_df(n=(5, 5, 8), seed=0):
    return make_features({"1": n[0], "2": n[1], "3": n[2]}, n_unlabelled=3, noise=0.2, seed=seed)


def test_feature_columns_and_view_parsing():
    df = separable_df()
    cols = ev.feature_columns(df)
    assert "sample_id" not in cols and "batch_id" not in cols
    assert ev.view_of("kpi__BSE__porosity__frac") == "BSE"
    assert ev.view_of("kpi__InLens__edge_network_fraction__frac") == "InLens"
    assert ev.view_of("embedding_0") == "default"
    assert ev.feature_columns(df, prefix="kpi__SE__") == ["kpi__SE__pore_d50__px", "kpi__SE__thickness__px"]


def test_view_weights_split_evenly_within_view():
    cols = ["kpi__BSE__a__u", "kpi__BSE__b__u", "kpi__SE__c__u"]
    w = ev.column_weights(cols, {"SE": 2.0})
    assert w.tolist() == pytest.approx([0.5, 0.5, 2.0])


def test_unlabelled_rows_are_ignored_and_duplicates_rejected():
    df = separable_df()
    lab = ev.labelled_rows(df)
    assert len(lab) == 18
    dup = pd.concat([df, df.iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate"):
        ev.labelled_rows(dup)


@pytest.mark.parametrize("scorer", [ev.ProfileDistanceScorer, ev.GaussianScorer])
def test_loo_perfect_on_separable_data(scorer):
    df = separable_df()
    res = ev.leave_one_sample_out(df, ev.feature_columns(df), scorer)
    assert res.accuracy() == 1.0
    assert res.predictions["evaluable"].all()
    assert len(res.mistakes()) == 0
    cm = res.confusion_matrix()
    assert list(cm.index) == ["1", "2", "3"] and np.trace(cm.to_numpy()) == 18
    # held-out sample never contributes to its own profile: scores exist for every batch
    assert res.predictions[[f"score_batch_{b}" for b in "123"]].notna().all().all()


def test_small_batch_is_reported_not_silently_evaluated():
    df = separable_df(n=(1, 5, 6))
    res = ev.leave_one_sample_out(df, ev.feature_columns(df), ev.ProfileDistanceScorer, min_train_per_batch=2)
    assert any("batch 1" in w for w in res.warnings)
    row = res.predictions[res.predictions["true_batch"] == "1"].iloc[0]
    assert not row["evaluable"]
    assert res.predictions["evaluable"].sum() == 11
    assert res.accuracy() == 1.0


def test_nothing_evaluable_when_only_one_batch_has_enough_samples():
    df = separable_df(n=(1, 1, 6))
    res = ev.leave_one_sample_out(df, ev.feature_columns(df), ev.ProfileDistanceScorer)
    assert res.accuracy() is None
    assert len(res.mistakes()) == 0


def test_contributions_favour_winner_and_sum_to_score_gap():
    df = separable_df()
    cols = ev.feature_columns(df)
    lab = ev.labelled_rows(df)
    scorer = ev.ProfileDistanceScorer().fit(lab[cols], lab["batch_id"])
    x = lab.iloc[0]
    s = scorer.scores(x)
    ranked = sorted(s, key=s.get)
    c = scorer.contributions(x, ranked[0], ranked[1])
    assert c.sum() == pytest.approx(s[ranked[1]] ** 2 - s[ranked[0]] ** 2)
    assert c.iloc[0] >= c.iloc[-1]


def test_shuffle_baseline_is_near_chance():
    df = separable_df(n=(6, 6, 6))
    base = ev.shuffle_baseline(df, ev.feature_columns(df), ev.ProfileDistanceScorer, n_shuffles=15, seed=1)
    assert base["n_shuffles"] == 15
    assert base["mean_accuracy"] < 0.6


def test_cli_writes_all_outputs(tmp_path):
    csv = tmp_path / "f.csv"
    separable_df().to_csv(csv, index=False)
    out = tmp_path / "out"
    ev.main(["--features", str(csv), "--out", str(out), "--shuffles", "3"])
    for name in ("predictions.csv", "confusion_matrix.csv", "mistakes.csv", "summary.json"):
        assert (out / name).exists()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["accuracy"] == 1.0
    assert summary["n_unlabelled_rows_ignored"] == 3
    assert summary["label_shuffle_baseline"]["n_shuffles"] == 3
    pred = pd.read_csv(out / "predictions.csv")
    assert set(["sample_id", "true_batch", "predicted_batch", "margin", "margin_rel", "review_flag", "top_contributors"]) <= set(pred.columns)


def test_embedding_csv_without_kpi_prefix_works():
    rng = np.random.default_rng(0)
    rows = []
    for b, centre in zip("123", ([0, 0], [5, 0], [0, 5])):
        for i in range(5):
            e = np.array(centre) + rng.normal(0, 0.3, 2)
            rows.append({"sample_id": f"s{b}{i}", "batch_id": b, "emb_0": e[0], "emb_1": e[1]})
    df = pd.DataFrame(rows)
    res = ev.leave_one_sample_out(df, ev.feature_columns(df), ev.ProfileDistanceScorer)
    assert res.accuracy() == 1.0
