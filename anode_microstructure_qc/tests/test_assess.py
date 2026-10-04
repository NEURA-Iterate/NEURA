import numpy as np
import pandas as pd
import pytest

pytest.importorskip("neura_uq")

from anode_qc import assess as qa  # noqa: E402

CENTRES = {"Batch_1": (0.08, 42.0, 2.0), "Batch_2": (0.10, 50.0, 1.75), "Batch_3": (0.15, 68.0, 1.95)}


def synthetic_reference(n=(7, 7, 17), seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (b, (pore, crack, cv)), k in zip(CENTRES.items(), n, strict=True):
        for i in range(k):
            rows.append(
                {
                    "image_id": f"{b}_{i}",
                    "batch": b,
                    "frac_pore": pore * np.exp(rng.normal(0, 0.08)),
                    "graphite_crack_density": crack * np.exp(rng.normal(0, 0.06)),
                    "si_cv_w256": cv * np.exp(rng.normal(0, 0.03)),
                    "frac_pore_sd": 0.005,
                    "graphite_crack_density_sd": 1.0,
                    "si_cv_w256_sd": 0.02,
                    "si_fraction_of_solids": 0.06,
                    "trust_flags": "",
                }
            )
    return pd.DataFrame(rows)


def sample(values, flags=""):
    runs = np.array([values, [v * 1.01 for v in values], [v * 0.99 for v in values]])
    row = {"image_id": "s", "batch": "unknown", "trust_flags": flags, "si_fraction_of_solids": 0.06}
    row.update(dict(zip(qa.DECISION_NAMES, values, strict=True)))
    return {"row": row, "runs": runs, "thumbnail": None}


@pytest.fixture
def ref_path(tmp_path):
    p = tmp_path / "ref.csv"
    synthetic_reference().to_csv(p, index=False)
    qa._model.cache_clear()
    return p


def test_baseline_like_sample_is_accepted(ref_path):
    a = qa.assess(sample(CENTRES["Batch_3"]), ref_path)
    assert a.batch == "Batch_3"
    assert a.verdict == "ACCEPT"
    assert all(s.status == "within baseline" for s in a.shifts)
    assert a.segmentation_stable


def test_drifted_sample_is_rejected_with_direction(ref_path):
    a = qa.assess(sample(CENTRES["Batch_1"]), ref_path)
    assert a.batch == "Batch_1"
    assert a.verdict == "REJECT"
    pore = next(s for s in a.shifts if s.kpi == "frac_pore")
    assert pore.z < -2 and "denser" in pore.meaning
    assert a.probabilities["Batch_1"] > 0.8


def test_flags_downgrade_accept_and_confidence(ref_path):
    a = qa.assess(sample(CENTRES["Batch_3"], flags="charging"), ref_path)
    assert a.verdict == "INVESTIGATE"
    assert a.confidence == "review"


def test_far_sample_is_out_of_distribution_but_still_assigned(ref_path):
    a = qa.assess(sample((0.40, 300.0, 4.0)), ref_path)
    assert a.out_of_distribution
    assert a.batch in qa.CLASSES
    assert a.verdict == "REJECT"


def test_report_and_validation(ref_path, tmp_path):
    a = qa.assess(sample(CENTRES["Batch_2"]), ref_path)
    page = qa.render_html(a)
    assert a.verdict in page and "Decision KPIs" in page and "NO GROUND TRUTH" not in page
    v = a.validation
    assert v["n_images"] == 31 and 0.0 <= v["balanced_accuracy"] <= 1.0
    assert sum(t["n"] for t in v["tiers"].values()) == 31
    out = qa.write_outputs(a, tmp_path / "out")
    assert out.exists() and (tmp_path / "out" / "s.json").exists()


def test_bundled_reference_is_valid():
    ref = qa.load_reference()
    assert set(ref.batch) == set(qa.CLASSES)
    assert ref[qa.DECISION_NAMES].notna().all().all()


def test_value_beyond_baseline_percentiles_is_not_called_typical(ref_path):
    ref = qa.load_reference(ref_path)
    b3 = ref[ref.batch == "Batch_3"]
    cv = float(np.percentile(b3.si_cv_w256, 95)) * 1.01
    a = qa.assess(sample((CENTRES["Batch_3"][0], CENTRES["Batch_3"][1], cv)), ref_path)
    s = next(s for s in a.shifts if s.kpi == "si_cv_w256")
    assert s.status != "within baseline"
    assert "typical" not in s.meaning
