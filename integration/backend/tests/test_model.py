from __future__ import annotations

import numpy as np
import pandas as pd
from neura_app.model import BASE, BATCHES, DualSourceClassifier


def _tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(17)
    rows = {"rule": [], "learned": []}
    for batch_index, batch in enumerate(BATCHES):
        for sample_index in range(8):
            values = {
                "frac_pore": 0.08 + batch_index * 0.03 + rng.normal(0, 0.006),
                "graphite_crack_density": 30 + batch_index * 14 + rng.normal(0, 2),
                "graphite_aspect_ratio_median": 1.4 + batch_index * 0.1 + rng.normal(0, 0.03),
                "si_cv_w256": 1.7 + batch_index * 0.1 + rng.normal(0, 0.025),
                "si_contact_graphite": 0.5 + batch_index * 0.05 + rng.normal(0, 0.02),
                "si_fraction_of_solids": 0.08 + batch_index * 0.01 + rng.normal(0, 0.005),
                "si_ecd_d50": 20 + batch_index * 4 + rng.normal(0, 1),
                "si_ecd_d90": 60 + batch_index * 8 + rng.normal(0, 3),
                "si_crack_density": 5 + batch_index * 2 + rng.normal(0, 0.5),
                "si_dispersion_index_w512": 1 + batch_index * 0.05 + rng.normal(0, 0.02),
                "cbd_fraction_of_solids": 0.1 + batch_index * 0.01 + rng.normal(0, 0.005),
                "graphite_alignment": 0.1 + batch_index * 0.03 + rng.normal(0, 0.01),
            }
            for source, source_rows in rows.items():
                source_rows.append(
                    {
                        "image_id": f"{source}-{batch_index}-{sample_index}",
                        "batch": batch,
                        **values,
                    }
                )
    return pd.DataFrame(rows["rule"]), pd.DataFrame(rows["learned"])


def _fitted(replicates: int = 24) -> tuple[DualSourceClassifier, pd.DataFrame, pd.DataFrame]:
    rule, learned = _tables()
    return DualSourceClassifier(rule, learned, bootstrap_replicates=replicates), rule, learned


def test_geometric_combination_and_contributions_reproduce_log_odds() -> None:
    model, rule, learned = _fitted()
    rule_row = rule.iloc[1].to_dict()
    learned_row = learned.iloc[1].to_dict()
    result = model.classify(rule_row, learned_row)
    probability = result["prediction"]["probabilities"]
    assert np.isclose(sum(probability.values()), 1)
    assert probability == {
        batch: np.sqrt(
            result["classifiers"]["rule"]["probabilities"][batch]
            * result["classifiers"]["learned"]["probabilities"][batch]
        )
        / sum(
            np.sqrt(
                result["classifiers"]["rule"]["probabilities"][other]
                * result["classifiers"]["learned"]["probabilities"][other]
            )
            for other in BATCHES
        )
        for batch in BATCHES
    }
    for first_index, first in enumerate(BATCHES):
        for second in BATCHES[first_index + 1 :]:
            log_odds = np.log(probability[first] / probability[second])
            contribution_odds = sum(
                item["per_batch"][first] - item["per_batch"][second]
                for item in result["contributions"]
            )
            assert abs(log_odds - contribution_odds) < 1e-6


def test_missing_kpi_is_refit_without_that_feature() -> None:
    model, rule, learned = _fitted()
    rule_row = rule.iloc[0].to_dict()
    rule_row["graphite_crack_density"] = np.nan
    result = model.classify(rule_row, learned.iloc[0].to_dict())
    assert "graphite_crack_density" not in result["classifiers"]["rule"]["evidence"]
    assert any("Missing KPI graphite_crack_density" in warning for warning in result["warnings"])


def test_bootstrap_intervals_cover_training_point_estimate() -> None:
    model, rule, learned = _fitted(replicates=200)
    result = model.classify(rule.iloc[2].to_dict(), learned.iloc[2].to_dict())
    for source in ("rule", "learned"):
        for batch, point in result["classifiers"][source]["probabilities"].items():
            low, high = result["classifiers"][source]["interval"][batch]
            assert low <= point <= high
    for batch, point in result["prediction"]["probabilities"].items():
        low, high = result["prediction"]["interval"][batch]
        assert low <= point <= high


def test_baseline_z_sign_uses_transformed_batch3_reference() -> None:
    model, rule, learned = _fitted()
    rule_row = rule.loc[rule["batch"] == "Batch_1"].iloc[0].to_dict()
    learned_row = learned.loc[learned["batch"] == "Batch_1"].iloc[0].to_dict()
    result = model.classify(rule_row, learned_row)
    item = next(
        entry
        for entry in result["baseline"]["kpis"]
        if entry["source"] == "rule" and entry["kpi"] == "frac_pore"
    )
    assert item["z"] < 0
    assert item["direction"] == "lower"
    assert set(BASE) == {entry["kpi"] for entry in result["baseline"]["kpis"] if entry["source"] == "rule"}
