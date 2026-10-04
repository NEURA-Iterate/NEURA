import pytest

from neura_uq import Prediction, confidence_tier


def make_prediction(probability: float, typicality: float) -> Prediction:
    return Prediction(
        classes=["Batch_1", "Batch_2"],
        kpis=["kpi"],
        probabilities={"Batch_1": probability, "Batch_2": 1 - probability},
        log_likelihood={},
        typicality_pvalue={"Batch_1": typicality, "Batch_2": 0.5},
        kpi_log_likelihood={},
        predicted="Batch_1",
        runner_up="Batch_2",
        outlier=False,
        ambiguous=False,
    )


@pytest.mark.parametrize(
    ("probability", "typicality", "expected"),
    [
        (0.95, 0.10, "high"),
        (0.75, 0.10, "medium"),
        (0.59, 0.10, "review"),
        (0.95, 0.049, "review"),
        (0.90, 0.10, "high"),
        (0.60, 0.10, "medium"),
    ],
)
def test_confidence_tier(probability, typicality, expected):
    assert confidence_tier(make_prediction(probability, typicality)) == expected
