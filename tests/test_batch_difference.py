import numpy as np
import pandas as pd

from batch_difference_test import energy_distance, perm_pvalue, robust_scale
from kpi_reliability import reliability_table


def test_energy_distance_near_zero_for_same_distribution_and_positive_for_shifted():
    rng = np.random.default_rng(0)
    A = rng.normal(0, 1, (200, 3))
    B = rng.normal(0, 1, (200, 3))
    assert abs(energy_distance(A, B)) < 0.1  # U-statistic: unbiased, so can be slightly negative
    assert energy_distance(A, A + 3) > 1


def test_perm_pvalue_enumerates_all_splits_and_detects_shift():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(0, 1, (7, 2)), rng.normal(4, 1, (7, 2))])
    stat, p, count = perm_pvalue(robust_scale(X), 7, max_perms=10000, seed=0)
    assert count == 3432
    assert p < 0.01


def test_perm_pvalue_is_large_for_same_distribution():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (14, 2))
    _, p, _ = perm_pvalue(robust_scale(X), 7, max_perms=10000, seed=0)
    assert p > 0.05


def test_reliability_table_ranks_consistent_kpi_above_noise():
    rng = np.random.default_rng(3)
    n = 20
    true = rng.normal(10, 2, n)
    rows = []
    for i in range(n):
        for half in ("left", "right"):
            rows.append({"sample_id": f"s{i}", "batch_id": "1", "half": half,
                         "kpi__BSE__stable__px": true[i] + rng.normal(0, 0.1),
                         "kpi__BSE__noise__px": rng.normal(0, 1)})
    table = reliability_table(pd.DataFrame(rows)).set_index("kpi")
    assert table.loc["kpi__BSE__stable__px", "reliability"] > 0.9
    assert table.loc["kpi__BSE__noise__px", "reliability"] < 0.3
