import numpy as np
import pandas as pd

from augment_train import OUT_RES, TILE, augment, sample_level_loo, tile_grid, to_224


def test_tile_grid_counts_full_tiles_only():
    img = np.zeros((2 * TILE + 10, 3 * TILE - 1))
    assert len(tile_grid(img)) == 2 * 2


def test_augment_keeps_range_and_is_square():
    rng = np.random.default_rng(0)
    t = rng.random((TILE, TILE))
    for photometric in (False, True):
        a = augment(t, rng, photometric)
        assert a.shape[0] == a.shape[1]
        assert 0.6 * TILE - 1 <= a.shape[0] <= TILE
        assert a.min() >= 0 and a.max() <= 1
    assert to_224(a).shape == (OUT_RES, OUT_RES)


def test_geometric_augment_preserves_intensity_distribution():
    rng = np.random.default_rng(1)
    t = rng.random((TILE, TILE))
    a = augment(t, rng, photometric=False)
    assert abs(a.mean() - t.mean()) < 0.05


def test_sample_level_loo_never_trains_on_held_out_tiles():
    rng = np.random.default_rng(0)
    samples = [f"s{i}" for i in range(6)]
    labels = pd.Series(["1", "1", "1", "2", "2", "2"], index=samples)
    rows, X = [], []
    for s in samples:
        shift = 3.0 if labels[s] == "2" else 0.0
        for tile in range(4):
            for aug in range(3):
                rows.append(dict(sample_id=s, batch_id=labels[s], view="BSE", tile=tile, aug=aug))
                X.append(rng.normal(shift, 1.0, 5))
    meta = pd.DataFrame(rows)
    df = sample_level_loo(meta, np.array(X), labels, "BSE", train_aug=True, C=1.0)
    assert len(df) == 6
    assert set(df.columns) >= {"sample_id", "true", "pred", "p_1", "p_2"}
    assert (df.pred == df.true).mean() == 1.0
