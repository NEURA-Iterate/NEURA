from types import SimpleNamespace

import numpy as np
import torch

from anode_qc.dinoseg import (
    IGNORE,
    Specimen,
    draft_targets,
    patch_features,
    predict_proba,
    to_labels,
    train_decoder,
)
from anode_qc.segment import CBD, GAP, GRAPHITE, PORE, SI


class FakeBackbone(torch.nn.Module):
    """Mimics a transformers DINO model: CLS + register tokens followed by patch tokens (mean grey per patch)."""

    def __init__(self, patch=4, registers=2, hidden=3):
        super().__init__()
        self.config = SimpleNamespace(patch_size=patch, num_register_tokens=registers, hidden_size=hidden)

    def forward(self, pixel_values):
        p, c = self.config.patch_size, self.config.hidden_size
        pooled = torch.nn.functional.avg_pool2d(pixel_values[:, :1], p).flatten(2).transpose(1, 2)
        extra = torch.zeros(pixel_values.shape[0], 1 + self.config.num_register_tokens, 1)
        return SimpleNamespace(last_hidden_state=torch.cat([extra, pooled], 1).repeat(1, 1, c))


def _field(h=60, w=90):
    labels = np.full((h, w), GRAPHITE, np.uint8)
    labels[:, : w // 3] = PORE
    labels[:, 2 * w // 3 :] = SI
    grey = np.choose(labels // 2, [40, 120]).astype(np.uint8)
    grey[labels == PORE] = 5
    return labels, grey


def test_patch_features_grid_and_token_offset():
    _, grey = _field()
    f = patch_features(FakeBackbone(), grey, tile=32, batch=2)
    assert f.shape == (15, 22, 3)
    assert f[:, 0, 0].max() < f[:, -1, 0].min()


def test_draft_targets_ignores_edges_and_gaps():
    labels, _ = _field()
    labels[10:12, 40:50] = GAP
    t = draft_targets(labels, erode_px=2)
    assert (t[10:12, 40:50] == IGNORE).all()
    assert t[30, 29] == IGNORE and t[30, 5] == PORE and t[30, 80] == SI
    assert not np.isin(t, [GAP, CBD]).any()


def test_decoder_learns_toy_field_and_keeps_gaps():
    labels, grey = _field()
    model = FakeBackbone()
    feats = [patch_features(model, grey, tile=32)] * 3
    spec = Specimen.build("a", "Batch_1", feats, [grey] * 3, 4, draft_targets(labels))
    dec = train_decoder([spec], iters=80, crop=24, batch=4, lr=1e-2)
    proba = predict_proba(dec, spec, tile=32)
    assert proba.shape == (4, *spec.shape)
    np.testing.assert_allclose(proba.sum(0), 1, atol=1e-5)
    gap = np.zeros(spec.shape, bool)
    gap[5, 35:40] = True
    lab = to_labels(proba, gap)
    known = spec.targets != IGNORE
    assert (lab[known & ~gap] == spec.targets[known & ~gap]).mean() > 0.95
    assert (lab[gap] == GAP).all()
