import numpy as np
import torch
import torch.nn.functional as F

from prithvi_unet.inference.metrics import confusion_matrix, per_class_metrics
from prithvi_unet.training.loss import class_weights_from_counts, weighted_cross_entropy


def test_class_weights_rule():
    w = class_weights_from_counts([1_000_000, 10_000, 100, 50_000])
    assert abs(float(w.mean()) - 1.0) < 1e-6
    assert float(w.max() / w.min()) <= 10.0 + 1e-5  # clip [0.5, 5]
    assert int(torch.argmax(w)) == 2  # rarest class weighs most


def test_wce_matches_pytorch_weighted_cross_entropy():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(2, 11, 16, 16, generator=g)
    target = torch.randint(0, 11, (2, 16, 16), generator=g)
    target[0, :4] = 255
    weights = torch.rand(11, generator=g) + 0.5
    ours = weighted_cross_entropy(logits, target, weights)
    ref = F.cross_entropy(logits, target, weight=weights, ignore_index=255)
    assert torch.allclose(ours, ref, atol=1e-6)


def test_macro_f1_counts_absent_predictions_as_zero():
    truth = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 0, 1, 1, 1, 1])
    summary, table = per_class_metrics(confusion_matrix(truth, pred, 3), {"1": "a", "2": "b", "3": "c"})
    assert summary["predicted_classes"] == 2
    assert table.F1.iloc[2] == 0.0
    assert abs(summary["macro_F1_11"] - (1.0 + 2 / 3 + 0.0) / 3) < 1e-9
