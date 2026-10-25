"""Confusion-matrix metrics over the K fixed classes and the acceptance gate.

Ported from the completed runs, ``metrics`` and the gate in ``validation``:
https://github.com/hongphuc-maker/xuan-thuy-prithvi-segmentation/blob/42290823beeb42e44ec87bb8845e439e10c18735/reference/wce_completed/prithvi_xtseg/inference.py#L14-L28
https://github.com/hongphuc-maker/xuan-thuy-prithvi-segmentation/blob/42290823beeb42e44ec87bb8845e439e10c18735/reference/wce_completed/prithvi_xtseg/inference.py#L73-L77
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def confusion_matrix(truth: np.ndarray, pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Rows = true class, columns = predicted class (both 0-based)."""
    return np.bincount(truth.astype(np.int64) * num_classes + pred, minlength=num_classes**2).reshape(num_classes, num_classes)


def per_class_metrics(confusion: np.ndarray, class_map: dict[str, str]) -> tuple[dict[str, Any], pd.DataFrame]:
    k = confusion.shape[0]
    tp = np.diag(confusion).astype(float)
    support, predicted = confusion.sum(1).astype(float), confusion.sum(0).astype(float)
    precision = np.divide(tp, predicted, out=np.zeros(k), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(k), where=support > 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(k), where=support + predicted > 0)
    union = support + predicted - tp
    iou = np.divide(tp, union, out=np.zeros(k), where=union > 0)
    table = pd.DataFrame({"code": range(1, k + 1), "class": [class_map[str(i)] for i in range(1, k + 1)],
                          "support": support.astype(int), "predicted": predicted.astype(int),
                          "precision": precision, "recall": recall, "F1": f1, "IoU": iou})
    summary = {"OA": float(tp.sum() / max(support.sum(), 1)),
               "macro_F1_11": float(f1.mean()),  # mean over all 11 classes, absent ones count as 0
               "macro_IoU_11": float(iou.mean()),
               "predicted_classes": int((predicted > 0).sum()),
               "n_pixels": int(support.sum())}
    return summary, table


def acceptance_gate(summary: dict[str, Any], table: pd.DataFrame, gate: dict[str, Any]) -> bool:
    """All classes predicted, and recall >= threshold for each listed rare class."""
    ok = summary["predicted_classes"] == gate["required_predicted_classes"]
    for code, minimum in gate["minimum_recall_by_code"].items():
        ok &= float(table.loc[table.code == int(code), "recall"].iloc[0]) >= minimum
    return bool(ok)
