from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from toxic_comments import LABELS


def multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}

    # ROC-AUC per class
    per_class = []
    for idx, name in enumerate(LABELS):
        try:
            auc = roc_auc_score(y_true[:, idx], y_prob[:, idx])
        except ValueError:
            auc = float("nan")
        out[f"val_auc_{name}"] = auc
        if not np.isnan(auc):
            per_class.append(auc)

    out["val_auc_mean"] = float(np.mean(per_class)) if per_class else float("nan")

    # F1 with threshold 0.5
    y_pred = (y_prob >= 0.5).astype(int)
    out["val_f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    out["val_f1_micro"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))

    return out
