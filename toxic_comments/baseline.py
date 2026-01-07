from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from toxic_comments.metrics import multilabel_metrics


def train_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    tfidf_cfg: Dict,
    logreg_cfg: Dict,
) -> Tuple[Pipeline, Dict[str, float]]:
    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=int(tfidf_cfg["max_features"]),
                    ngram_range=tuple(tfidf_cfg["ngram_range"]),
                    min_df=int(tfidf_cfg["min_df"]),
                ),
            ),
            (
                "clf",
                OneVsRestClassifier(
                    LogisticRegression(
                        C=float(logreg_cfg["C"]),
                        max_iter=int(logreg_cfg["max_iter"]),
                        solver="liblinear",
                    )
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)
    y_prob = model.predict_proba(x_val)
    metrics = multilabel_metrics(y_val, y_prob)
    return model, metrics


def save_baseline(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
