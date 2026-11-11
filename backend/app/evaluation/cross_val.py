"""Stratified k-fold cross-validation. The whole pipeline is rebuilt and refit inside every fold,
so feature selection / scaling / PCA never see the validation data (no leakage)."""
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from ..core.pipeline import ClassificationPipeline


def cross_validate_pipeline(
    steps: List[Dict[str, Any]],
    classifier: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict:
    n_splits = max(2, min(n_splits, int(np.bincount(y).min())))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rows = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        pipe = ClassificationPipeline(steps, classifier).fit(X[train_idx], y[train_idx])
        pred = pipe.predict(X[val_idx])
        rows.append({
            "fold": fold,
            "accuracy": accuracy_score(y[val_idx], pred),
            "f1_macro": f1_score(y[val_idx], pred, average="macro", zero_division=0),
        })

    folds = pd.DataFrame(rows)
    summary = {
        "n_splits": n_splits,
        "accuracy_mean": float(folds["accuracy"].mean()),
        "accuracy_std": float(folds["accuracy"].std(ddof=0)),
        "f1_macro_mean": float(folds["f1_macro"].mean()),
        "f1_macro_std": float(folds["f1_macro"].std(ddof=0)),
    }
    return {"folds": folds, "summary": summary}
