"""Benchmark many pipelines side by side."""
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..core.data import Dataset, split_dataset
from ..core.pipeline import ClassificationPipeline, ClusteringPipeline
from ..core.registry import CLASSIFIERS
from .cross_val import cross_validate_pipeline
from .metrics import classification_metrics, clustering_metrics


def configs_for_all_classifiers(steps: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """One config per registered classifier, all sharing the same preprocessing steps."""
    steps = steps or [{"name": "standard_scaler"}]
    return [{"label": c["name"], "steps": steps, "classifier": {"name": c["name"]}} for c in CLASSIFIERS.describe()]


def compare_classifiers(
    ds: Dataset,
    configs: List[Dict[str, Any]],
    test_size: float = 0.25,
    cv_splits: int = 5,
    random_state: int = 42,
):
    """Cross-validate on the training split, then score once on the untouched test split.

    config = {"label": str (optional), "steps": [...], "classifier": {"name": ..., "params": ...}}
    Returns (summary DataFrame, details dict keyed by label).
    """
    Xtr, Xte, ytr, yte = split_dataset(ds, test_size=test_size, random_state=random_state)
    rows, details = [], {}

    for i, cfg in enumerate(configs):
        steps, clf = cfg.get("steps", []), cfg["classifier"]
        label = cfg.get("label") or "+".join([s["name"] for s in steps] + [clf["name"]]) or f"config_{i}"

        t0 = time.perf_counter()
        pipe = ClassificationPipeline(steps, clf).fit(Xtr, ytr)
        fit_seconds = time.perf_counter() - t0

        y_pred = pipe.predict(Xte)
        y_proba = pipe.predict_proba(Xte)
        m = classification_metrics(yte, y_pred, y_proba, ds.class_names)
        cv = cross_validate_pipeline(steps, clf, Xtr, ytr, n_splits=cv_splits, random_state=random_state)["summary"]

        rows.append({
            "label": label,
            "cv_accuracy_mean": cv["accuracy_mean"],
            "cv_f1_macro_mean": cv["f1_macro_mean"],
            "cv_f1_macro_std": cv["f1_macro_std"],
            "test_accuracy": m["accuracy"],
            "test_f1_macro": m["f1_macro"],
            "test_roc_auc": m["roc_auc"],
            "fit_seconds": round(fit_seconds, 4),
        })
        details[label] = {"pipeline": pipe, "y_true": yte, "y_pred": y_pred, "y_proba": y_proba, "metrics": m}

    summary = pd.DataFrame(rows).sort_values("cv_f1_macro_mean", ascending=False).reset_index(drop=True)
    return summary, details


def compare_clusterers(ds: Dataset, configs: List[Dict[str, Any]]):
    """config = {"label": str (optional), "steps": [...], "clusterer": {"name": ..., "params": ...}}
    Clustering is unsupervised: labels are never given to the pipeline, only used afterwards for
    external metrics (ARI, NMI) if the dataset has them."""
    rows, details = [], {}
    for i, cfg in enumerate(configs):
        steps, clu = cfg.get("steps", []), cfg["clusterer"]
        label = cfg.get("label") or "+".join([s["name"] for s in steps] + [clu["name"]]) or f"config_{i}"

        t0 = time.perf_counter()
        pipe = ClusteringPipeline(steps, clu)
        labels = pipe.fit_predict(ds.X)
        seconds = time.perf_counter() - t0

        m = clustering_metrics(pipe.transformed_, labels, ds.y)
        rows.append({"label": label, **m, "seconds": round(seconds, 4)})
        details[label] = {"labels": labels, "features": pipe.transformed_, "metrics": m}

    return pd.DataFrame(rows), details


def sweep_n_clusters(
    X: np.ndarray,
    clusterer: str = "kmeans",
    k_range=range(2, 9),
    steps: Optional[List[Dict[str, Any]]] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Try several cluster counts (elbow / silhouette analysis)."""
    param = "n_components" if clusterer == "gmm" else "n_clusters"
    rows = []
    for k in k_range:
        pipe = ClusteringPipeline(steps or [], {"name": clusterer, "params": {param: k}})
        labels = pipe.fit_predict(X)
        m = clustering_metrics(pipe.transformed_, labels)
        inertia = getattr(pipe.clusterer.model, "inertia_", None)
        rows.append({
            "k": k,
            "inertia": None if inertia is None else float(inertia),
            "silhouette": m["silhouette"],
            "davies_bouldin": m["davies_bouldin"],
            "calinski_harabasz": m["calinski_harabasz"],
        })
    return pd.DataFrame(rows)
