import json

import matplotlib
matplotlib.use("Agg")
import numpy as np

from app.core.data import load_builtin
from app.evaluation import plots
from app.evaluation.compare import (compare_classifiers, compare_clusterers,
                                    configs_for_all_classifiers, sweep_n_clusters)
from app.evaluation.cross_val import cross_validate_pipeline
from app.evaluation.metrics import classification_metrics, clustering_metrics


def test_classification_metrics_perfect_and_json():
    y = np.array([0, 1, 2, 0, 1, 2])
    m = classification_metrics(y, y, class_names=["a", "b", "c"])
    assert m["accuracy"] == 1.0 and m["f1_macro"] == 1.0
    assert m["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    json.dumps(m)


def test_cross_validation_no_crash_and_shape():
    ds = load_builtin("iris")
    out = cross_validate_pipeline([{"name": "standard_scaler"}, {"name": "select_k_best", "params": {"k": 2}}],
                                  {"name": "knn"}, ds.X, ds.y, n_splits=5)
    assert len(out["folds"]) == 5
    assert out["summary"]["accuracy_mean"] > 0.8


def test_compare_classifiers_wine():
    ds = load_builtin("wine")
    summary, details = compare_classifiers(ds, configs_for_all_classifiers())
    assert len(summary) == 6
    assert summary["test_accuracy"].max() > 0.9
    assert summary["cv_f1_macro_mean"].is_monotonic_decreasing
    best = summary.loc[0, "label"]
    m = details[best]["metrics"]
    plots.plot_confusion_matrix(m["confusion_matrix"], m["class_names"])
    plots.plot_roc_curves(details[best]["y_true"], details[best]["y_proba"], ds.class_names)
    plots.plot_model_comparison(summary)


def test_binary_dataset_roc():
    ds = load_builtin("breast_cancer")
    summary, details = compare_classifiers(ds, [{"classifier": {"name": "logistic_regression"},
                                                  "steps": [{"name": "standard_scaler"}]}])
    assert summary.loc[0, "test_roc_auc"] > 0.9
    d = details[summary.loc[0, "label"]]
    plots.plot_roc_curves(d["y_true"], d["y_proba"], ds.class_names)


def test_compare_clusterers_handles_noise():
    ds = load_builtin("iris")
    steps = [{"name": "standard_scaler"}]
    cfgs = [{"steps": steps, "clusterer": {"name": n}} for n in ["kmeans", "dbscan", "hierarchical", "gmm"]]
    summary, details = compare_clusterers(ds, cfgs)
    assert len(summary) == 4
    for d in details.values():
        plots.plot_clusters(d["features"], d["labels"])


def test_clustering_metrics_single_cluster_is_safe():
    X = np.random.RandomState(0).randn(30, 2)
    m = clustering_metrics(X, np.zeros(30, dtype=int))
    assert m["silhouette"] is None


def test_k_sweep():
    ds = load_builtin("iris")
    sweep = sweep_n_clusters(ds.X, "kmeans", range(2, 6), [{"name": "standard_scaler"}])
    assert list(sweep["k"]) == [2, 3, 4, 5]
    plots.plot_k_sweep(sweep)
    sweep_g = sweep_n_clusters(ds.X, "gmm", range(2, 4))
    assert sweep_g["inertia"].isna().all()
    plots.plot_k_sweep(sweep_g)
