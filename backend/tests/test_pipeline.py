import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from app.core.pipeline import ClassificationPipeline, ClusteringPipeline
from app.core.registry import CLASSIFIERS, CLUSTERERS, FEATURE_EXTRACTORS


def test_registries_populated():
    assert {"pca", "lda", "standard_scaler"} <= {m["name"] for m in FEATURE_EXTRACTORS.describe()}
    assert len(CLASSIFIERS.describe()) == 6
    assert len(CLUSTERERS.describe()) == 4


def test_every_classifier_runs():
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)
    for c in CLASSIFIERS.describe():
        p = ClassificationPipeline([{"name": "standard_scaler"}, {"name": "pca", "params": {"n_components": 2}}],
                                   {"name": c["name"]}).fit(Xtr, ytr)
        assert (p.predict(Xte) == yte).mean() > 0.7, c["name"]


def test_every_clusterer_runs():
    X, _ = load_iris(return_X_y=True)
    for c in CLUSTERERS.describe():
        labels = ClusteringPipeline([{"name": "standard_scaler"}], {"name": c["name"]}).fit_predict(X)
        assert len(labels) == len(X), c["name"]


def test_unknown_module_message():
    try:
        CLASSIFIERS.get("nope")
    except KeyError as e:
        assert "Available" in str(e)
