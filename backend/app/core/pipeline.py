"""Pipeline: an ordered list of feature extractors followed by one model."""
from typing import Any, Dict, List

from .. import modules  # noqa: F401  (registers built-ins)
from .registry import CLASSIFIERS, CLUSTERERS, FEATURE_EXTRACTORS


class ClassificationPipeline:
    def __init__(self, steps: List[Dict[str, Any]], classifier: Dict[str, Any]):
        self.extractors = [FEATURE_EXTRACTORS.create(s["name"], **s.get("params", {})) for s in steps]
        self.classifier = CLASSIFIERS.create(classifier["name"], **classifier.get("params", {}))

    def _transform(self, X):
        for ex in self.extractors:
            X = ex.transform(X)
        return X

    def fit(self, X, y):
        for ex in self.extractors:
            X = ex.fit_transform(X, y)
        self.classifier.fit(X, y)
        return self

    def predict(self, X):
        return self.classifier.predict(self._transform(X))

    def predict_proba(self, X):
        return self.classifier.predict_proba(self._transform(X))


class ClusteringPipeline:
    def __init__(self, steps: List[Dict[str, Any]], clusterer: Dict[str, Any]):
        self.extractors = [FEATURE_EXTRACTORS.create(s["name"], **s.get("params", {})) for s in steps]
        self.clusterer = CLUSTERERS.create(clusterer["name"], **clusterer.get("params", {}))

    def fit_predict(self, X, y=None):
        for ex in self.extractors:
            X = ex.fit_transform(X, y)
        self.transformed_ = X
        return self.clusterer.fit_predict(X)
