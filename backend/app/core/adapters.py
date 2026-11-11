"""Thin wrappers that turn scikit-learn estimators into workbench modules."""
from .base import Classifier, Clusterer, FeatureExtractor


class SklearnExtractor(FeatureExtractor):
    estimator_cls = None

    def __init__(self, **params):
        self.params = {**self.default_params, **params}
        self.model = self.estimator_cls(**self.params)

    def fit(self, X, y=None):
        self.model.fit(X, y)
        return self

    def transform(self, X):
        return self.model.transform(X)


class SklearnClassifier(Classifier):
    estimator_cls = None

    def __init__(self, **params):
        self.params = {**self.default_params, **params}
        self.model = self.estimator_cls(**self.params)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X) if hasattr(self.model, "predict_proba") else None


class SklearnClusterer(Clusterer):
    estimator_cls = None

    def __init__(self, **params):
        self.params = {**self.default_params, **params}
        self.model = self.estimator_cls(**self.params)

    def fit_predict(self, X):
        return self.model.fit_predict(X)
