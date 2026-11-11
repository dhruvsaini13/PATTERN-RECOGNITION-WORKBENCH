"""Common interfaces. Every module in the workbench implements one of these."""
from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class FeatureExtractor(ABC):
    name: str = ""
    default_params: Dict[str, Any] = {}

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "FeatureExtractor": ...

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray: ...

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class Classifier(ABC):
    name: str = ""
    default_params: Dict[str, Any] = {}

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Classifier": ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        return None


class Clusterer(ABC):
    name: str = ""
    default_params: Dict[str, Any] = {}

    @abstractmethod
    def fit_predict(self, X: np.ndarray) -> np.ndarray: ...
