"""Dataset container, built-in datasets, CSV loading and train/test splitting."""
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn import datasets as skd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BUILTIN_LOADERS = {
    "iris": skd.load_iris,
    "wine": skd.load_wine,
    "breast_cancer": skd.load_breast_cancer,
    "digits": skd.load_digits,
}


@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: Optional[np.ndarray]
    feature_names: List[str]
    class_names: Optional[List[str]] = None

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def n_classes(self) -> int:
        return 0 if self.y is None else len(np.unique(self.y))

    def features_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.X, columns=self.feature_names)

    def info(self) -> dict:
        return {
            "name": self.name,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "has_labels": self.y is not None,
            "class_names": self.class_names,
        }


def list_builtin() -> List[str]:
    return list(BUILTIN_LOADERS)


def load_builtin(name: str) -> Dataset:
    if name not in BUILTIN_LOADERS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {list_builtin()}")
    bunch = BUILTIN_LOADERS[name]()
    return Dataset(
        name=name,
        X=np.asarray(bunch.data, dtype=float),
        y=np.asarray(bunch.target),
        feature_names=[str(f) for f in bunch.feature_names],
        class_names=[str(c) for c in bunch.target_names],
    )


def load_csv(
    source,
    target_column: Optional[str] = None,
    drop_columns: Sequence[str] = (),
    missing: str = "median",
    encode_categorical: bool = True,
    name: str = "uploaded_csv",
) -> Dataset:
    """Load a CSV (path or file-like object) into a numeric Dataset.

    missing: 'median' | 'mean' | 'zero' | 'drop'
    encode_categorical: one-hot encode text features (otherwise they are dropped)
    """
    df = pd.read_csv(source)
    df = df.drop(columns=list(drop_columns), errors="ignore")

    y, class_names = None, None
    if target_column is not None:
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found. Columns: {list(df.columns)}")
        df = df.dropna(subset=[target_column]).reset_index(drop=True)
        encoder = LabelEncoder()
        y = encoder.fit_transform(df[target_column].astype(str))
        class_names = [str(c) for c in encoder.classes_]
        df = df.drop(columns=[target_column])

    text_cols = list(df.select_dtypes(exclude="number").columns)
    if text_cols:
        if encode_categorical:
            df = pd.get_dummies(df, columns=text_cols, dtype=float)
        else:
            df = df.drop(columns=text_cols)

    df = df.astype(float)

    if missing == "drop":
        keep = df.notna().all(axis=1).to_numpy()
        df = df[keep].reset_index(drop=True)
        y = y[keep] if y is not None else None
    elif missing == "median":
        df = df.fillna(df.median())
    elif missing == "mean":
        df = df.fillna(df.mean())
    elif missing == "zero":
        df = df.fillna(0.0)
    else:
        raise ValueError("missing must be one of: median, mean, zero, drop")

    return Dataset(name=name, X=df.to_numpy(), y=y, feature_names=list(df.columns), class_names=class_names)


def split_dataset(ds: Dataset, test_size: float = 0.25, random_state: int = 42, stratify: bool = True):
    """Returns X_train, X_test, y_train, y_test."""
    if ds.y is None:
        raise ValueError("Cannot split an unlabeled dataset into train/test for classification.")
    strat = ds.y if stratify and np.bincount(ds.y).min() >= 2 else None
    return train_test_split(ds.X, ds.y, test_size=test_size, random_state=random_state, stratify=strat)
