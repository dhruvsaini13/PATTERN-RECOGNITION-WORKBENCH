"""One call that runs the whole statistical analysis and returns JSON-ready output."""
from typing import Dict

import pandas as pd

from ..core.data import Dataset
from . import stats


def frame_to_records(df: pd.DataFrame, index: bool = False) -> list:
    if index:
        df = df.reset_index().rename(columns={"index": "feature"})
    return df.astype(object).where(df.notna(), None).to_dict("records")


def analyze(ds: Dataset, corr_threshold: float = 0.9) -> Dict:
    df = ds.features_frame()
    corr = stats.correlation_matrix(df)
    report = {
        "dataset": ds.info(),
        "summary": frame_to_records(stats.summary_statistics(df), index=True),
        "high_correlations": frame_to_records(stats.high_correlation_pairs(corr, corr_threshold)),
        "outliers_iqr": frame_to_records(stats.detect_outliers(df, "iqr")),
        "normality": frame_to_records(stats.normality_tests(df)),
    }
    if ds.y is not None:
        report["class_balance"] = stats.class_balance(ds.y, ds.class_names)
        report["feature_ranking"] = frame_to_records(stats.feature_ranking(ds.X, ds.y, ds.feature_names))
    return report
