import io
import json

import numpy as np

from app.analysis import stats
from app.analysis.report import analyze
from app.core.data import list_builtin, load_builtin, load_csv, split_dataset


def test_builtin_datasets_load():
    for name in list_builtin():
        ds = load_builtin(name)
        assert ds.X.shape[0] == len(ds.y)
        assert len(ds.feature_names) == ds.X.shape[1]


def test_csv_loading_handles_text_and_missing():
    csv = io.StringIO("age,city,income,label\n25,Delhi,50000,yes\n30,Mumbai,,no\n,Delhi,70000,yes\n40,Pune,65000,no\n")
    ds = load_csv(csv, target_column="label")
    assert not np.isnan(ds.X).any()
    assert ds.class_names == ["no", "yes"]
    assert any(f.startswith("city_") for f in ds.feature_names)


def test_csv_bad_target():
    try:
        load_csv(io.StringIO("a,b\n1,2\n"), target_column="zzz")
    except ValueError as e:
        assert "not found" in str(e)


def test_split_is_stratified():
    ds = load_builtin("iris")
    Xtr, Xte, ytr, yte = split_dataset(ds, test_size=0.3)
    assert len(Xtr) + len(Xte) == ds.n_samples
    assert set(np.bincount(yte)) == {15}


def test_stats_functions():
    ds = load_builtin("wine")
    df = ds.features_frame()
    assert "skew" in stats.summary_statistics(df).columns
    assert stats.class_balance(ds.y)["imbalance_ratio"] >= 1
    assert len(stats.detect_outliers(df, "zscore")) == ds.n_features
    assert len(stats.normality_tests(df)) == ds.n_features
    ranking = stats.feature_ranking(ds.X, ds.y, ds.feature_names)
    assert ranking["mean_rank"].is_monotonic_increasing


def test_report_is_json_serializable():
    for name in list_builtin():
        json.dumps(analyze(load_builtin(name)))
