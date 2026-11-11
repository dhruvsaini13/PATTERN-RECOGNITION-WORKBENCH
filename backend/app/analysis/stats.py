"""Statistical analysis of a dataset. Every function returns a DataFrame or plain dict."""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.describe().T
    out["skew"] = df.skew()
    out["kurtosis"] = df.kurt()
    out["missing"] = df.isna().sum()
    return out.round(4)


def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    return df.corr(method=method)


def high_correlation_pairs(corr: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    cols = corr.columns
    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = corr.iloc[i, j]
            if abs(value) >= threshold:
                rows.append({"feature_a": cols[i], "feature_b": cols[j], "correlation": round(float(value), 4)})
    out = pd.DataFrame(rows, columns=["feature_a", "feature_b", "correlation"])
    return out.reindex(out["correlation"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def class_balance(y: np.ndarray, class_names: Optional[List[str]] = None) -> Dict:
    labels, counts = np.unique(y, return_counts=True)
    names = [class_names[l] if class_names else str(l) for l in labels]
    return {
        "classes": names,
        "counts": counts.tolist(),
        "percent": (counts / counts.sum() * 100).round(2).tolist(),
        "imbalance_ratio": round(float(counts.max() / counts.min()), 3),
    }


def detect_outliers(df: pd.DataFrame, method: str = "iqr", threshold: Optional[float] = None) -> pd.DataFrame:
    """method='iqr' (threshold = IQR multiplier, default 1.5) or 'zscore' (default 3.0)."""
    rows = []
    for col in df.columns:
        s = df[col].dropna()
        if method == "iqr":
            k = 1.5 if threshold is None else threshold
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            mask = (s < q1 - k * iqr) | (s > q3 + k * iqr)
        elif method == "zscore":
            t = 3.0 if threshold is None else threshold
            std = s.std()
            mask = pd.Series(False, index=s.index) if std == 0 else ((s - s.mean()).abs() / std > t)
        else:
            raise ValueError("method must be 'iqr' or 'zscore'")
        rows.append({"feature": col, "n_outliers": int(mask.sum()), "percent": round(float(mask.mean() * 100), 2)})
    return pd.DataFrame(rows)


def normality_tests(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Shapiro-Wilk for n <= 5000, D'Agostino-Pearson otherwise."""
    rows = []
    for col in df.columns:
        s = df[col].dropna().to_numpy()
        test, stat, p = None, np.nan, np.nan
        if len(s) >= 8 and np.std(s) > 0:
            if len(s) <= 5000:
                test = "shapiro"
                stat, p = sps.shapiro(s)
            else:
                test = "dagostino"
                stat, p = sps.normaltest(s)
        rows.append({
            "feature": col,
            "test": test,
            "statistic": None if np.isnan(stat) else round(float(stat), 4),
            "p_value": None if np.isnan(p) else float(p),
            "is_normal": None if np.isnan(p) else bool(p > alpha),
        })
    return pd.DataFrame(rows)


def feature_ranking(X: np.ndarray, y: np.ndarray, feature_names: List[str], random_state: int = 42) -> pd.DataFrame:
    """Rank features by ANOVA F-score, mutual information and random-forest importance."""
    f_scores, f_p = f_classif(X, y)
    f_scores = np.nan_to_num(f_scores)
    f_p = np.nan_to_num(f_p, nan=1.0)
    mi = mutual_info_classif(X, y, random_state=random_state)
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state).fit(X, y)

    out = pd.DataFrame({
        "feature": feature_names,
        "f_score": f_scores.round(4),
        "f_p_value": f_p,
        "mutual_info": mi.round(4),
        "rf_importance": rf.feature_importances_.round(4),
    })
    for col in ["f_score", "mutual_info", "rf_importance"]:
        out[f"rank_{col}"] = out[col].rank(ascending=False, method="min").astype(int)
    out["mean_rank"] = out[[c for c in out.columns if c.startswith("rank_")]].mean(axis=1).round(2)
    return out.sort_values("mean_rank").reset_index(drop=True)
