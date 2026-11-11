"""Performance metrics for classifiers and clusterers. Outputs are JSON-friendly."""
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    homogeneity_completeness_v_measure,
    matthews_corrcoef,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
    roc_auc_score,
    silhouette_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict:
    labels = np.unique(np.concatenate([y_true, y_pred]))
    names = [class_names[l] if class_names else str(l) for l in labels]

    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    p_c, r_c, f_c, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)

    roc_auc = None
    if y_proba is not None:
        try:
            if y_proba.shape[1] == 2:
                roc_auc = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                roc_auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
        except ValueError:
            roc_auc = None

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(p_mac),
        "recall_macro": float(r_mac),
        "f1_macro": float(f_mac),
        "precision_weighted": float(p_w),
        "recall_weighted": float(r_w),
        "f1_weighted": float(f_w),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "class_names": names,
        "per_class": [
            {"class": n, "precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for n, p, r, f, s in zip(names, p_c, r_c, f_c, support)
        ],
    }


def clustering_metrics(X: np.ndarray, labels: np.ndarray, y_true: Optional[np.ndarray] = None) -> Dict:
    """Internal metrics ignore noise points (label -1). External metrics need ground truth."""
    labels = np.asarray(labels)
    mask = labels != -1
    n_clusters = len(np.unique(labels[mask]))

    out = {
        "n_clusters": int(n_clusters),
        "n_noise": int((~mask).sum()),
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
    }
    if n_clusters >= 2 and mask.sum() > n_clusters:
        out["silhouette"] = float(silhouette_score(X[mask], labels[mask]))
        out["davies_bouldin"] = float(davies_bouldin_score(X[mask], labels[mask]))
        out["calinski_harabasz"] = float(calinski_harabasz_score(X[mask], labels[mask]))

    if y_true is not None:
        h, c, v = homogeneity_completeness_v_measure(y_true, labels)
        out.update({
            "adjusted_rand": float(adjusted_rand_score(y_true, labels)),
            "nmi": float(normalized_mutual_info_score(y_true, labels)),
            "homogeneity": float(h),
            "completeness": float(c),
            "v_measure": float(v),
        })
    return out
