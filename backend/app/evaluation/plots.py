"""Evaluation plots. Each function returns a Figure (use fig_to_base64 from analysis.plots for the API)."""
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize


def plot_confusion_matrix(cm, class_names: List[str], normalize: bool = False, title: str = "Confusion matrix"):
    cm = np.array(cm, dtype=float)
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(1.2 * len(class_names) + 3, 1.0 * len(class_names) + 2.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def plot_roc_curves(y_true: np.ndarray, y_proba: np.ndarray, class_names: Optional[List[str]] = None, title="ROC curves"):
    n_classes = y_proba.shape[1]
    names = class_names or [str(i) for i in range(n_classes)]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.3f}")
    else:
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        for i in range(n_classes):
            if y_bin[:, i].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            ax.plot(fpr, tpr, label=f"{names[i]} (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def plot_model_comparison(summary: pd.DataFrame):
    labels = summary["label"].tolist()
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(labels) + 2))
    ax.barh(y - 0.2, summary["cv_f1_macro_mean"], height=0.38, xerr=summary["cv_f1_macro_std"],
            label="CV F1 (mean ± std)", color="#4C72B0")
    ax.barh(y + 0.2, summary["test_f1_macro"], height=0.38, label="Test F1", color="#DD8452")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(max(0, summary["cv_f1_macro_mean"].min() - 0.15), 1.02)
    ax.set_xlabel("Macro F1")
    ax.set_title("Model comparison")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def plot_clusters(features: np.ndarray, labels: np.ndarray, title: str = "Clusters"):
    Z = features if features.shape[1] == 2 else PCA(n_components=2).fit_transform(features)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for lab in np.unique(labels):
        m = labels == lab
        if lab == -1:
            ax.scatter(Z[m, 0], Z[m, 1], s=14, c="lightgray", marker="x", label="noise")
        else:
            ax.scatter(Z[m, 0], Z[m, 1], s=14, alpha=0.75, label=f"cluster {lab}")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_k_sweep(sweep: pd.DataFrame, title: str = "Choosing the number of clusters"):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    if sweep["inertia"].notna().any():
        axes[0].plot(sweep["k"], sweep["inertia"], "o-")
        axes[0].set_title("Inertia (elbow)")
    else:
        axes[0].axis("off")
    axes[1].plot(sweep["k"], sweep["silhouette"], "o-", color="#55A868")
    axes[1].set_title("Silhouette (higher is better)")
    axes[2].plot(sweep["k"], sweep["davies_bouldin"], "o-", color="#C44E52")
    axes[2].set_title("Davies-Bouldin (lower is better)")
    for ax in axes:
        ax.set_xlabel("k")
    fig.suptitle(title, y=1.04)
    fig.tight_layout()
    return fig
