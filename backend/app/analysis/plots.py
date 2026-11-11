"""Matplotlib plots. Each function returns a Figure; fig_to_base64 is for the API later."""
import base64
import io
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def plot_histograms(df: pd.DataFrame, max_cols: int = 12, bins: int = 20):
    cols = list(df.columns[:max_cols])
    n_cols = min(4, len(cols))
    n_rows = int(np.ceil(len(cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    for ax, col in zip(axes.ravel(), cols):
        ax.hist(df[col].dropna(), bins=bins, color="#4C72B0", edgecolor="white")
        ax.set_title(col, fontsize=9)
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    fig.suptitle("Feature distributions", y=1.02)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(corr: pd.DataFrame, max_features: int = 20):
    corr = corr.iloc[:max_features, :max_features]
    size = max(5, 0.45 * len(corr))
    fig, ax = plt.subplots(figsize=(size + 1, size))
    im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Correlation matrix")
    fig.tight_layout()
    return fig


def plot_boxplots(df: pd.DataFrame, max_cols: int = 12):
    cols = list(df.columns[:max_cols])
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(cols)), 4))
    ax.boxplot([df[c].dropna() for c in cols])
    ax.set_xticklabels(cols, rotation=60, ha="right")
    ax.set_title("Boxplots (outlier view)")
    fig.tight_layout()
    return fig


def plot_class_balance(balance: dict):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(balance["classes"], balance["counts"], color="#55A868")
    ax.bar_label(bars)
    ax.set_title(f"Class balance (imbalance ratio {balance['imbalance_ratio']})")
    ax.set_ylabel("samples")
    fig.tight_layout()
    return fig


def plot_pca_2d(X: np.ndarray, y: Optional[np.ndarray] = None, class_names: Optional[List[str]] = None):
    Z = PCA(n_components=2).fit_transform(X)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    if y is None:
        ax.scatter(Z[:, 0], Z[:, 1], s=14, alpha=0.7)
    else:
        for label in np.unique(y):
            m = y == label
            ax.scatter(Z[m, 0], Z[m, 1], s=14, alpha=0.7,
                       label=class_names[label] if class_names else str(label))
        ax.legend(fontsize=8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("2D PCA projection")
    fig.tight_layout()
    return fig
