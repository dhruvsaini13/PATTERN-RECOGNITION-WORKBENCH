"""Helpers shared by the routers: config conversion, safe parameters, plot lock."""
import math
import threading
from typing import Any, Dict, List

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif, mutual_info_classif

from ..core.data import Dataset
from .schemas import ClassificationConfig, ClusteringConfig, ModuleConfig

# JSON can't carry Python functions, so UI-facing params refer to them by name.
SCORE_FUNCS = {"f_classif": f_classif, "mutual_info_classif": mutual_info_classif}

# matplotlib is not thread-safe and FastAPI runs sync endpoints in a thread pool.
PLOT_LOCK = threading.Lock()

MAX_NUMERIC_PARAM = 1_000_000


def resolve_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(params or {})
    for key, value in out.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and (not math.isfinite(value) or abs(value) > MAX_NUMERIC_PARAM):
            raise ValueError(f"Parameter '{key}' is out of the allowed range (|value| <= {MAX_NUMERIC_PARAM}).")
    score_func = out.get("score_func")
    if isinstance(score_func, str):
        if score_func not in SCORE_FUNCS:
            raise ValueError(f"Unknown score_func '{score_func}'. Available: {list(SCORE_FUNCS)}")
        out["score_func"] = SCORE_FUNCS[score_func]
    return out


def resolve_module(cfg: ModuleConfig) -> Dict[str, Any]:
    return {"name": cfg.name, "params": resolve_params(cfg.params)}


def resolve_steps(steps: List[ModuleConfig]) -> List[Dict[str, Any]]:
    return [resolve_module(s) for s in steps]


def jsonable_params(params: Dict[str, Any]) -> Dict[str, Any]:
    return {k: (v.__name__ if callable(v) else v) for k, v in params.items()}


def _unique(label: str, taken: set) -> str:
    candidate, n = label, 2
    while candidate in taken:
        candidate, n = f"{label} ({n})", n + 1
    taken.add(candidate)
    return candidate


def classification_configs(configs: List[ClassificationConfig]) -> List[Dict[str, Any]]:
    taken: set = set()
    out = []
    for c in configs:
        steps = resolve_steps(c.steps)
        clf = resolve_module(c.classifier)
        label = c.label or "+".join([s["name"] for s in steps] + [clf["name"]])
        out.append({"label": _unique(label, taken), "steps": steps, "classifier": clf})
    return out


def clustering_configs(configs: List[ClusteringConfig]) -> List[Dict[str, Any]]:
    taken: set = set()
    out = []
    for c in configs:
        steps = resolve_steps(c.steps)
        clu = resolve_module(c.clusterer)
        label = c.label or "+".join([s["name"] for s in steps] + [clu["name"]])
        out.append({"label": _unique(label, taken), "steps": steps, "clusterer": clu})
    return out


def require_labels(ds: Dataset) -> None:
    if ds.y is None:
        raise ValueError("This dataset has no labels. Upload it with a target_column to use classification.")


def project_2d(features: np.ndarray) -> np.ndarray:
    if features.shape[1] == 2:
        return features
    if features.shape[1] == 1:
        return np.hstack([features, np.zeros((len(features), 1))])
    return PCA(n_components=2).fit_transform(features)
