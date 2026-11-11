"""Clustering: run one pipeline, compare several, sweep the number of clusters."""
import numpy as np
from fastapi import APIRouter

from ...analysis.report import frame_to_records
from ...core.pipeline import ClusteringPipeline
from ...evaluation.compare import compare_clusterers, sweep_n_clusters
from ...evaluation.metrics import clustering_metrics
from .. import store
from ..schemas import ClusterCompareRequest, ClusterRunRequest, SweepRequest
from ..utils import clustering_configs, project_2d, resolve_module, resolve_steps

router = APIRouter(prefix="/clustering", tags=["clustering"])

MAX_POINTS_SHOWN = 2000
MAX_LABELS_RETURNED = 5000
SWEEPABLE = ["kmeans", "hierarchical", "gmm"]


@router.post("/run")
def run(req: ClusterRunRequest):
    ds = store.datasets.get(req.dataset_id)
    pipe = ClusteringPipeline(resolve_steps(req.steps), resolve_module(req.clusterer))
    labels = pipe.fit_predict(ds.X)  # unsupervised: labels are never passed to the pipeline
    metrics = clustering_metrics(pipe.transformed_, labels, ds.y)

    Z = project_2d(pipe.transformed_)
    truth = None  # the real classes, only used to colour a second scatter plot for comparison
    if ds.y is not None:
        class_names = ds.class_names or [str(i) for i in range(ds.n_classes)]
        truth = [class_names[int(c)] for c in ds.y]
    idx = np.arange(len(labels))
    if len(idx) > MAX_POINTS_SHOWN:
        idx = np.sort(np.random.RandomState(0).choice(idx, MAX_POINTS_SHOWN, replace=False))

    return {
        "dataset": ds.info(),
        "metrics": metrics,
        "cluster_sizes": {str(int(l)): int((labels == l).sum()) for l in np.unique(labels)},
        "points": [
            {"x": float(Z[i, 0]), "y": float(Z[i, 1]), "label": int(labels[i]), "truth": truth[i] if truth else None}
            for i in idx
        ],
        "n_points_shown": int(len(idx)),
        "labels": labels.tolist() if len(labels) <= MAX_LABELS_RETURNED else None,
    }


@router.post("/compare")
def compare(req: ClusterCompareRequest):
    ds = store.datasets.get(req.dataset_id)
    summary, _ = compare_clusterers(ds, clustering_configs(req.configs))
    return {"dataset": ds.info(), "summary": frame_to_records(summary)}


@router.post("/sweep")
def sweep(req: SweepRequest):
    ds = store.datasets.get(req.dataset_id)
    if req.clusterer not in SWEEPABLE:
        raise ValueError(f"Sweeping the number of clusters only applies to {SWEEPABLE}.")
    if req.k_min > req.k_max:
        raise ValueError("k_min must be <= k_max.")
    result = sweep_n_clusters(ds.X, req.clusterer, range(req.k_min, req.k_max + 1), resolve_steps(req.steps))
    return {"dataset": ds.info(), "clusterer": req.clusterer, "sweep": frame_to_records(result)}
