"""Classification: benchmark pipelines, train a model, predict, evaluation plots."""
import numpy as np
from fastapi import APIRouter, HTTPException

from ...analysis.plots import fig_to_base64
from ...analysis.report import frame_to_records
from ...core.data import split_dataset
from ...core.pipeline import ClassificationPipeline
from ...evaluation import plots as eval_plots
from ...evaluation.compare import compare_classifiers
from ...evaluation.metrics import classification_metrics
from .. import store
from ..schemas import CompareRequest, PredictRequest, TrainRequest
from ..utils import PLOT_LOCK, classification_configs, require_labels, resolve_module, resolve_steps

router = APIRouter(tags=["classification"])


def run_compare(req: CompareRequest) -> dict:
    """Shared by the synchronous endpoint and the background job."""
    ds = store.datasets.get(req.dataset_id)
    require_labels(ds)
    configs = classification_configs(req.configs)
    summary, details = compare_classifiers(
        ds, configs, test_size=req.test_size, cv_splits=req.cv_splits, random_state=req.random_state
    )
    return {
        "dataset": ds.info(),
        "summary": frame_to_records(summary),
        "metrics": {label: d["metrics"] for label, d in details.items()},
    }


@router.post("/classification/compare")
def compare(req: CompareRequest):
    return run_compare(req)


@router.post("/classification/train", status_code=201)
def train(req: TrainRequest):
    ds = store.datasets.get(req.dataset_id)
    require_labels(ds)
    steps, clf = resolve_steps(req.steps), resolve_module(req.classifier)

    Xtr, Xte, ytr, yte = split_dataset(ds, test_size=req.test_size, random_state=req.random_state)
    pipe = ClassificationPipeline(steps, clf).fit(Xtr, ytr)
    y_pred = pipe.predict(Xte)
    y_proba = pipe.predict_proba(Xte)
    metrics = classification_metrics(yte, y_pred, y_proba, ds.class_names)

    model_id = store.models.add({
        "pipeline": pipe,
        "dataset_id": req.dataset_id,
        "n_features": ds.n_features,
        "feature_names": ds.feature_names,
        "class_names": ds.class_names,
        "y_true": yte,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "metrics": metrics,
        "config": {"steps": [s.model_dump() for s in req.steps],
                   "classifier": req.classifier.model_dump()},
    })
    return {"model_id": model_id, "dataset_id": req.dataset_id, "n_features": ds.n_features,
            "feature_names": ds.feature_names, "class_names": ds.class_names, "metrics": metrics}


@router.get("/models/{model_id}")
def get_model(model_id: str):
    m = store.models.get(model_id)
    return {"model_id": model_id, "dataset_id": m["dataset_id"], "n_features": m["n_features"],
            "feature_names": m["feature_names"], "class_names": m["class_names"],
            "config": m["config"], "metrics": m["metrics"]}


@router.post("/models/{model_id}/predict")
def predict(model_id: str, req: PredictRequest):
    m = store.models.get(model_id)
    if any(len(r) != m["n_features"] for r in req.rows):
        raise ValueError(f"Every row must have exactly {m['n_features']} values "
                         f"(features: {m['feature_names']}).")
    X = np.asarray(req.rows, dtype=float)
    pipe = m["pipeline"]
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)
    names = m["class_names"] or [str(i) for i in range(int(pred.max()) + 1)]
    return {
        "predictions": [int(p) for p in pred],
        "labels": [names[int(p)] for p in pred],
        "probabilities": None if proba is None else np.round(proba, 4).tolist(),
    }


MODEL_PLOTS = ["confusion_matrix", "confusion_matrix_normalized", "roc"]


@router.get("/models/{model_id}/plots/{kind}")
def model_plot(model_id: str, kind: str):
    m = store.models.get(model_id)
    if kind not in MODEL_PLOTS:
        raise HTTPException(status_code=404, detail=f"Unknown plot '{kind}'. Available: {MODEL_PLOTS}")
    metrics = m["metrics"]
    with PLOT_LOCK:
        if kind == "roc":
            if m["y_proba"] is None:
                raise ValueError("This classifier does not provide probabilities, so no ROC curve is available.")
            fig = eval_plots.plot_roc_curves(m["y_true"], m["y_proba"], m["class_names"])
        else:
            fig = eval_plots.plot_confusion_matrix(
                metrics["confusion_matrix"], metrics["class_names"],
                normalize=kind.endswith("normalized"))
        image = fig_to_base64(fig)
    return {"kind": kind, "mime": "image/png", "image_base64": image}
