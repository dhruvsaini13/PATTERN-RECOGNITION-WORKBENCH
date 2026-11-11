"""Dataset lifecycle, statistical analysis and data plots."""
import io
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ...analysis import plots
from ...analysis.report import analyze, frame_to_records
from ...analysis.stats import class_balance, correlation_matrix
from ...core.data import Dataset, list_builtin, load_builtin, load_csv
from .. import store
from ..utils import PLOT_LOCK

router = APIRouter(prefix="/datasets", tags=["datasets"])

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def _summary(dataset_id: str, ds: Dataset) -> dict:
    return {"dataset_id": dataset_id, **ds.info()}


@router.get("/builtin")
def builtin_names():
    return {"datasets": list_builtin()}


@router.post("/builtin/{name}", status_code=201)
def load_builtin_dataset(name: str):
    try:
        ds = load_builtin(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=e.args[0])
    return _summary(store.datasets.add(ds), ds)


@router.post("/upload", status_code=201)
def upload_csv(
    file: UploadFile = File(...),
    target_column: Optional[str] = Form(None),
    drop_columns: str = Form(""),
    missing: str = Form("median"),
    encode_categorical: bool = Form(True),
):
    raw = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (limit 10 MB).")
    drops = [c.strip() for c in drop_columns.split(",") if c.strip()]
    ds = load_csv(
        io.BytesIO(raw),
        target_column=target_column or None,
        drop_columns=drops,
        missing=missing,
        encode_categorical=encode_categorical,
        name=file.filename or "uploaded_csv",
    )
    if ds.n_features == 0:
        raise ValueError("No usable numeric features found in this CSV.")
    if ds.y is not None and ds.n_classes < 2:
        raise ValueError("The target column has fewer than 2 distinct classes.")
    if ds.y is not None and ds.n_classes > 50 and ds.n_classes > 0.2 * ds.n_samples:
        raise ValueError(
            f"The target column has {ds.n_classes} distinct values across {ds.n_samples} rows, so it looks like a "
            "continuous number (such as an amount) rather than class labels. This workbench does classification: "
            "pick a categorical target column, or leave the target empty to use statistics and clustering."
        )
    return _summary(store.datasets.add(ds), ds)


@router.get("")
def list_datasets():
    return {"datasets": [_summary(i, ds) for i, ds in store.datasets.items()]}


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str):
    return _summary(dataset_id, store.datasets.get(dataset_id))


@router.delete("/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: str):
    store.datasets.delete(dataset_id)


@router.get("/{dataset_id}/preview")
def preview(dataset_id: str, rows: int = 10):
    ds = store.datasets.get(dataset_id)
    rows = max(1, min(rows, 100))
    df = ds.features_frame().head(rows)
    if ds.y is not None:
        names = ds.class_names or [str(i) for i in range(ds.n_classes)]
        df = df.assign(target=[names[i] for i in ds.y[:rows]])
    return {"columns": list(df.columns), "rows": frame_to_records(df)}


@router.get("/{dataset_id}/analysis")
def dataset_analysis(dataset_id: str, corr_threshold: float = 0.9):
    return analyze(store.datasets.get(dataset_id), corr_threshold)


PLOT_KINDS = ["histograms", "correlation", "boxplots", "class_balance", "pca"]


@router.get("/{dataset_id}/plots/{kind}")
def dataset_plot(dataset_id: str, kind: str):
    ds = store.datasets.get(dataset_id)
    if kind not in PLOT_KINDS:
        raise HTTPException(status_code=404, detail=f"Unknown plot '{kind}'. Available: {PLOT_KINDS}")
    if kind in ("class_balance",) and ds.y is None:
        raise ValueError("This dataset has no labels.")
    if kind == "pca" and ds.n_features < 2:
        raise ValueError("PCA projection needs at least 2 features.")

    df = ds.features_frame()
    with PLOT_LOCK:
        if kind == "histograms":
            fig = plots.plot_histograms(df)
        elif kind == "correlation":
            fig = plots.plot_correlation_heatmap(correlation_matrix(df))
        elif kind == "boxplots":
            fig = plots.plot_boxplots(df)
        elif kind == "class_balance":
            fig = plots.plot_class_balance(class_balance(ds.y, ds.class_names))
        else:
            fig = plots.plot_pca_2d(ds.X, ds.y, ds.class_names)
        image = plots.fig_to_base64(fig)
    return {"kind": kind, "mime": "image/png", "image_base64": image}
