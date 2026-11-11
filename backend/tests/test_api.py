import io
import time

import matplotlib
matplotlib.use("Agg")
import pytest
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


@pytest.fixture(scope="module")
def wine_id():
    r = client.post("/datasets/builtin/wine")
    assert r.status_code == 201
    return r.json()["dataset_id"]


def test_health_and_catalog():
    assert client.get("/health").json()["status"] == "ok"
    mods = client.get("/modules").json()
    assert {m["name"] for m in mods["classifiers"]} >= {"svm", "knn", "random_forest"}
    skb = next(m for m in mods["feature_extractors"] if m["name"] == "select_k_best")
    assert skb["params"]["score_func"] == "f_classif"  # function exposed by name, JSON-safe


def test_builtin_and_dataset_lifecycle(wine_id):
    assert "wine" in client.get("/datasets/builtin").json()["datasets"]
    info = client.get(f"/datasets/{wine_id}").json()
    assert info["n_features"] == 13 and info["n_classes"] == 3
    preview = client.get(f"/datasets/{wine_id}/preview?rows=3").json()
    assert len(preview["rows"]) == 3 and "target" in preview["columns"]
    tmp = client.post("/datasets/builtin/iris").json()["dataset_id"]
    assert client.delete(f"/datasets/{tmp}").status_code == 204
    assert client.get(f"/datasets/{tmp}").status_code == 404
    assert client.post("/datasets/builtin/nope").status_code == 404


def test_csv_upload_and_errors():
    csv = b"a,b,label\n1,2,x\n2,3,y\n3,4,x\n4,5,y\n5,6,x\n6,7,y\n"
    r = client.post("/datasets/upload", files={"file": ("t.csv", io.BytesIO(csv), "text/csv")},
                    data={"target_column": "label"})
    assert r.status_code == 201 and r.json()["n_classes"] == 2
    bad = client.post("/datasets/upload", files={"file": ("t.csv", io.BytesIO(csv), "text/csv")},
                      data={"target_column": "missing_col"})
    assert bad.status_code == 422 and "not found" in bad.json()["detail"]
    one_class = client.post("/datasets/upload", files={"file": ("t.csv", io.BytesIO(b"a,label\n1,x\n2,x\n"), "text/csv")},
                            data={"target_column": "label"})
    assert one_class.status_code == 422


def test_analysis_and_plots(wine_id):
    rep = client.get(f"/datasets/{wine_id}/analysis").json()
    assert {"summary", "class_balance", "feature_ranking", "normality"} <= set(rep)
    for kind in ["histograms", "correlation", "boxplots", "class_balance", "pca"]:
        r = client.get(f"/datasets/{wine_id}/plots/{kind}")
        assert r.status_code == 200 and len(r.json()["image_base64"]) > 1000
    assert client.get(f"/datasets/{wine_id}/plots/nope").status_code == 404


def test_classification_compare_train_predict(wine_id):
    body = {"dataset_id": wine_id, "configs": [
        {"classifier": {"name": "svm"}, "steps": [{"name": "standard_scaler"}]},
        {"label": "lda+knn", "classifier": {"name": "knn", "params": {"n_neighbors": 7}},
         "steps": [{"name": "standard_scaler"}, {"name": "lda", "params": {"n_components": 2}}]},
        {"classifier": {"name": "svm"}, "steps": [{"name": "standard_scaler"}]},  # duplicate label
        {"classifier": {"name": "random_forest"},
         "steps": [{"name": "select_k_best", "params": {"k": 5, "score_func": "mutual_info_classif"}}]},
    ]}
    r = client.post("/classification/compare", json=body)
    assert r.status_code == 200
    out = r.json()
    labels = [row["label"] for row in out["summary"]]
    assert len(labels) == len(set(labels)) == 4 and "lda+knn" in labels
    assert out["summary"][0]["test_accuracy"] > 0.85
    assert set(out["metrics"]) == set(labels)

    t = client.post("/classification/train", json={
        "dataset_id": wine_id, "steps": [{"name": "standard_scaler"}], "classifier": {"name": "svm"}})
    assert t.status_code == 201
    model_id = t.json()["model_id"]
    assert client.get(f"/models/{model_id}").json()["n_features"] == 13

    row = [13.2, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050]
    p = client.post(f"/models/{model_id}/predict", json={"rows": [row, row]}).json()
    assert len(p["predictions"]) == 2 and len(p["probabilities"][0]) == 3
    assert p["labels"][0] in ("class_0", "class_1", "class_2")

    wrong = client.post(f"/models/{model_id}/predict", json={"rows": [[1, 2, 3]]})
    assert wrong.status_code == 422 and "13" in wrong.json()["detail"]

    for kind in ["confusion_matrix", "confusion_matrix_normalized", "roc"]:
        assert client.get(f"/models/{model_id}/plots/{kind}").status_code == 200
    assert client.get("/models/doesnotexist").status_code == 404


def test_bad_configs_return_422_not_500(wine_id):
    def compare(cfg):
        return client.post("/classification/compare", json={"dataset_id": wine_id, "configs": [cfg]})

    r = compare({"classifier": {"name": "no_such_model"}})
    assert r.status_code == 422 and "Unknown classifier" in r.json()["detail"]
    r = compare({"classifier": {"name": "knn", "params": {"bogus": 1}}})
    assert r.status_code == 422 and "bogus" in r.json()["detail"]
    r = compare({"classifier": {"name": "knn"}, "steps": [{"name": "lda", "params": {"n_components": 9}}]})
    assert r.status_code == 422
    r = compare({"classifier": {"name": "random_forest", "params": {"n_estimators": 10**12}}})
    assert r.status_code == 422 and "allowed range" in r.json()["detail"]
    r = compare({"classifier": {"name": "knn"}, "steps": [{"name": "select_k_best", "params": {"score_func": "nope"}}]})
    assert r.status_code == 422
    assert client.post("/classification/compare", json={"dataset_id": "zzz", "configs": [
        {"classifier": {"name": "knn"}}]}).status_code == 404


def test_classification_requires_labels():
    csv = b"a,b\n1,2\n2,3\n3,4\n4,5\n5,6\n6,7\n7,8\n8,9\n"
    did = client.post("/datasets/upload", files={"file": ("u.csv", io.BytesIO(csv), "text/csv")}).json()["dataset_id"]
    r = client.post("/classification/train", json={"dataset_id": did, "classifier": {"name": "knn"}})
    assert r.status_code == 422 and "no labels" in r.json()["detail"]
    c = client.post("/clustering/run", json={"dataset_id": did, "clusterer": {"name": "kmeans", "params": {"n_clusters": 2}}})
    assert c.status_code == 200 and c.json()["metrics"]["n_clusters"] == 2
    assert c.json()["points"][0]["truth"] is None
    assert "adjusted_rand" not in c.json()["metrics"]


def test_clustering_endpoints(wine_id):
    run = client.post("/clustering/run", json={
        "dataset_id": wine_id, "steps": [{"name": "standard_scaler"}],
        "clusterer": {"name": "kmeans", "params": {"n_clusters": 3}}}).json()
    assert run["metrics"]["n_clusters"] == 3 and run["metrics"]["adjusted_rand"] > 0.7
    assert len(run["points"]) == 178 and sum(run["cluster_sizes"].values()) == 178
    assert run["points"][0]["truth"] in ("class_0", "class_1", "class_2")

    comp = client.post("/clustering/compare", json={"dataset_id": wine_id, "configs": [
        {"steps": [{"name": "standard_scaler"}], "clusterer": {"name": n}} for n in ["kmeans", "gmm", "dbscan"]]})
    assert comp.status_code == 200 and len(comp.json()["summary"]) == 3

    sw = client.post("/clustering/sweep", json={"dataset_id": wine_id, "k_min": 2, "k_max": 5,
                                                "steps": [{"name": "standard_scaler"}]}).json()
    assert [row["k"] for row in sw["sweep"]] == [2, 3, 4, 5]
    assert client.post("/clustering/sweep", json={"dataset_id": wine_id, "clusterer": "dbscan"}).status_code == 422
    assert client.post("/clustering/sweep", json={"dataset_id": wine_id, "k_min": 6, "k_max": 3}).status_code == 422


def test_background_job_flow(wine_id):
    body = {"dataset_id": wine_id, "configs": [{"classifier": {"name": "svm"}, "steps": [{"name": "standard_scaler"}]}]}
    r = client.post("/jobs/classification/compare", json=body)
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    for _ in range(100):
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert job["status"] == "done" and job["result"]["summary"][0]["label"] == "standard_scaler+svm"

    bad = client.post("/jobs/classification/compare", json={**body, "configs": [{"classifier": {"name": "nope"}}]}).json()["job_id"]
    for _ in range(100):
        job = client.get(f"/jobs/{bad}").json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert job["status"] == "error" and "Unknown classifier" in job["error"]
    assert client.get("/jobs/nope").status_code == 404
    assert client.post("/jobs/classification/compare", json={**body, "dataset_id": "zzz"}).status_code == 404


def test_continuous_target_is_rejected():
    rows = "\n".join(f"{i},{i * 1.37:.2f}" for i in range(300))
    csv = ("x,loss\n" + rows + "\n").encode()
    r = client.post("/datasets/upload", files={"file": ("c.csv", io.BytesIO(csv), "text/csv")}, data={"target_column": "loss"})
    assert r.status_code == 422 and "continuous" in r.json()["detail"]
    ok = client.post("/datasets/upload", files={"file": ("c.csv", io.BytesIO(csv), "text/csv")})
    assert ok.status_code == 201 and ok.json()["has_labels"] is False
