"""Background jobs for slow work. Submit -> get a job_id -> poll GET /jobs/{job_id}."""
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException

from .. import store
from ..schemas import CompareRequest
from ..utils import require_labels
from .classification import run_compare

router = APIRouter(prefix="/jobs", tags=["jobs"])

_executor = ThreadPoolExecutor(max_workers=2)


def _submit(kind: str, fn, *args) -> str:
    job = {"kind": kind, "status": "pending", "result": None, "error": None, "created": time.time()}
    job_id = store.jobs.add(job)

    def runner():
        job["status"] = "running"
        try:
            job["result"] = fn(*args)
            job["status"] = "done"
        except HTTPException as e:
            job["error"], job["status"] = e.detail, "error"
        except Exception as e:  # report any failure to the poller instead of losing it
            job["error"], job["status"] = f"{type(e).__name__}: {e}", "error"

    _executor.submit(runner)
    return job_id


@router.post("/classification/compare", status_code=202)
def submit_compare(req: CompareRequest):
    ds = store.datasets.get(req.dataset_id)  # fail fast (404) instead of inside the job
    require_labels(ds)
    return {"job_id": _submit("classification_compare", run_compare, req)}


@router.get("/{job_id}")
def get_job(job_id: str):
    job = store.jobs.get(job_id)
    return {"job_id": job_id, "kind": job["kind"], "status": job["status"],
            "error": job["error"], "result": job["result"]}
