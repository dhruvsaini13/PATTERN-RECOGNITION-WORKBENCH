"""FastAPI entry point.  Run from the backend/ folder:  uvicorn app.api.main:app --reload"""
import matplotlib

matplotlib.use("Agg")  # server has no display

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from . import store  # noqa: E402
from .routers import catalog, classification, clustering, datasets, jobs  # noqa: E402

app = FastAPI(
    title="Pattern Recognition Workbench API",
    version="0.1.0",
    description="Feature extraction, statistics, classification, clustering and evaluation as a service.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_methods=["*"],
    allow_headers=["*"],
)

for r in (catalog.router, datasets.router, classification.router, clustering.router, jobs.router):
    app.include_router(r)


@app.get("/health", tags=["catalog"])
def health():
    return {"status": "ok", "datasets": len(store.datasets), "models": len(store.models)}


# Bad user input (wrong parameter, impossible configuration) -> 422 with a readable message.
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(KeyError)
async def key_error_handler(request: Request, exc: KeyError):
    msg = str(exc.args[0]) if exc.args else ""
    if msg.startswith("Unknown "):  # registry lookup of a module name that doesn't exist
        return JSONResponse(status_code=422, content={"detail": msg})
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.exception_handler(TypeError)
async def type_error_handler(request: Request, exc: TypeError):
    if "unexpected keyword argument" in str(exc):  # a parameter name the module doesn't accept
        return JSONResponse(status_code=422, content={"detail": f"Invalid parameter: {exc}"})
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
