"""Discovery endpoint: the frontend builds its parameter forms from this."""
from fastapi import APIRouter

from ... import modules as _modules  # noqa: F401  (registers the built-in modules)
from ...core.registry import CLASSIFIERS, CLUSTERERS, FEATURE_EXTRACTORS
from ..utils import SCORE_FUNCS, jsonable_params

router = APIRouter(tags=["catalog"])


def _describe(registry):
    return [{"name": m["name"], "params": jsonable_params(m["params"])} for m in registry.describe()]


@router.get("/modules")
def list_modules():
    return {
        "feature_extractors": _describe(FEATURE_EXTRACTORS),
        "classifiers": _describe(CLASSIFIERS),
        "clusterers": _describe(CLUSTERERS),
        "score_funcs": list(SCORE_FUNCS),
    }
