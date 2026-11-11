"""Request bodies. These double as the contract the frontend builds against (see /docs)."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ModuleConfig(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class ClassificationConfig(BaseModel):
    label: Optional[str] = None
    steps: List[ModuleConfig] = Field(default_factory=list)
    classifier: ModuleConfig


class ClusteringConfig(BaseModel):
    label: Optional[str] = None
    steps: List[ModuleConfig] = Field(default_factory=list)
    clusterer: ModuleConfig


class CompareRequest(BaseModel):
    dataset_id: str
    configs: List[ClassificationConfig] = Field(min_length=1, max_length=20)
    test_size: float = Field(0.25, gt=0.05, lt=0.6)
    cv_splits: int = Field(5, ge=2, le=10)
    random_state: int = 42


class TrainRequest(BaseModel):
    dataset_id: str
    steps: List[ModuleConfig] = Field(default_factory=list)
    classifier: ModuleConfig
    test_size: float = Field(0.25, gt=0.05, lt=0.6)
    random_state: int = 42


class PredictRequest(BaseModel):
    rows: List[List[float]] = Field(min_length=1, max_length=1000)


class ClusterRunRequest(BaseModel):
    dataset_id: str
    steps: List[ModuleConfig] = Field(default_factory=list)
    clusterer: ModuleConfig


class ClusterCompareRequest(BaseModel):
    dataset_id: str
    configs: List[ClusteringConfig] = Field(min_length=1, max_length=20)


class SweepRequest(BaseModel):
    dataset_id: str
    clusterer: str = "kmeans"
    k_min: int = Field(2, ge=2, le=30)
    k_max: int = Field(8, ge=2, le=30)
    steps: List[ModuleConfig] = Field(default_factory=list)
