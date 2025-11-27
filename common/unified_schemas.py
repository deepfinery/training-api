"""Lightweight Pydantic schemas for the unified training API."""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, validator


class Framework(str, Enum):
    """Supported training frameworks handled by the dispatcher."""

    NEMO = "nemo"
    HF = "hf"
    META = "meta"


def _normalize_uri(uri: str) -> str:
    if uri.endswith("/"):
        return uri.rstrip("/")
    return uri


class TrainingJobRequest(BaseModel):
    """Incoming payload that captures what to launch on Kubernetes."""

    framework: Framework = Field(..., description="Framework dispatcher to run")
    model_id: str = Field(..., description="Identifier of the model to fine-tune")
    run_id: str = Field(..., description="Unique identifier assigned by the API")
    checkpoint_base_uri: str = Field(
        ..., description="Base S3 URI where checkpoints/logs/configs are stored"
    )
    dataset_uri: str = Field(..., description="Input dataset location")
    image: str = Field(
        "ghcr.io/deepfinery/trainer:latest",
        description="Container image that ships with train.py + dependencies",
    )
    num_nodes: int = Field(1, ge=1, description="Number of PyTorchJob workers")
    gpus_per_node: int = Field(1, ge=1, description="torchrun --nproc_per_node")
    cpus_per_node: int = Field(32, ge=1, description="CPU request per replica")
    memory_per_node_gb: int = Field(256, ge=1, description="Memory request per replica")
    config_path: Optional[str] = Field(
        None, description="Optional configuration artifact to hand to train.py"
    )
    resume_from_checkpoint: bool = Field(
        False, description="Resume from the most recent checkpoint if available"
    )
    extra_args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Framework-specific overrides forwarded to the container",
    )
    env: Dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary environment overrides merged into every replica",
    )
    labels: Dict[str, str] = Field(default_factory=dict, description="Kubernetes labels")
    annotations: Dict[str, str] = Field(
        default_factory=dict, description="Kubernetes annotations"
    )
    namespace: Optional[str] = Field(
        None, description="Optional namespace override for the PyTorchJob"
    )
    job_name: Optional[str] = Field(
        None, description="Optional explicit name for the PyTorchJob resource"
    )

    @validator("checkpoint_base_uri", "dataset_uri", pre=True)
    def _strip_trailing_slash(cls, value: str) -> str:  # type: ignore[override]
        if not isinstance(value, str):
            raise TypeError("URI fields must be strings")
        return _normalize_uri(value)

    @property
    def checkpoint_prefix(self) -> str:
        return f"{self.checkpoint_base_uri}/{self.run_id}/framework={self.framework.value}"

    @property
    def checkpoints_uri(self) -> str:
        return f"{self.checkpoint_prefix}/checkpoints"

    @property
    def logs_uri(self) -> str:
        return f"{self.checkpoint_prefix}/logs"

    @property
    def config_uri(self) -> str:
        return f"{self.checkpoint_prefix}/config"


class TrainingJobStatus(BaseModel):
    name: str
    status: str
    detail: Optional[str] = None
    framework: Framework
    run_id: str

