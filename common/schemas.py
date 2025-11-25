"""Pydantic schemas shared across all trainer APIs."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl, validator


class AuthConfig(BaseModel):
    token: Optional[str] = Field(None, description="Bearer/API token")
    username: Optional[str] = Field(None, description="Username for basic auth")
    password: Optional[str] = Field(None, description="Password for basic auth")
    role_arn: Optional[str] = Field(None, description="Cloud role to assume")


class BaseModelSpec(BaseModel):
    provider: Literal["nemo", "meta", "huggingface", "bim", "custom"]
    model_name: str = Field(..., description="Model identifier, e.g. meta-llama/Llama-2-7b")
    revision: Optional[str] = Field("main", description="Git/model revision")
    weights_url: Optional[HttpUrl] = Field(None, description="Direct download URL override")
    auth_token: Optional[str] = Field(None, description="Provider-specific auth token")
    huggingface_token: Optional[str] = Field(
        None, description="Optional Hugging Face access token from Training Studio"
    )


class DatasetSpec(BaseModel):
    source: str = Field(..., description="URI or path to dataset")
    format: Literal["jsonl", "parquet", "csv", "hf", "nemo"]
    split: Optional[str] = Field(None, description="Dataset split name")
    auth: Optional[AuthConfig] = None
    streaming: bool = Field(False, description="Stream data instead of full download")


class LoRAConfig(BaseModel):
    rank: int = Field(16, ge=1)
    target_modules: List[str] = Field(default_factory=list)
    alpha: int = Field(32, ge=1)
    dropout: float = Field(0.05, ge=0, le=1)


class QLoRAConfig(BaseModel):
    use_double_quant: bool = True
    bnb_4bit_compute_dtype: Literal["float16", "bfloat16"] = "bfloat16"
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"


class PEFTConfig(BaseModel):
    use: bool = False
    config: Dict[str, Any] = Field(default_factory=dict)


class CustomizationOptions(BaseModel):
    method: Literal["full", "lora", "qlora", "peft", "adapter"] = "lora"
    trainable_layers: List[str] = Field(default_factory=list)
    precision: Literal["fp32", "fp16", "bf16", "int8"] = "bf16"
    gradient_checkpointing: bool = True
    lora: Optional[LoRAConfig] = None
    qlora: Optional[QLoRAConfig] = None
    peft: Optional[PEFTConfig] = None

    @validator("lora", always=True)
    def _lora_required(cls, value, values):  # type: ignore[override]
        if values.get("method") == "lora" and value is None:
            return LoRAConfig()
        return value

    @validator("qlora", always=True)
    def _qlora_required(cls, value, values):  # type: ignore[override]
        if values.get("method") == "qlora" and value is None:
            return QLoRAConfig()
        return value


class ResourceSpec(BaseModel):
    gpus: int = Field(1, ge=0)
    gpu_type: Optional[str] = None
    cpus: int = Field(8, ge=1)
    memory_gb: int = Field(64, ge=1)
    max_duration_minutes: int = Field(480, ge=1)


class ArtifactSpec(BaseModel):
    log_uri: str
    status_stream_url: Optional[str] = None
    output_uri: str


class CallbackSpec(BaseModel):
    webhook_url: Optional[HttpUrl] = None
    auth_header: Optional[str] = None
    event_filter: List[Literal["submitted", "running", "failed", "succeeded"]] = Field(
        default_factory=lambda: ["submitted", "failed", "succeeded"]
    )


class HyperParameterSpec(BaseModel):
    learning_rate: float = Field(5e-5, gt=0)
    batch_size: int = Field(32, ge=1)
    micro_batch_size: Optional[int] = Field(None, ge=1)
    num_epochs: int = Field(3, ge=1)
    warmup_ratio: float = Field(0.1, ge=0, le=1)
    max_sequence_length: int = Field(2048, ge=1)
    weight_decay: float = Field(0.01, ge=0)


class TrainingRequest(BaseModel):
    job_id: str
    base_model: BaseModelSpec
    datasets: List[DatasetSpec]
    customization: CustomizationOptions
    resources: ResourceSpec
    artifacts: ArtifactSpec
    tuning_parameters: HyperParameterSpec
    callbacks: Optional[CallbackSpec] = None
    extra_parameters: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        schema_extra = {"example": {"job_id": "job-123", "base_model": {"provider": "huggingface", "model_name": "meta-llama/Llama-2-7b"}, "datasets": [{"source": "s3://bucket/data.jsonl", "format": "jsonl"}], "customization": {"method": "lora"}, "resources": {"gpus": 4, "cpus": 32, "memory_gb": 256, "max_duration_minutes": 720}, "artifacts": {"log_uri": "s3://bucket/logs/", "output_uri": "s3://bucket/output/"}, "tuning_parameters": {"learning_rate": 2e-4, "batch_size": 64, "num_epochs": 3, "warmup_ratio": 0.1, "max_sequence_length": 4096}}}


class TrainingResponse(BaseModel):
    job_id: str
    backend_job_id: str
    status: Literal["submitted", "running", "failed", "succeeded"]
    detail: Optional[str] = None
    dashboard_url: Optional[HttpUrl] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
