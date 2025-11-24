"""Hugging Face + Unsloth accelerated trainer backend."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from common import BaseTrainerBackend, TrainingRequest  # noqa: E402


class HFUnslothTrainerBackend(BaseTrainerBackend):
    SUPPORTED_PROVIDERS = {"huggingface", "meta", "mistral", "tii", "custom"}
    DEFAULT_IMAGE = "ghcr.io/unslothai/unsloth-trainer:latest"

    def validate_request(self, request: TrainingRequest) -> None:  # noqa: D401
        if request.base_model.provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError("Hugging Face backend is limited to HF-hosted models")
        if request.customization.method not in {"lora", "qlora", "peft"}:
            raise ValueError("Unsloth path targets parameter-efficient tuning methods")
        if request.resources.gpus < 1:
            raise ValueError("At least one GPU required for Unsloth fine-tuning")

    def build_job_spec(self, request: TrainingRequest) -> Dict[str, Any]:
        unsloth_opts = request.extra_parameters.get("unsloth", {})
        return {
            "job_id": request.job_id,
            "backend": "hf-unsloth",
            "container_image": request.extra_parameters.get(
                "container_image", self.DEFAULT_IMAGE
            ),
            "launcher": "accelerate",
            "model": request.base_model.dict(),
            "datasets": [dataset.dict() for dataset in request.datasets],
            "resources": request.resources.dict(),
            "customization": request.customization.dict(),
            "tuning_parameters": request.tuning_parameters.dict(),
            "unsloth": {
                "optimize_long_context": unsloth_opts.get("optimize_long_context", True),
                "quantization": unsloth_opts.get("quantization", "bnb-nf4"),
                "gradient_accumulation": unsloth_opts.get("gradient_accumulation", 4),
            },
            "artifacts": request.artifacts.dict(),
            "callbacks": request.callbacks.dict() if request.callbacks else None,
            "backend_job_id": f"hf-unsloth-{request.job_id}",
        }
