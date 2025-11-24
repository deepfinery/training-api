"""Meta based tuning backend that can proxy to Nemo-style runners."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from common import BaseTrainerBackend, TrainingRequest  # noqa: E402


class MetaTrainerBackend(BaseTrainerBackend):
    SUPPORTED_PROVIDERS = {"meta", "huggingface"}
    DEFAULT_IMAGE = "ghcr.io/meta-llm/meta-trainer:latest"

    def validate_request(self, request: TrainingRequest) -> None:  # noqa: D401
        if request.base_model.provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                "Meta backend expects Meta released models or Hugging Face mirrors"
            )
        if request.resources.gpus < 2:
            raise ValueError("Meta adapters need at least 2 GPUs for ZeRO stages")
        if request.customization.method == "full" and request.resources.memory_gb < 128:
            raise ValueError("Full Meta fine-tuning typically needs >=128GB host memory")

    def build_job_spec(self, request: TrainingRequest) -> Dict[str, Any]:
        executor = request.extra_parameters.get("executor", "torchrun")
        return {
            "job_id": request.job_id,
            "backend": "meta-wrapper",
            "container_image": request.extra_parameters.get(
                "container_image", self.DEFAULT_IMAGE
            ),
            "launcher": executor,
            "entrypoint": "meta_trainer.launch:main",
            "model": request.base_model.dict(),
            "datasets": [dataset.dict() for dataset in request.datasets],
            "resources": request.resources.dict(),
            "customization": request.customization.dict(),
            "tuning_parameters": request.tuning_parameters.dict(),
            "artifacts": request.artifacts.dict(),
            "callbacks": request.callbacks.dict() if request.callbacks else None,
            "backend_job_id": f"meta-{request.job_id}",
        }
