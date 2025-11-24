"""Nemo/Nemotron specific backend implementation."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from common import BaseTrainerBackend, TrainingRequest  # noqa: E402


class NemoTrainerBackend(BaseTrainerBackend):
    SUPPORTED_PROVIDERS = {"nemo", "nemotron", "bim"}
    DEFAULT_IMAGE = "nvcr.io/nvidia/nemo:24.03"

    def validate_request(self, request: TrainingRequest) -> None:  # noqa: D401
        if request.base_model.provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Nemo backend only supports providers {self.SUPPORTED_PROVIDERS}, "
                f"got {request.base_model.provider}"
            )
        if request.resources.gpus < 1:
            raise ValueError("Nemo fine-tuning requires at least one GPU")
        if request.customization.method == "full" and request.resources.gpus < 4:
            raise ValueError("Full fine-tuning typically needs >=4 GPUs for stability")

    def build_job_spec(self, request: TrainingRequest) -> Dict[str, Any]:
        return {
            "job_id": request.job_id,
            "backend": "nemo",
            "container_image": request.extra_parameters.get(
                "container_image", self.DEFAULT_IMAGE
            ),
            "launcher": "nemo-launcher",
            "model": request.base_model.dict(),
            "datasets": [dataset.dict() for dataset in request.datasets],
            "resources": request.resources.dict(),
            "artifacts": request.artifacts.dict(),
            "customization": request.customization.dict(),
            "tuning_parameters": request.tuning_parameters.dict(),
            "callbacks": request.callbacks.dict() if request.callbacks else None,
            "backend_job_id": f"nemo-{request.job_id}",
        }
