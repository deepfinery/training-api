"""Meta based tuning backend that can proxy to Nemo-style runners."""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from common import BaseTrainerBackend, TrainingRequest  # noqa: E402
from common.kubeflow_spec import (  # noqa: E402
    DEFAULT_API_VERSION,
    build_resource_requirements,
    build_train_job_manifest,
)


class MetaTrainerBackend(BaseTrainerBackend):
    SUPPORTED_PROVIDERS = {"meta", "huggingface"}
    DEFAULT_IMAGE = "ghcr.io/meta-llm/meta-trainer:latest"

    def __init__(self, job_runner: Any | None = None) -> None:
        super().__init__(job_runner)
        self.runtime_name = os.getenv("TRAINER_RUNTIME_NAME", "meta-runtime")
        self.runtime_kind = os.getenv("TRAINER_RUNTIME_KIND", "ClusterTrainingRuntime")
        self.runtime_api_group = os.getenv("TRAINER_RUNTIME_API_GROUP", "trainer.kubeflow.org")
        self.trainjob_api_version = os.getenv("TRAINER_TRAINJOB_API_VERSION", DEFAULT_API_VERSION)

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
        runtime_name, runtime_kind, runtime_api_group = self._resolve_runtime_ref(request)
        container_image = request.extra_parameters.get("container_image", self.DEFAULT_IMAGE)
        num_nodes = max(1, int(request.extra_parameters.get("num_nodes", 1)))
        num_proc = request.extra_parameters.get("num_proc_per_node")
        if num_proc is None and request.resources.gpus > 0:
            num_proc = max(1, math.ceil(request.resources.gpus / num_nodes))
        executor = request.extra_parameters.get("executor", "torchrun")
        resources_override = request.extra_parameters.get("resources_per_node")
        if isinstance(resources_override, dict):
            resources_per_node = resources_override
        else:
            resources_per_node = build_resource_requirements(
                request.resources.cpus, request.resources.memory_gb, request.resources.gpus
            )
        dataset_uri = request.datasets[0].source if request.datasets else None
        manifest = build_train_job_manifest(
            request,
            backend="meta",
            runtime_name=runtime_name,
            runtime_kind=runtime_kind,
            runtime_api_group=runtime_api_group,
            trainer_image=container_image,
            trainer_command=request.extra_parameters.get("trainer_command")
            or request.extra_parameters.get("command")
            or executor,
            trainer_args=request.extra_parameters.get("trainer_args"),
            trainer_env=request.extra_parameters.get("trainer_env"),
            num_nodes=num_nodes,
            num_proc_per_node=num_proc,
            resources_per_node=resources_per_node,
            dataset_uri=dataset_uri,
            model_uri=request.base_model.weights_url,
            labels=request.extra_parameters.get("job_labels"),
            annotations=request.extra_parameters.get("job_annotations"),
            api_version=self.trainjob_api_version,
        )
        return {
            "job_id": request.job_id,
            "backend": "meta",
            "backend_job_id": manifest["metadata"]["name"],
            "train_job": manifest,
            "callbacks": request.callbacks.dict(exclude_none=True) if request.callbacks else None,
        }

    def _resolve_runtime_ref(self, request: TrainingRequest) -> tuple[str, str, str]:
        override = request.extra_parameters.get("runtime_ref") or {}
        return (
            override.get("name") or self.runtime_name,
            override.get("kind") or self.runtime_kind,
            override.get("apiGroup") or self.runtime_api_group,
        )
