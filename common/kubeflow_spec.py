"""Helpers for building Kubeflow TrainJob manifests from training requests."""
from __future__ import annotations

import json
import shlex
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional

from .kube import sanitize_k8s_name
from .schemas import TrainingRequest

DEFAULT_API_VERSION = "trainer.kubeflow.org/v1alpha1"
BACKEND_LABEL = "trainer.deepfinery/backend"
JOB_ID_LABEL = "trainer.deepfinery/job-id"
REQUEST_ANNOTATION = "trainer.deepfinery/request-json"


def _ensure_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, Iterable):
        result: List[str] = []
        for item in value:
            if item is None:
                continue
            result.append(str(item))
        return result or None
    raise TypeError("Expected string or iterable for trainer command/args.")


def _normalize_env(env: Any) -> List[Dict[str, str]]:
    if not env:
        return []
    normalized: List[Dict[str, str]] = []
    if isinstance(env, dict):
        normalized = [{"name": str(k), "value": str(v)} for k, v in env.items()]
    elif isinstance(env, list):
        for item in env:
            if isinstance(item, dict) and "name" in item:
                normalized.append(
                    {"name": str(item["name"]), "value": str(item.get("value", ""))}
                )
    else:
        raise TypeError("Environment overrides must be a mapping or list of mappings.")
    return normalized


def _merge_env(
    base: List[Dict[str, str]], extra: Optional[List[Dict[str, str]]]
) -> List[Dict[str, str]]:
    ordered: OrderedDict[str, str] = OrderedDict()
    for item in base:
        ordered[item["name"]] = item["value"]
    if extra:
        for item in extra:
            ordered[item["name"]] = item["value"]
    return [{"name": key, "value": value} for key, value in ordered.items()]


def _format_request_env(request: TrainingRequest, backend: str) -> List[Dict[str, str]]:
    payload = request.dict(exclude_none=True)
    env: List[Dict[str, str]] = [
        {"name": "TRAINING_JOB_ID", "value": request.job_id},
        {"name": "TRAINING_BACKEND", "value": backend},
        {"name": "TRAINING_REQUEST_JSON", "value": json.dumps(payload)},
        {"name": "TRAINING_RESOURCES_JSON", "value": json.dumps(request.resources.dict())},
        {
            "name": "TRAINING_CUSTOMIZATION_JSON",
            "value": json.dumps(request.customization.dict()),
        },
        {
            "name": "TRAINING_TUNING_PARAMETERS_JSON",
            "value": json.dumps(request.tuning_parameters.dict()),
        },
    ]
    if request.datasets:
        env.append(
            {
                "name": "TRAINING_DATASETS_JSON",
                "value": json.dumps([dataset.dict() for dataset in request.datasets]),
            }
        )
    if request.artifacts:
        env.append({"name": "TRAINING_LOG_URI", "value": request.artifacts.log_uri})
        env.append({"name": "TRAINING_OUTPUT_URI", "value": request.artifacts.output_uri})
        if request.artifacts.status_stream_url:
            env.append(
                {
                    "name": "TRAINING_STATUS_STREAM_URL",
                    "value": request.artifacts.status_stream_url,
                }
            )
    env.append({"name": "TRAINING_BASE_MODEL_JSON", "value": json.dumps(request.base_model.dict())})
    return env


def build_resource_requirements(
    cpu_cores: int, memory_gb: int, gpus: int
) -> Dict[str, Dict[str, str]]:
    requests: Dict[str, str] = {
        "cpu": str(cpu_cores),
        "memory": f"{memory_gb}Gi",
    }
    if gpus > 0:
        requests["nvidia.com/gpu"] = str(gpus)
    limits = requests.copy()
    return {"requests": requests, "limits": limits}


def build_train_job_manifest(
    request: TrainingRequest,
    *,
    backend: str,
    runtime_name: str,
    runtime_kind: str,
    runtime_api_group: str,
    trainer_image: str,
    trainer_command: Any = None,
    trainer_args: Any = None,
    trainer_env: Any = None,
    num_nodes: int | None = None,
    num_proc_per_node: Any = None,
    resources_per_node: Dict[str, Any],
    dataset_uri: Optional[str] = None,
    model_uri: Optional[str] = None,
    labels: Optional[Dict[str, str]] = None,
    annotations: Optional[Dict[str, str]] = None,
    api_version: str = DEFAULT_API_VERSION,
) -> Dict[str, Any]:
    job_name = sanitize_k8s_name(request.job_id, prefix=f"{backend}-job")
    metadata_labels = {
        BACKEND_LABEL: backend,
        JOB_ID_LABEL: request.job_id,
    }
    if labels:
        metadata_labels.update({str(k): str(v) for k, v in labels.items()})
    metadata_annotations = {
        REQUEST_ANNOTATION: json.dumps(request.dict(exclude_none=True)),
        JOB_ID_LABEL: request.job_id,
    }
    if annotations:
        metadata_annotations.update({str(k): str(v) for k, v in annotations.items()})

    trainer_spec: Dict[str, Any] = {"image": trainer_image, "resourcesPerNode": resources_per_node}
    command_list = _ensure_list(trainer_command)
    if command_list:
        trainer_spec["command"] = command_list
    args_list = _ensure_list(trainer_args)
    if args_list:
        trainer_spec["args"] = args_list
    env_values = _format_request_env(request, backend)
    extra_env = _normalize_env(trainer_env)
    trainer_spec["env"] = _merge_env(env_values, extra_env)
    if num_nodes:
        trainer_spec["numNodes"] = num_nodes
    if num_proc_per_node is not None:
        trainer_spec["numProcPerNode"] = num_proc_per_node

    runtime_ref = {"name": runtime_name, "kind": runtime_kind, "apiGroup": runtime_api_group}
    manifest: Dict[str, Any] = {
        "apiVersion": api_version,
        "kind": "TrainJob",
        "metadata": {"name": job_name, "labels": metadata_labels, "annotations": metadata_annotations},
        "spec": {"runtimeRef": runtime_ref, "trainer": trainer_spec, "labels": metadata_labels},
    }
    initializer: Dict[str, Any] = {}
    if dataset_uri:
        initializer["dataset"] = {"storageUri": dataset_uri}
    if model_uri:
        initializer.setdefault("model", {"storageUri": model_uri})["storageUri"] = model_uri
    if initializer:
        manifest["spec"]["initializer"] = initializer
    return manifest
