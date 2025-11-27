"""Helpers for rendering PyTorchJob manifests."""
from __future__ import annotations

import json
import shlex
from typing import Any, Dict, List

from .kube import sanitize_k8s_name
from .unified_schemas import TrainingJobRequest


def _baseline_env(request: TrainingJobRequest) -> List[Dict[str, str]]:
    env: List[Dict[str, str]] = [
        {"name": "FRAMEWORK", "value": request.framework.value},
        {"name": "MODEL_ID", "value": request.model_id},
        {"name": "RUN_ID", "value": request.run_id},
        {"name": "NUM_NODES", "value": str(request.num_nodes)},
        {"name": "GPUS_PER_NODE", "value": str(request.gpus_per_node)},
        {"name": "CHECKPOINT_BASE_URI", "value": request.checkpoint_base_uri},
        {"name": "CHECKPOINT_PREFIX", "value": request.checkpoint_prefix},
        {"name": "CHECKPOINTS_URI", "value": request.checkpoints_uri},
        {"name": "LOGS_URI", "value": request.logs_uri},
        {"name": "CONFIG_URI", "value": request.config_uri},
        {"name": "DATASET_URI", "value": request.dataset_uri},
        {"name": "TRAINING_EXTRA_ARGS_JSON", "value": json.dumps(request.extra_args)},
    ]
    if request.config_path:
        env.append({"name": "CONFIG_PATH", "value": request.config_path})
    if request.resume_from_checkpoint:
        env.append({"name": "RESUME_FROM_CHECKPOINT", "value": "1"})
    for key, value in request.env.items():
        env.append({"name": str(key), "value": str(value)})
    return env


def _cli_from_extra_args(extra_args: Dict[str, Any]) -> List[str]:
    cli: List[str] = []
    for key, value in extra_args.items():
        flag = f"--{str(key).replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cli.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                cli.extend([flag, shlex.quote(str(item))])
            continue
        cli.extend([flag, shlex.quote(str(value))])
    return cli


def _torchrun_launch(request: TrainingJobRequest) -> str:
    cli: List[str] = [
        "torchrun",
        f"--nproc_per_node=$GPUS_PER_NODE",
        f"--nnodes=$NUM_NODES",
        "--node_rank=$RANK",
        "--master_addr=$MASTER_ADDR",
        "--master_port=$MASTER_PORT",
        "train.py",
        "--framework ${FRAMEWORK}",
        "--run-id ${RUN_ID}",
        "--checkpoint-base-uri ${CHECKPOINT_BASE_URI}",
        "--checkpoint-prefix ${CHECKPOINT_PREFIX}",
        "--dataset-uri ${DATASET_URI}",
        "--logs-uri ${LOGS_URI}",
        "--config-uri ${CONFIG_URI}",
        "--model-id ${MODEL_ID}",
    ]
    if request.config_path:
        cli.extend(["--config", "${CONFIG_PATH}"])
    if request.resume_from_checkpoint:
        cli.append("--resume")
    cli.extend(_cli_from_extra_args(request.extra_args))
    return " ".join(cli)


def _container_spec(request: TrainingJobRequest) -> Dict[str, Any]:
    limits = {
        "cpu": str(request.cpus_per_node),
        "memory": f"{request.memory_per_node_gb}Gi",
        "nvidia.com/gpu": str(request.gpus_per_node),
    }
    command = ["/bin/bash", "-lc"]
    args = [_torchrun_launch(request)]
    return {
        "name": "trainer",
        "image": request.image,
        "imagePullPolicy": "IfNotPresent",
        "command": command,
        "args": args,
        "env": _baseline_env(request),
        "resources": {"limits": limits, "requests": limits},
    }


def _replica_spec(request: TrainingJobRequest) -> Dict[str, Any]:
    template = {
        "spec": {
            "restartPolicy": "OnFailure",
            "containers": [_container_spec(request)],
        }
    }
    return {"replicas": 1, "restartPolicy": "OnFailure", "template": template}


def build_pytorchjob_manifest(request: TrainingJobRequest) -> Dict[str, Any]:
    """Render a PyTorchJob manifest for the provided request."""

    name = request.job_name or sanitize_k8s_name(f"{request.framework.value}-{request.run_id}")
    metadata_labels = request.labels.copy()
    metadata_labels.setdefault("trainer.deepfinery/framework", request.framework.value)
    metadata_labels.setdefault("trainer.deepfinery/run-id", request.run_id)
    metadata: Dict[str, Any] = {"name": name, "labels": metadata_labels}
    if request.annotations:
        metadata["annotations"] = request.annotations.copy()
    if request.namespace:
        metadata["namespace"] = request.namespace

    container_request = _replica_spec(request)
    worker_spec = None
    if request.num_nodes > 1:
        worker_spec = _replica_spec(request)
        worker_spec["replicas"] = request.num_nodes - 1

    pytorch_replicas = {
        "Master": container_request,
    }
    if worker_spec:
        pytorch_replicas["Worker"] = worker_spec

    manifest: Dict[str, Any] = {
        "apiVersion": "kubeflow.org/v1",
        "kind": "PyTorchJob",
        "metadata": metadata,
        "spec": {
            "pytorchReplicaSpecs": pytorch_replicas,
        },
    }
    return manifest
