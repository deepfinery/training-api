"""Unified training API backend that submits PyTorchJobs."""
from __future__ import annotations

import logging
from typing import Optional

from kubernetes import client, config
from kubernetes.client import ApiException
from kubernetes.config import ConfigException

from common.kube import detect_namespace
from common.pytorch_job import build_pytorchjob_manifest
from common.unified_schemas import Framework, TrainingJobRequest, TrainingJobStatus

logger = logging.getLogger(__name__)


class PyTorchJobBackend:
    """Submit PyTorchJobs and report their status."""

    def __init__(self, namespace: str | None = None) -> None:
        self.namespace = namespace or detect_namespace()
        self.api_client = self._build_client()
        self.custom_api = client.CustomObjectsApi(self.api_client)

    def _build_client(self) -> client.ApiClient:
        try:
            config.load_incluster_config()
        except ConfigException:
            config.load_kube_config()
        return client.ApiClient()

    def submit_training_job(self, request: TrainingJobRequest) -> TrainingJobStatus:
        manifest = build_pytorchjob_manifest(request)
        namespace = request.namespace or self.namespace
        metadata = manifest.setdefault("metadata", {})
        metadata.setdefault("namespace", namespace)
        name = metadata["name"]
        body = manifest
        try:
            self.custom_api.create_namespaced_custom_object(
                group="kubeflow.org",
                version="v1",
                namespace=namespace,
                plural="pytorchjobs",
                body=body,
            )
        except ApiException as exc:
            if exc.status == 409:
                raise ValueError(f"PyTorchJob {name} already exists") from exc
            logger.exception("Failed to submit PyTorchJob %s", name)
            raise
        return TrainingJobStatus(
            name=name,
            status="submitted",
            detail="PyTorchJob created",
            framework=request.framework,
            run_id=request.run_id,
        )

    def get_training_job(self, name: str, namespace: Optional[str] = None) -> TrainingJobStatus:
        ns = namespace or self.namespace
        try:
            obj = self.custom_api.get_namespaced_custom_object(
                group="kubeflow.org",
                version="v1",
                namespace=ns,
                plural="pytorchjobs",
                name=name,
            )
        except ApiException as exc:
            if exc.status == 404:
                raise KeyError(name) from exc
            logger.exception("Failed to read PyTorchJob %s", name)
            raise
        return self._status_from_pytorchjob(obj)

    def _status_from_pytorchjob(self, obj: dict) -> TrainingJobStatus:
        metadata = obj.get("metadata", {})
        labels = metadata.get("labels", {})
        run_id = labels.get("trainer.deepfinery/run-id", metadata.get("name", ""))
        framework_raw = labels.get("trainer.deepfinery/framework", Framework.HF.value)
        try:
            framework = Framework(framework_raw)
        except ValueError:
            framework = Framework.HF
        status_block = obj.get("status", {})
        conditions = status_block.get("conditions", []) or []
        status = "submitted"
        detail = "PyTorchJob submitted"

        def _pick(condition_type: str) -> Optional[dict]:
            for cond in conditions:
                if cond.get("type") == condition_type and cond.get("status") == "True":
                    return cond
            return None

        failed = _pick("Failed")
        succeeded = _pick("Succeeded")
        running = _pick("Running")
        created = _pick("Created")

        if failed:
            status = "failed"
            detail = failed.get("message") or "PyTorchJob failed"
        elif succeeded:
            status = "succeeded"
            detail = succeeded.get("message") or "PyTorchJob completed"
        elif running:
            status = "running"
            detail = running.get("message") or "PyTorchJob running"
        elif created:
            status = "submitted"
            detail = created.get("message") or detail
        elif status_block.get("active", 0) > 0:
            status = "running"
            detail = "PyTorchJob pods active"

        return TrainingJobStatus(
            name=metadata.get("name", ""),
            status=status,
            detail=detail,
            framework=framework,
            run_id=run_id,
        )


def submit_training_job(request: TrainingJobRequest) -> TrainingJobStatus:
    """Helper that mirrors the simplified contract from the requirements."""

    backend = PyTorchJobBackend()
    return backend.submit_training_job(request)
