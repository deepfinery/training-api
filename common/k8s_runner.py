"""Job runner that translates trainer requests into Kubeflow TrainJob resources."""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

import requests
from kubernetes import client, config
from kubernetes.client import ApiException
from kubernetes.config import ConfigException

from .job_runner import CALLBACK_INTERVAL_SECONDS, JobState, JobStatus, TERMINAL_STATES
from .kube import sanitize_k8s_name

logger = logging.getLogger(__name__)


class KubeflowTrainJobRunner:
    """Submit TrainJobs via the Kubernetes CustomObjects API."""

    def __init__(
        self,
        *,
        namespace: str,
        api_version: str = "trainer.kubeflow.org/v1alpha1",
        plural: str = "trainjobs",
    ) -> None:
        self.namespace = namespace
        self.api_version = api_version
        if "/" not in api_version:
            raise ValueError("api_version must be in the form <group>/<version>")
        self.group, self.version = api_version.split("/", 1)
        self.plural = plural
        self.api_client = self._build_client()
        self.custom_api = client.CustomObjectsApi(self.api_client)
        self._callback_configs: Dict[str, Dict[str, Optional[str]]] = {}
        self._callback_stop_events: Dict[str, threading.Event] = {}

    def _build_client(self) -> client.ApiClient:
        try:
            config.load_incluster_config()
        except ConfigException:
            config.load_kube_config()
        return client.ApiClient()

    def submit(self, job_spec: Dict[str, Any]) -> JobStatus:
        manifest = job_spec.get("train_job")
        if not manifest:
            raise ValueError("Kubeflow runner requires a train_job manifest.")
        metadata = manifest.setdefault("metadata", {})
        if "name" not in metadata:
            metadata["name"] = sanitize_k8s_name(job_spec["job_id"])
        job_name = metadata["name"]
        body = manifest.copy()
        try:
            self.custom_api.create_namespaced_custom_object(
                self.group, self.version, self.namespace, self.plural, body
            )
        except ApiException as exc:
            if exc.status == 409:
                raise ValueError(f"TrainJob {job_name} already exists") from exc
            raise
        status = JobStatus(
            job_id=job_spec["job_id"],
            backend_job_id=job_name,
            state=JobState.SUBMITTED,
            detail="TrainJob created",
            metadata={"train_job": manifest},
        )
        self._register_callback(job_spec, status)
        self._send_callback(status)
        return status

    def get(self, job_id: str) -> JobStatus:
        name = sanitize_k8s_name(job_id)
        try:
            obj = self.custom_api.get_namespaced_custom_object(
                self.group, self.version, self.namespace, self.plural, name
            )
        except ApiException as exc:
            if exc.status == 404:
                raise KeyError(job_id) from exc
            raise
        return self._status_from_trainjob(obj, job_id)

    def cancel(self, job_id: str) -> JobStatus:
        name = sanitize_k8s_name(job_id)
        try:
            self.custom_api.delete_namespaced_custom_object(
                self.group, self.version, self.namespace, self.plural, name
            )
        except ApiException as exc:
            if exc.status == 404:
                raise KeyError(job_id) from exc
            raise
        status = JobStatus(
            job_id=job_id,
            backend_job_id=name,
            state=JobState.CANCELLED,
            detail="TrainJob deletion requested",
        )
        self._send_callback(status)
        self._stop_callback(job_id)
        return status

    def _status_from_trainjob(self, obj: Dict[str, Any], job_id: str) -> JobStatus:
        metadata = obj.get("metadata", {})
        backend_job_id = metadata.get("name", sanitize_k8s_name(job_id))
        conditions = obj.get("status", {}).get("conditions", []) or []
        state = JobState.SUBMITTED
        detail = "TrainJob submitted"
        deletion_timestamp = metadata.get("deletionTimestamp")

        def _condition(condition_type: str) -> Optional[Dict[str, Any]]:
            for cond in conditions:
                if cond.get("type") == condition_type and cond.get("status") == "True":
                    return cond
            return None

        if deletion_timestamp:
            state = JobState.CANCELLED
            detail = "TrainJob is terminating"
        else:
            failed = _condition("Failed")
            if failed:
                state = JobState.FAILED
                detail = failed.get("message") or "TrainJob failed"
            else:
                complete = _condition("Complete")
                if complete:
                    state = JobState.SUCCEEDED
                    detail = complete.get("message") or "TrainJob completed"
                else:
                    suspended = _condition("Suspended")
                    if suspended:
                        state = JobState.CANCELLED
                        detail = suspended.get("message") or "TrainJob is suspended"
                    else:
                        jobs_status = obj.get("status", {}).get("jobsStatus", []) or []
                        active = any(job.get("active", 0) > 0 for job in jobs_status)
                        if active:
                            state = JobState.RUNNING
                            detail = "TrainJob is running"
        return JobStatus(
            job_id=job_id,
            backend_job_id=backend_job_id,
            state=state,
            detail=detail,
            metadata={"train_job": obj},
        )

    # Callback management copied from MockJobRunner to keep consistency
    def _register_callback(self, job_spec: Dict[str, Any], status: JobStatus) -> None:
        callbacks: Dict[str, Any] | None = job_spec.get("callbacks")
        if not callbacks:
            return
        webhook_url = callbacks.get("webhook_url")
        if not webhook_url:
            return
        config = {
            "webhook_url": webhook_url,
            "auth_header": callbacks.get("auth_header"),
            "job_id": status.job_id,
        }
        self._callback_configs[status.job_id] = config
        stop_event = threading.Event()
        self._callback_stop_events[status.job_id] = stop_event
        thread = threading.Thread(
            target=self._callback_loop, args=(status.job_id, stop_event), daemon=True
        )
        thread.start()

    def _callback_loop(self, job_id: str, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            try:
                status = self.get(job_id)
            except KeyError:
                break
            self._send_callback(status)
            if status.state in TERMINAL_STATES:
                break
            stop_event.wait(CALLBACK_INTERVAL_SECONDS)
        self._stop_callback(job_id)

    def _send_callback(self, status: JobStatus) -> None:
        config = self._callback_configs.get(status.job_id)
        if not config:
            return
        headers = {"Content-Type": "application/json"}
        auth_header = config.get("auth_header")
        if auth_header:
            headers["Authorization"] = auth_header
        payload = {
            "job_id": status.job_id,
            "backend_job_id": status.backend_job_id,
            "status": status.state.value,
            "detail": status.detail or f"Job {status.state.value}",
            "timestamp": time.time(),
        }
        try:
            requests.post(config["webhook_url"], json=payload, headers=headers, timeout=5)
        except Exception as exc:  # pragma: no cover
            logger.warning("Callback dispatch failed for %s: %s", status.job_id, exc)

    def _stop_callback(self, job_id: str) -> None:
        stop_event = self._callback_stop_events.pop(job_id, None)
        if stop_event:
            stop_event.set()
        self._callback_configs.pop(job_id, None)
