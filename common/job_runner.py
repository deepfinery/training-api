"""Mock job runner that simulates scheduling and status updates."""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import requests

CALLBACK_INTERVAL_SECONDS = int(os.getenv("TRAINER_CALLBACK_INTERVAL_SECONDS", "60"))

logger = logging.getLogger(__name__)


class JobState(str, Enum):
    SUBMITTED = "submitted"
    RUNNING = "running"
    FAILED = "failed"
    SUCCEEDED = "succeeded"
    CANCELLED = "cancelled"


TERMINAL_STATES = {JobState.FAILED, JobState.SUCCEEDED, JobState.CANCELLED}


@dataclass
class JobStatus:
    job_id: str
    backend_job_id: str
    state: JobState
    submitted_at: float = field(default_factory=time.time)
    detail: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MockJobRunner:
    """Naive in-memory job runner useful for development and unit tests."""

    def __init__(self) -> None:
        self._jobs: Dict[str, JobStatus] = {}
        self._callback_stop_events: Dict[str, threading.Event] = {}
        self._callback_configs: Dict[str, Dict[str, Optional[str]]] = {}

    def submit(self, job_spec: Dict[str, Any]) -> JobStatus:
        backend_job_id = job_spec.get("backend_job_id") or str(uuid.uuid4())
        state = JobState.SUBMITTED
        status = JobStatus(
            job_id=job_spec["job_id"],
            backend_job_id=backend_job_id,
            state=state,
            detail="Job accepted for processing",
            metadata={"job_spec": job_spec},
        )
        self._jobs[job_spec["job_id"]] = status
        self._register_callback(job_spec, status)
        return status

    def get(self, job_id: str) -> JobStatus:
        if job_id not in self._jobs:
            raise KeyError(job_id)
        return self._jobs[job_id]

    def cancel(self, job_id: str) -> JobStatus:
        status = self.get(job_id)
        if status.state in {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}:
            status.detail = f"Job already {status.state.value}"
            self._send_callback(status)
            self._stop_callback(job_id)
            return status
        status.state = JobState.CANCELLED
        status.detail = "Job cancellation requested"
        self._send_callback(status)
        self._stop_callback(job_id)
        return status

    # Callback helpers

    def _register_callback(self, job_spec: Dict[str, Any], status: JobStatus) -> None:
        callbacks: Dict[str, Any] | None = job_spec.get("callbacks")
        if not callbacks:
            return
        webhook_url: Optional[str] = callbacks.get("webhook_url")
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
            target=self._callback_loop,
            args=(status.job_id, stop_event),
            daemon=True,
        )
        thread.start()

    def _callback_loop(self, job_id: str, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            status = self.get(job_id)
            if status.state == JobState.SUBMITTED:
                status.state = JobState.RUNNING
                status.detail = "Job is running"
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
        except Exception as exc:  # pragma: no cover - network errors OK during tests
            logger.warning("Callback dispatch failed for %s: %s", status.job_id, exc)

    def _stop_callback(self, job_id: str) -> None:
        stop_event = self._callback_stop_events.pop(job_id, None)
        if stop_event:
            stop_event.set()
        self._callback_configs.pop(job_id, None)
