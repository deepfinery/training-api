"""Mock job runner that simulates scheduling and status updates."""
from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class JobState(str, Enum):
    SUBMITTED = "submitted"
    RUNNING = "running"
    FAILED = "failed"
    SUCCEEDED = "succeeded"


@dataclass
class JobStatus:
    job_id: str
    backend_job_id: str
    state: JobState
    submitted_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MockJobRunner:
    """Naive in-memory job runner useful for development and unit tests."""

    def __init__(self) -> None:
        self._jobs: Dict[str, JobStatus] = {}

    def submit(self, job_spec: Dict[str, Any]) -> JobStatus:
        backend_job_id = job_spec.get("backend_job_id") or str(uuid.uuid4())
        state = random.choice([JobState.SUBMITTED, JobState.RUNNING])
        status = JobStatus(
            job_id=job_spec["job_id"],
            backend_job_id=backend_job_id,
            state=state,
            metadata={"job_spec": job_spec},
        )
        self._jobs[job_spec["job_id"]] = status
        return status

    def get(self, job_id: str) -> JobStatus:
        return self._jobs[job_id]
