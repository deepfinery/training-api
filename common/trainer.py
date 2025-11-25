"""Base helper for building trainer backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from .job_runner import JobStatus, MockJobRunner
from .schemas import TrainingRequest, TrainingResponse


class BaseTrainerBackend(ABC):
    """Shared validation + job submission pipeline."""

    def __init__(self, job_runner: MockJobRunner | None = None) -> None:
        self.job_runner = job_runner or MockJobRunner()

    def schedule_training(self, request: TrainingRequest) -> TrainingResponse:
        self.validate_request(request)
        job_spec = self.build_job_spec(request)
        status = self.job_runner.submit(job_spec)
        return self._status_to_response(status)

    def get_status(self, job_id: str) -> TrainingResponse:
        status = self.job_runner.get(job_id)
        return self._status_to_response(status)

    def cancel_training(self, job_id: str) -> TrainingResponse:
        status = self.job_runner.cancel(job_id)
        return self._status_to_response(status)

    @abstractmethod
    def validate_request(self, request: TrainingRequest) -> None:
        """Raise a ValueError if the payload is incompatible with this backend."""

    @abstractmethod
    def build_job_spec(self, request: TrainingRequest) -> Dict[str, Any]:
        """Translate the request into whatever the job runner understands."""

    def _status_to_response(self, status: JobStatus) -> TrainingResponse:
        return TrainingResponse(
            job_id=status.job_id,
            backend_job_id=status.backend_job_id,
            status=status.state.value,
            detail=status.detail or f"Job {status.state.value}",
            metadata=status.metadata,
        )
