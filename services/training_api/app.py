"""Unified Training API entrypoint."""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, status

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from common.auth import require_token  # noqa: E402
from common.unified_schemas import TrainingJobRequest, TrainingJobStatus  # noqa: E402
from services.training_api.backend import PyTorchJobBackend  # noqa: E402

app = FastAPI(title="Unified Trainer API", version="0.2.0")
backend = PyTorchJobBackend()


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/train", response_model=TrainingJobStatus, status_code=status.HTTP_201_CREATED)
def schedule_training(
    request: TrainingJobRequest,
    _: None = Depends(require_token),
) -> TrainingJobStatus:
    try:
        return backend.submit_training_job(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/train/{job_name}", response_model=TrainingJobStatus)
def get_training_status(
    job_name: str,
    _: None = Depends(require_token),
) -> TrainingJobStatus:
    try:
        return backend.get_training_job(job_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_name} not found") from exc
