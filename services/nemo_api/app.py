"""FastAPI application for the Nemo Customizer API."""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from common import TrainingRequest, TrainingResponse  # noqa: E402
from common.auth import require_token  # noqa: E402
from services.nemo_api.backend import NemoTrainerBackend  # noqa: E402

app = FastAPI(title="Nemo Trainer API", version="0.1.0")
backend = NemoTrainerBackend()


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/train", response_model=TrainingResponse)
def schedule_training(
    request: TrainingRequest,
    _: None = Depends(require_token),
) -> TrainingResponse:
    try:
        return backend.schedule_training(request)
    except ValueError as exc:  # pragma: no cover - FastAPI already tested
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/train/{job_id}", response_model=TrainingResponse)
def get_training_status(job_id: str, _: None = Depends(require_token)) -> TrainingResponse:
    try:
        return backend.get_status(job_id)
    except KeyError as exc:  # pragma: no cover
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found") from exc


@app.post("/train/{job_id}/cancel", response_model=TrainingResponse)
def cancel_training(job_id: str, _: None = Depends(require_token)) -> TrainingResponse:
    try:
        return backend.cancel_training(job_id)
    except KeyError as exc:  # pragma: no cover
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found") from exc
