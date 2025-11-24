"""FastAPI app exposing Hugging Face + Unsloth training."""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from common import TrainingRequest, TrainingResponse  # noqa: E402
from services.hf_unsloth_api.backend import HFUnslothTrainerBackend  # noqa: E402

app = FastAPI(title="HF Unsloth Trainer API", version="0.1.0")
backend = HFUnslothTrainerBackend()


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/train", response_model=TrainingResponse)
def schedule(request: TrainingRequest) -> TrainingResponse:
    try:
        return backend.schedule_training(request)
    except ValueError as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=str(exc)) from exc
