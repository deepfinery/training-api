"""FastAPI app for Meta adapter trainer API."""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from common import TrainingRequest, TrainingResponse  # noqa: E402
from common.auth import require_token  # noqa: E402
from services.meta_api.backend import MetaTrainerBackend  # noqa: E402

app = FastAPI(title="Meta Trainer API", version="0.1.0")
backend = MetaTrainerBackend()


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/train", response_model=TrainingResponse)
def train(
    request: TrainingRequest,
    _: None = Depends(require_token),
) -> TrainingResponse:
    try:
        return backend.schedule_training(request)
    except ValueError as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=str(exc)) from exc
