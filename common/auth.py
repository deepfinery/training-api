"""Simple token-based auth utilities shared by the trainer APIs."""
from __future__ import annotations

import os
from typing import Optional

from fastapi import Header, HTTPException, status

API_TOKEN_ENV_VAR = "TRAINER_API_TOKEN"


def _load_expected_token() -> Optional[str]:
    """Read the expected token from the environment."""
    return os.environ.get(API_TOKEN_ENV_VAR)


def require_token(x_trainer_token: Optional[str] = Header(None, alias="X-TRAINER-TOKEN")) -> None:
    """FastAPI dependency that blocks requests without the configured token."""
    expected = _load_expected_token()
    if not expected:
        # Token protection disabled when env var not set.
        return

    if not x_trainer_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-TRAINER-TOKEN header",
        )

    if x_trainer_token != expected:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid trainer API token",
        )
