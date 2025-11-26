"""Utility helpers for interacting with Kubernetes resources."""
from __future__ import annotations

import os
import re
from pathlib import Path

DNS_1123_LABEL_MAX_LENGTH = 63
_INVALID_CHAR_PATTERN = re.compile(r"[^a-z0-9-]+")
_LEADING_TRAILING_DASH = re.compile(r"^-+|-+$")
_STARTS_WITH_ALPHA = re.compile(r"^[a-z]")
_NAMESPACE_FILE = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")


def sanitize_k8s_name(value: str, prefix: str = "job") -> str:
    """Convert arbitrary identifiers into RFC 1123 compliant names."""
    slug = _INVALID_CHAR_PATTERN.sub("-", value.lower())
    slug = _LEADING_TRAILING_DASH.sub("", slug)
    if not slug:
        slug = prefix
    if not _STARTS_WITH_ALPHA.match(slug):
        slug = f"{prefix}-{slug}"
    if len(slug) > DNS_1123_LABEL_MAX_LENGTH:
        slug = slug[:DNS_1123_LABEL_MAX_LENGTH].rstrip("-")
    if not slug:
        slug = prefix
    return slug


def detect_namespace(env_var: str = "TRAINER_JOBS_NAMESPACE") -> str:
    """Return the namespace to use for creating TrainJobs."""
    override = os.getenv(env_var)
    if override:
        return override
    if _NAMESPACE_FILE.exists():
        try:
            return _NAMESPACE_FILE.read_text(encoding="utf-8").strip()
        except OSError:
            pass
    return "default"
