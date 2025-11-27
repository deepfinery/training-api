"""Framework-specific trainer registry."""
from __future__ import annotations

from importlib import import_module
from typing import Callable, Dict

FrameworkFn = Callable[..., None]

MODULES: Dict[str, str] = {
    "hf": "frameworks.hf_runner",
    "huggingface": "frameworks.hf_runner",
    "meta": "frameworks.meta_runner",
    "nemo": "frameworks.nemo_runner",
}


def get_runner(framework: str) -> FrameworkFn:
    if framework not in MODULES:
        raise ValueError(f"Unsupported framework '{framework}'")
    module = import_module(MODULES[framework])
    return getattr(module, "run")
