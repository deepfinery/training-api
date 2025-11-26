"""Factory helpers for selecting the appropriate job runner."""
from __future__ import annotations

import os

from .job_runner import MockJobRunner
from .kube import detect_namespace

DEFAULT_API_VERSION = "trainer.kubeflow.org/v1alpha1"


def build_job_runner() -> MockJobRunner:
    """Return the configured job runner implementation."""
    backend = os.getenv("TRAINER_JOB_RUNNER", "mock").lower()
    if backend in {"kubernetes", "k8s"}:
        from .k8s_runner import KubeflowTrainJobRunner

        namespace = detect_namespace()
        api_version = os.getenv("TRAINER_TRAINJOB_API_VERSION", DEFAULT_API_VERSION)
        resource_plural = os.getenv("TRAINER_TRAINJOB_PLURAL", "trainjobs")
        return KubeflowTrainJobRunner(
            namespace=namespace, api_version=api_version, plural=resource_plural
        )
    return MockJobRunner()
