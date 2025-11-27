#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-latest}"
REGISTRY="${CONTAINER_REGISTRY:-docker.io}"
REPO="${CONTAINER_REPO:-${DOCKERHUB_USER:-}}"
SERVICE_DIR="services/training_api"
IMAGE_NAME="training-api"
APP_LABEL="training-api"

if [[ -z "${REPO}" ]]; then
  echo "Set CONTAINER_REPO (or DOCKERHUB_USER) to the registry namespace to push images into." >&2
  exit 1
fi

IMAGE="${REGISTRY}/${REPO}/${IMAGE_NAME}:${IMAGE_TAG}"
MANIFEST_PATH="${SERVICE_DIR}/k8s"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Building ${IMAGE}..."
docker build -t "${IMAGE}" -f "${REPO_ROOT}/${SERVICE_DIR}/Dockerfile" "${REPO_ROOT}"

echo "Pushing ${IMAGE}..."
docker push "${IMAGE}"

echo "Deleting previous Kubernetes objects (if any)..."
kubectl delete -f "${MANIFEST_PATH}" --ignore-not-found

echo "Applying updated manifests..."
kubectl apply -f "${MANIFEST_PATH}"

echo "Waiting for rollout of ${APP_LABEL}..."
kubectl rollout status "deployment/${APP_LABEL}"

echo "Current pods:"
kubectl get pods -l "app=${APP_LABEL}"

echo "Service endpoint:"
kubectl get svc "${APP_LABEL}" -o wide
