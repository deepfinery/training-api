#!/usr/bin/env bash
set -euo pipefail

SERVICE="${1:-nemo}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
REGISTRY="${CONTAINER_REGISTRY:-docker.io}"
REPO="${CONTAINER_REPO:-${DOCKERHUB_USER:-}}"

if [[ -z "${REPO}" ]]; then
  echo "Set CONTAINER_REPO (or DOCKERHUB_USER) to the registry namespace to push images into." >&2
  exit 1
fi

case "${SERVICE}" in
  nemo)
    SERVICE_DIR="services/nemo_api"
    IMAGE_NAME="nemo-trainer"
    APP_LABEL="nemo-trainer"
    ;;
  meta)
    SERVICE_DIR="services/meta_api"
    IMAGE_NAME="meta-trainer"
    APP_LABEL="meta-trainer"
    ;;
  hf|hf_unsloth|hf-unsloth)
    SERVICE="hf_unsloth"
    SERVICE_DIR="services/hf_unsloth_api"
    IMAGE_NAME="hf-unsloth-trainer"
    APP_LABEL="hf-unsloth-trainer"
    ;;
  *)
    echo "Unsupported service '${SERVICE}'. Use one of: nemo, meta, hf_unsloth." >&2
    exit 1
    ;;
esac

IMAGE="${REGISTRY}/${REPO}/${IMAGE_NAME}:${IMAGE_TAG}"
MANIFEST_PATH="${SERVICE_DIR}/k8s"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Building ${IMAGE} from ${SERVICE_DIR}..."
docker build -t "${IMAGE}" -f "${REPO_ROOT}/${SERVICE_DIR}/Dockerfile" "${REPO_ROOT}"

echo "Pushing ${IMAGE}..."
docker push "${IMAGE}"

echo "Deleting previous Kubernetes objects (if any)..."
kubectl delete -f "${MANIFEST_PATH}" --ignore-not-found

echo "Applying updated manifests..."
kubectl apply -f "${MANIFEST_PATH}"

echo "Waiting for rollout of ${IMAGE_NAME}..."
kubectl rollout status "deployment/${APP_LABEL}"

echo "Current pods:"
kubectl get pods -l "app=${APP_LABEL}"

echo "Service endpoint:"
kubectl get svc "${APP_LABEL}" -o wide
