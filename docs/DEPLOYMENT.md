# Deployment

## Container Image

Build the unified API image from the repo root (Docker context must include `common/`, `services/training_api/`, and `train.py`).

```bash
export CONTAINER_REPO="docker.io/<org>"
export IMAGE_TAG="1.0"
docker build -t "$CONTAINER_REPO/training-api:$IMAGE_TAG" -f services/training_api/Dockerfile .
docker push "$CONTAINER_REPO/training-api:$IMAGE_TAG"
```

## Kubernetes Manifests

`services/training_api/k8s` contains:

- `deployment.yaml` – FastAPI Deployment + ConfigMap + RBAC + ServiceAccount.
- `service.yaml` – LoadBalancer Service.

1. Update `image:` in the Deployment.
2. Set `TRAINER_API_TOKEN` (inline or via Secret) and any registry credentials.
3. Apply:

```bash
kubectl apply -f services/training_api/k8s/
```

### RBAC

The Training API talks directly to the `kubeflow.org/v1` `pytorchjobs` resource, so the Role grants: `get`, `list`, `watch`, `create`, and `delete`. Scope it per-namespace or swap to a ClusterRole if needed.

### Environment Variables

- `TRAINER_API_TOKEN` – shared secret required by `X-TRAINER-TOKEN` header.
- `TRAINER_LOG_LEVEL` (optional) – forwarded to `train.py` logging configuration.

### Troubleshooting

- `kubectl logs deployment/training-api` – inspect API logs.
- `kubectl get pytorchjobs` – ensure resources are created.
- `kubectl describe pytorchjob <name>` – view Training Operator conditions referenced by `GET /train/{name}`.
