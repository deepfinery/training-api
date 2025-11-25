# Deployment

## Container Images

Each API has its own Dockerfile under `services/<api>/Dockerfile`. Build and push images with the tags used in the manifests:

```bash
# Nemo
cd services/nemo_api
docker build -t ghcr.io/your-org/nemo-trainer:latest .
docker push ghcr.io/your-org/nemo-trainer:latest

# Meta
cd ../meta_api
docker build -t ghcr.io/your-org/meta-trainer:latest .
docker push ghcr.io/your-org/meta-trainer:latest

# HF + Unsloth
cd ../hf_unsloth_api
docker build -t ghcr.io/your-org/hf-unsloth-trainer:latest .
docker push ghcr.io/your-org/hf-unsloth-trainer:latest
```

## Kubernetes

Apply the manifests inside `services/<api>/k8s`. Customize container images, replica counts, and resource requests before deploying.

```bash
kubectl apply -f services/nemo_api/k8s/
kubectl apply -f services/meta_api/k8s/
kubectl apply -f services/hf_unsloth_api/k8s/
```

For production you may want to front the APIs with an Ingress controller and wire ConfigMaps or Secrets for credentials (tokens, storage URIs, dataset roles). You can also template these manifests with Helm; each Deployment is intentionally small to serve as a baseline chart template.

### Callback Interval

Set the optional `TRAINER_CALLBACK_INTERVAL_SECONDS` environment variable inside the Deployment to control how often the service posts status updates to `callbacks.webhook_url` (defaults to 60 seconds). This is useful when Training Studio needs a different refresh cadence.
