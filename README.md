# Trainer APIs

This repository provides three training APIs that wrap modern LLM fine-tuning stacks:

1. **Nemo Customizer API** – exposes a Nemotron/NVIDIA Nemo style fine-tuning endpoint.
2. **Meta Adapter API** – wraps Meta-style tuning workflows (or proxies to Nemo-compatible flow when native tooling is unavailable).
3. **Hugging Face + Unsloth API** – accelerates Hugging Face fine-tuning with Unsloth adapters while matching the same request schema.

Each API accepts a superset of tuning parameters (model location, adapters, PEFT/LORA/QLoRA options, logging sinks, artifacts, etc.), executes validation, and orchestrates a background job runner. The APIs are intentionally pluggable: you can point them at real training backends, or use the included mock job runner for local development and CI.

Every service ships with a Dockerfile and Kubernetes manifests to help you deploy onto a cluster quickly.

See `docs/USAGE.md` for workflows, `docs/API_SPEC.md` for the HTTP contract, and the `services/*/k8s` folders for Kubernetes deployment assets. Import the Postman collection under `docs/postman/trainer-apis.postman_collection.json` to exercise each API quickly.
# training-api

## Deploying to AKS with a Load Balancer

We assume you already have AKS up and running with `kubectl` pointing at it. The only external requirement is container image hosting. If you want to keep things simple, just push to Docker Hub (shown below). If your org prefers GHCR or another registry, adjust the tags/logins accordingly.

### 1. Build & Push Container Images

Authenticate Docker once (Docker Hub example):

```bash
export DOCKERHUB_USER="<dockerhub-username-or-org>"
export DOCKERHUB_PAT="<access-token-or-password>"
echo "$DOCKERHUB_PAT" | docker login docker.io -u "$DOCKERHUB_USER" --password-stdin
```

Build and push each API:

```bash
# Nemo
docker build -t docker.io/$DOCKERHUB_USER/nemo-trainer:1.0 services/nemo_api
docker push docker.io/$DOCKERHUB_USER/nemo-trainer:1.0

# Meta
docker build -t docker.io/$DOCKERHUB_USER/meta-trainer:1.0 services/meta_api
docker push docker.io/$DOCKERHUB_USER/meta-trainer:1.0

# HF + Unsloth
docker build -t docker.io/$DOCKERHUB_USER/hf-unsloth-trainer:1.0 services/hf_unsloth_api
docker push docker.io/$DOCKERHUB_USER/hf-unsloth-trainer:1.0
```

Update the `image:` fields inside `services/<api>/k8s/deployment.yaml` to reference the tags you just pushed (e.g., `docker.io/myuser/nemo-trainer:1.0`).

### 2. Configure the Hardcoded API Token

The `/train` endpoints require the `X-TRAINER-TOKEN` header when the `TRAINER_API_TOKEN` environment variable is present. The manifests now ship with a placeholder value (`trainer-demo-token`) so you can get going quickly:

```yaml
env:
  - name: TRAINER_API_TOKEN
    value: "trainer-demo-token"
```

Update the value before deploying (or better yet, source it from a Kubernetes Secret). For local testing you can run:

```bash
export TRAINER_API_TOKEN=trainer-demo-token
uvicorn services.nemo_api.app:app --reload
```

Then send requests with the matching header:

```bash
curl -H "X-TRAINER-TOKEN: trainer-demo-token" \
     -H "Content-Type: application/json" \
     -d @sample-request.json \
     http://localhost:8000/train
```

### 3. Apply the Manifests

The `services/*/k8s` directories contain a Deployment, ConfigMap (where applicable), and a Service. Each Service is of type `LoadBalancer`, so AKS automatically wires the pods to an Azure Public Load Balancer.

```bash
kubectl apply -f services/nemo_api/k8s/
kubectl apply -f services/meta_api/k8s/
kubectl apply -f services/hf_unsloth_api/k8s/
```

Within a few minutes each Service should list an external IP:

```bash
kubectl get service -l app=nemo-trainer
kubectl get service -l app=meta-trainer
kubectl get service -l app=hf-unsloth-trainer
```

Send requests by calling `http://<external-ip>/train` (or `/healthz`) and always include `X-TRAINER-TOKEN`.

### 4. Optional Hardening

- Swap the inline token for a Kubernetes Secret (e.g., `kubectl create secret generic trainer-api --from-literal=token=...` and reference it with `valueFrom.secretKeyRef`).
- Front the three Services with an Ingress Controller or Azure Application Gateway so you get TLS termination and a single hostname.
- Configure Azure Monitor/Container Insights for observability and add Horizontal Pod Autoscalers for bursty training loads.
