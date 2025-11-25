# Trainer APIs

This repository provides three training APIs that wrap modern LLM fine-tuning stacks:

1. **Nemo Customizer API** – exposes a Nemotron/NVIDIA Nemo style fine-tuning endpoint.
2. **Meta Adapter API** – wraps Meta-style tuning workflows (or proxies to Nemo-compatible flow when native tooling is unavailable).
3. **Hugging Face + Unsloth API** – accelerates Hugging Face fine-tuning with Unsloth adapters while matching the same request schema.

Each API accepts a superset of tuning parameters (model location, adapters, PEFT/LORA/QLoRA options, logging sinks, artifacts, etc.), executes validation, and orchestrates a background job runner. The APIs are intentionally pluggable: you can point them at real training backends, or use the included mock job runner for local development and CI.

Every service ships with a Dockerfile and Kubernetes manifests to help you deploy onto a cluster quickly.

See `docs/USAGE.md` for workflows, `docs/API_SPEC.md` for the HTTP contract, and the `services/*/k8s` folders for Kubernetes deployment assets. Import the Postman collection under `docs/postman/trainer-apis.postman_collection.json` to exercise each API quickly.

## Launcher / Training Studio Notes

- When `base_model.provider == "huggingface"` you can now supply the launcher-provided `huggingface_token` so the backend can fetch gated checkpoints on behalf of the user.
- Training Studio organizes inputs/outputs at `s3://deepfinery-training-data-<account>/users/<userId>/projects/<projectId>/`. Place dataset uploads in the `ingestion/` folder and save logs/results in `logs/` and `results/` respectively to keep IAM policies aligned.

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

If the training job needs to download/upload datasets from Deepfinery S3, also export the shared credentials (or rely on the included `.env`):

```bash
export AWS_ACCESS_KEY_ID=XXXX
export AWS_SECRET_ACCESS_KEY='YYYYY'
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

#### Example Nemo Redeploy Commands

```bash
export DOCKERHUB_USER=<dockerhub-username-or-org>
docker build -t docker.io/$DOCKERHUB_USER/nemo-trainer:1.0 -f services/nemo_api/Dockerfile .
docker push docker.io/$DOCKERHUB_USER/nemo-trainer:1.0
kubectl apply -f services/nemo_api/k8s/
kubectl get service -l app=nemo-trainer
kubectl get all
# Optional clean redeploy
kubectl delete -f services/nemo_api/k8s/
kubectl apply -f services/nemo_api/k8s/
kubectl get all
```

### 4. Optional Hardening

- Swap the inline token for a Kubernetes Secret (e.g., `kubectl create secret generic trainer-api --from-literal=token=...` and reference it with `valueFrom.secretKeyRef`).
- Front the three Services with an Ingress Controller or Azure Application Gateway so you get TLS termination and a single hostname.
- Configure Azure Monitor/Container Insights for observability and add Horizontal Pod Autoscalers for bursty training loads.

## Calling the API

1. Find the service endpoint:
   ```bash
   kubectl get svc nemo-trainer -o wide
   # or meta-trainer / hf-unsloth-trainer depending on the backend
   export TRAINER_BASE_URL="http://<external-ip-from-get-svc>"
   ```
2. Create a payload (example for the HF + Unsloth service showing the launcher-provided Hugging Face token plus the canonical Deepfinery S3 paths):
   ```json
   {
     "job_id": "hf-demo-001",
     "base_model": {
       "provider": "huggingface",
       "model_name": "meta-llama/Llama-2-7b",
       "auth_token": "hf_token",
       "huggingface_token": "hf_launchpad_token"
     },
     "datasets": [
       {
         "source": "s3://deepfinery-training-data-123456/users/e468b458-c061-70ca-966f-bb439ffde5e3/projects/6925ccd958905b1e58631d2c/ingestion/1764106730023-tx_aml_dataset.jsonl",
         "format": "jsonl"
       }
     ],
     "customization": {
       "method": "qlora",
       "precision": "bf16",
       "gradient_checkpointing": true
     },
     "resources": {
       "gpus": 4,
       "gpu_type": "A100",
       "cpus": 32,
       "memory_gb": 256,
       "max_duration_minutes": 720
     },
     "artifacts": {
       "log_uri": "s3://deepfinery-training-data-123456/users/e468b458-c061-70ca-966f-bb439ffde5e3/projects/6925ccd958905b1e58631d2c/logs/hf-demo-001/",
       "output_uri": "s3://deepfinery-training-data-123456/users/e468b458-c061-70ca-966f-bb439ffde5e3/projects/6925ccd958905b1e58631d2c/results/"
     },
     "tuning_parameters": {
       "learning_rate": 0.0002,
       "batch_size": 64,
       "num_epochs": 3,
       "warmup_ratio": 0.1,
       "max_sequence_length": 4096
     }
   }
   ```
   Save this payload as `payload.json`. Swap the `provider` and dataset/artifact URIs as needed if you're targeting the Nemo or Meta services.
3. Send the request with the trainer token header (matching whatever is set in `TRAINER_API_TOKEN` or the `.env` file):
   ```bash
   curl -X POST "$TRAINER_BASE_URL/train" \
        -H "Content-Type: application/json" \
        -H "X-TRAINER-TOKEN: ${TRAINER_API_TOKEN:-trainer-demo-token}" \
        -d @payload.json
   ```
4. Poll job status or cancel if needed:
   ```bash
   # Status
   curl "$TRAINER_BASE_URL/train/hf-demo-001" \
        -H "X-TRAINER-TOKEN: ${TRAINER_API_TOKEN:-trainer-demo-token}"

   # Cancel
   curl -X POST "$TRAINER_BASE_URL/train/hf-demo-001/cancel" \
        -H "X-TRAINER-TOKEN: ${TRAINER_API_TOKEN:-trainer-demo-token}"
   ```
5. If you include a `callbacks.webhook_url` in the submission payload, the trainer automatically POSTs status updates (job id, backend id, status, detail, timestamp) to Training Studio every minute while the job is running or until it reaches a terminal state. Set `TRAINER_CALLBACK_INTERVAL_SECONDS` to tune the cadence (default 60s).

### Postman

Import `docs/postman/trainer-apis.postman_collection.json`, set the `nemo_base_url`, `meta_base_url`, `hf_unsloth_base_url`, `trainer_api_token`, and the sample job ids, then choose the `Submit Training Job`/`Get Job Status`/`Cancel Job` requests for the service you deployed. The bodies already reference the Deepfinery S3 bucket layout, include the Hugging Face token field, and ship with a sample callback configuration so you only need to tweak IDs or resource sizes.
