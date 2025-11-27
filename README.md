# Training API

A single FastAPI service (`services/training_api`) now fronts every training workload. Launcher posts a unified request that includes the framework (`nemo | hf | meta`), run metadata, hardware layout, and S3 locations. The API renders a Kubeflow [`PyTorchJob`](https://www.kubeflow.org/docs/components/training/pytorch/) that boots every replica with the same container/entrypoint. Inside the container we always call:

```
torchrun --nproc_per_node=$GPUS_PER_NODE \
         --nnodes=$NUM_NODES --node_rank=$RANK \
         --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
         train.py --framework ${FRAMEWORK} --run-id ${RUN_ID} \
         --checkpoint-base-uri ${CHECKPOINT_BASE_URI} \
         --checkpoint-prefix ${CHECKPOINT_PREFIX} --dataset-uri ${DATASET_URI} \
         --logs-uri ${LOGS_URI} --config-uri ${CONFIG_URI} --model-id ${MODEL_ID} \
         ${EXTRA_ARGS}
```

The repository-root `train.py` script is the torchrun entrypoint. It parses the framework switch, initializes checkpoint/log/config folders under `s3://<runs-bucket>/<RUN_ID>/framework=<framework>/`, finds the most recent checkpoint when resume is requested, and dispatches to `run_nemo`, `run_hf`, or `run_meta` where framework-specific trainers live. torchrun is only responsible for distributed bootstrap; all checkpoint IO happens in those framework functions so every run follows the same S3 layout.

## Trainer Image Requirements

The container that runs `train.py` must include the ML dependencies used by the three frameworks:

- PyTorch with CUDA (`torch`, `torchvision`, `torch.distributed`),
- Hugging Face Transformers + Datasets (`transformers`, `datasets`),
- PyTorch Lightning (used by the NeMo-style runner),
- `boto3` for checkpoint/log uploads to S3,
- Optionally `nemo_toolkit` to take advantage of `nemo.utils.exp_manager` (the NeMo runner will use it when available).

The provided `train.py` imports the framework modules lazily, so your API pods can stay lightweight while the trainer image carries the heavy dependencies. Export your storage credentials (e.g., AWS keys) through the pod `env` or the node IAM role so boto3 can reach the buckets referenced by `CHECKPOINT_BASE_URI` and `DATASET_URI`.

## Framework Behavior

- **NeMo runner** – builds a Lightning-compatible trainer around Hugging Face causal LM weights, integrates with `nemo.utils.exp_manager` when available, and mirrors NeMo's checkpoint cadence by uploading `.ckpt` files and Lightning logs to the shared S3 prefix. Override `extra_args` (batch size, epochs, LR, etc.) to fine-tune the Lightning training loop.
- **Hugging Face runner** – uses `transformers.Trainer`. Every saved `checkpoint-<step>` directory is uploaded to `.../checkpoints/` immediately so you can resume mid-run. The latest checkpoint is automatically downloaded and passed to `Trainer.train(resume_from_checkpoint=...)` when `resume_from_checkpoint` is set.
- **Meta/custom runner** – spins up a PyTorch/FSDP training loop that tokenizes the JSONL dataset, wraps the model with FSDP whenever `world_size > 1`, and saves `.pt` checkpoints containing model + optimizer state. Those checkpoints live under `checkpoints/step-XXXXXX.pt` and are downloaded before resuming.

Datasets are expected to be JSONL files with a `text` field by default. Override `extra_args.dataset_text_field` to point at a different key. The helper utilities automatically download `DATASET_URI` (S3 or local), split it into train/eval shards, and feed each framework-specific trainer.

## Architecture

- **Orchestration layer** – Kubeflow `PyTorchJob` CRD. Each submission includes `num_nodes`, `gpus_per_node`, container image, and env vars so the Training Operator can launch workers.
- **Launch layer** – torchrun is always the container entrypoint; it receives cluster metadata via env vars injected by the Training Operator (`RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`).
- **Framework layer** – `train.py` dispatches to `run_nemo/run_hf/run_meta`. These helpers construct trainers, manage checkpoint + resume logic, and write configs/logs/metadata back to the normalized S3 hierarchy.
- **API layer** – `services/training_api` exposes `/healthz`, `POST /train`, and `GET /train/{name}`. Requests use the `TrainingJobRequest` schema from `common/unified_schemas.py` and responses return `TrainingJobStatus` summaries.

See `docs/USAGE.md` for an end-to-end walkthrough and `docs/API_SPEC.md` for the exact JSON schema.

## Running Locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r services/training_api/requirements.txt
export TRAINER_API_TOKEN=trainer-demo-token
uvicorn services.training_api.app:app --reload
```

Send requests with the `X-TRAINER-TOKEN` header:

```bash
curl -H "X-TRAINER-TOKEN: trainer-demo-token" \
     -H "Content-Type: application/json" \
     -d @sample-request.json \
     http://localhost:8000/train
```

`scripts/sample-unified-request.json` (create your own) should follow the structure below:

```json
{
  "framework": "nemo",
  "model_id": "nemotron-8b-fx",
  "num_nodes": 2,
  "gpus_per_node": 8,
  "cpus_per_node": 48,
  "memory_per_node_gb": 512,
  "image": "ghcr.io/deepfinery/trainer:latest",
  "run_id": "fx_ruleset_v3_2025_11_26_01",
  "checkpoint_base_uri": "s3://deepfinery-training/runs",
  "dataset_uri": "s3://deepfinery-datasets/fx/v3/",
  "resume_from_checkpoint": true,
  "extra_args": {
    "max_steps": 50000,
    "lr": 3e-5,
    "batch_size": 128
  }
}
```

## Container Build & Deploy

The unified API ships with a Dockerfile under `services/training_api/Dockerfile`.

```bash
export DOCKERHUB_USER="<dockerhub-username-or-org>"
export DOCKERHUB_PAT="<access-token-or-password>"
echo "$DOCKERHUB_PAT" | docker login docker.io -u "$DOCKERHUB_USER" --password-stdin

IMAGE_TAG=1.0
ROOT=$(pwd)
docker build -t docker.io/$DOCKERHUB_USER/training-api:$IMAGE_TAG -f services/training_api/Dockerfile "$ROOT"
docker push docker.io/$DOCKERHUB_USER/training-api:$IMAGE_TAG
```

Update `services/training_api/k8s/deployment.yaml` with your tag, then apply the manifests:

```bash
kubectl apply -f services/training_api/k8s/
```

The manifests create:

- A Deployment that runs the FastAPI app, injects `TRAINER_API_TOKEN`, and grants RBAC access to the `kubeflow.org/v1` `pytorchjobs` resource.
- A LoadBalancer Service exposing port 8080.
- ServiceAccount/Role/RoleBinding scoped to PyTorchJob CRUD.

Check the service IP and submit jobs:

```bash
kubectl get svc training-api -o wide
export TRAINER_BASE_URL="http://<external-ip>"
```

## Kubeflow & Torchrun Notes

- Each request becomes a `PyTorchJob` where `Master.replicas=1` and `Worker.replicas=num_nodes-1` (when `num_nodes>1`).
- Env vars such as `FRAMEWORK`, `RUN_ID`, `MODEL_ID`, `NUM_NODES`, `GPUS_PER_NODE`, `CHECKPOINT_BASE_URI`, `CHECKPOINT_PREFIX`, and `TRAINING_EXTRA_ARGS_JSON` are injected into every pod.
- `train.py` consumes these env vars, discovers the latest checkpoint under `s3://<runs>/<RUN_ID>/framework=<f>/checkpoints/step-*.json`, and resumes when `resume_from_checkpoint` (or `--resume`) is set.
- Checkpoints/logs/configs live under `s3://deepfinery-training/runs/<RUN_ID>/framework=<framework>/{checkpoints,logs,config}` so Training Studio can reason about any run regardless of framework.

## Tooling

- **API schema** – `docs/API_SPEC.md`
- **Usage walkthrough** – `docs/USAGE.md`
- **Deployment tips** – `docs/DEPLOYMENT.md`
- **Postman collection** – `docs/postman/trainer-apis.postman_collection.json` (set `trainer_base_url` and `trainer_api_token`).
- **Helper script** – `scripts/build_and_deploy.sh` builds/pushes the image and reapplies manifests.

Training Studio or other callers only need to choose the framework, fill in S3 URIs, hardware counts, and optional `extra_args`: the API handles everything else.
