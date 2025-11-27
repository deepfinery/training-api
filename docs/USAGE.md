# Usage Guide

## Requesting a Training Job

Send `POST /train` with the unified payload:

```json
{
  "framework": "hf",
  "model_id": "meta-llama/Llama-3-8B",
  "run_id": "aml_ruleset_v4",
  "num_nodes": 1,
  "gpus_per_node": 8,
  "cpus_per_node": 48,
  "memory_per_node_gb": 512,
  "image": "ghcr.io/deepfinery/trainer:latest",
  "checkpoint_base_uri": "s3://deepfinery-training/runs",
  "dataset_uri": "s3://deepfinery-datasets/fraud/aml_ruleset.jsonl",
  "resume_from_checkpoint": true,
  "extra_args": {
    "max_steps": 50000,
    "lr": 3e-5,
    "batch_size": 128
  }
}
```

### Field reference

| Field | Description |
| --- | --- |
| `framework` | `"nemo"`, `"hf"`, or `"meta"`; controls which trainer block `train.py` executes. |
| `model_id` | Human-readable id surfaced in logs; also forwarded to the container. |
| `run_id` | Unique id per run. Used to name the PyTorchJob and derive the S3 prefix. |
| `checkpoint_base_uri` | S3 prefix that stores all runs (e.g., `s3://deepfinery-training/runs`). |
| `dataset_uri` | Dataset location provided to `run_*` helpers. JSONL is expected by default (`{"text": "..."}` on each line). |
| `image` | Trainer container (must include `torchrun` + repo root `train.py`). |
| `num_nodes` / `gpus_per_node` / `cpus_per_node` / `memory_per_node_gb` | Shape of the PyTorchJob pods. |
| `resume_from_checkpoint` | If true, `train.py` searches `.../framework=<f>/checkpoints/step-*.json` and resumes from the highest step. |
| `config_path` | Optional config artifact to mount/read inside the container. |
| `extra_args` | Framework-specific knobs forwarded to the trainer. Examples: `dataset_text_field`, `max_seq_len`, `save_steps`, `gradient_accumulation`. |
| `env` | Additional env vars to add to the pod spec. |
| `labels`/`annotations` | Copied onto the PyTorchJob metadata. |

### What happens after submission?

1. The API validates the payload (Pydantic schema in `common/unified_schemas.py`).
2. `common/pytorch_job.py` renders a PyTorchJob manifest that sets env vars (`FRAMEWORK`, `RUN_ID`, `CHECKPOINT_PREFIX`, etc.) and builds a `torchrun ... train.py` command line.
3. `services/training_api.backend.PyTorchJobBackend` creates the `kubeflow.org/v1` `pytorchjobs` resource.
4. The Training Operator injects the distributed runtime env vars (`RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`), then launches the pods.
5. `train.py` parses CLI/env arguments, ensures each run writes to `s3://<base>/<RUN_ID>/framework=<framework>/{checkpoints,logs,config}`, persists configs, and dispatches to the requested framework helper where actual model training/checkpointing occurs.
6. The API surfaces status via `GET /train/{name}` by reading the PyTorchJob conditions.

### Framework-specific overrides

- **NeMo**: accepts `batch_size`, `num_epochs`, `lr`, `max_seq_len`, and `dataloader_workers`. Training is handled via PyTorch Lightning; set `extra_args.precision` to `bf16` or `fp16` to match GPU capability.
- **Hugging Face**: accepts any Trainer argument (examples: `gradient_accumulation`, `save_steps`, `eval_steps`, `logging_steps`, `dataset_text_field`). Checkpoints are uploaded incrementally as `checkpoints/checkpoint-<step>/...` alongside `step-<step>.json` metadata entries.
- **Meta/custom PyTorch**: accepts `batch_size`, `num_epochs`, `lr`, `max_seq_len`, `save_steps`. The FSDP runner writes `checkpoints/step-XXXXXX.pt` files (model + optimizer state) so resumes rebuild full state.

### S3 Layout

`train.py` enforces a normalized hierarchy for all frameworks:

```
s3://deepfinery-training/runs/<RUN_ID>/framework=<framework>/
  checkpoints/step-000001.json
  checkpoints/step-000002.json
  logs/.keep
  config/run.json
```

- **Checkpoints** – When `run_*` saves an artifact it should call `CheckpointManager.save_checkpoint(step, metadata)` so filenames conform to `step-XXXXXX.json` and resume discovery works automatically.
- **Logs** – Write framework logs or tensorboard data under `logs/`.
- **Config** – `CheckpointManager.persist_config()` records the final config for reproducibility.

### Checking Status

`GET /train/{name}` returns the `TrainingJobStatus` object with `status in {submitted,running,failed,succeeded}` and an optional `detail` message derived from PyTorchJob conditions.

### Authentication

Set `TRAINER_API_TOKEN` on the Deployment or your local shell. All requests must include `X-TRAINER-TOKEN: <same value>`.

### Postman

Import `docs/postman/trainer-apis.postman_collection.json`, set `trainer_base_url` and `trainer_api_token`, then use the `Submit Training Job` and `Get Job Status` requests.

## Running Locally

1. `python3 -m venv .venv && source .venv/bin/activate`
2. `pip install -r services/training_api/requirements.txt`
3. `export TRAINER_API_TOKEN=trainer-demo-token`
4. `uvicorn services.training_api.app:app --reload`

Submit the sample payload from `scripts/sample-unified-request.json` (update the URIs before running).

### Storage credentials

`train.py` relies on `boto3` for dataset downloads and checkpoint uploads. Provide credentials via standard AWS env vars inside the trainer pod:

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=...
```

If the cluster uses IAM roles for service accounts or workload identity, make sure the trainer pod inherits a role with read/write access to the referenced S3 prefixes.

## Kubernetes Requirements

- Kubeflow Training Operator installed (for `pytorchjobs` CRD).
- ServiceAccount with `kubeflow.org` `pytorchjobs` verbs (`get`, `list`, `watch`, `create`, `delete`).
- Access to the target object store (S3 credentials or IAM binding inside the container).

Use the manifests in `services/training_api/k8s` as a starting point.
