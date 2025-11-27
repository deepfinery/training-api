# Unified Trainer API Specification

All requests target the single FastAPI service deployed from `services/training_api`. The API surface:

- `GET /healthz`
- `POST /train`
- `GET /train/{job_name}`

## Health Check

`GET /healthz`

```json
{ "status": "ok" }
```

## Submit Training Job (`POST /train`)

The body must match `common.unified_schemas.TrainingJobRequest`:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `framework` | enum(`nemo`,`hf`,`meta`) | ✓ | Which framework the container should execute. |
| `model_id` | string | ✓ | Human-readable identifier, forwarded to the trainer container. |
| `run_id` | string | ✓ | Unique identifier per run; becomes part of the PyTorchJob name and S3 prefix. |
| `checkpoint_base_uri` | string | ✓ | Bucket/prefix that will contain all run artifacts. The service derives `<base>/<run_id>/framework=<framework>/...`. |
| `dataset_uri` | string | ✓ | Dataset location forwarded to the trainer. |
| `image` | string |  | Trainer container image (defaults to `ghcr.io/deepfinery/trainer:latest`). Must bundle `train.py`. |
| `num_nodes` | integer |  | Number of PyTorchJob replicas. `>=1`. |
| `gpus_per_node` | integer |  | GPUs per replica; maps to torchrun `--nproc_per_node`. |
| `cpus_per_node` | integer |  | CPU requests/limits per replica. |
| `memory_per_node_gb` | integer |  | Memory requests/limits (GiB) per replica. |
| `config_path` | string |  | Optional path/config artifact consumed by `train.py`. |
| `resume_from_checkpoint` | bool |  | When true, `train.py` resumes from the most recent checkpoint file. |
| `extra_args` | object |  | Framework-specific flags appended to the `train.py` CLI (`--<key> <value>`). Booleans become `--flag` when true. |
| `env` | object |  | Additional environment variables added to each pod. |
| `labels`/`annotations` | object |  | Copied to the PyTorchJob metadata. |
| `namespace` | string |  | Override namespace (defaults to pod namespace via `common.kube.detect_namespace`). |
| `job_name` | string |  | Explicit PyTorchJob name override. |

### Example

```json
{
  "framework": "meta",
  "model_id": "meta-llama-3-70b",
  "run_id": "meta_demo_001",
  "num_nodes": 4,
  "gpus_per_node": 8,
  "cpus_per_node": 64,
  "memory_per_node_gb": 512,
  "checkpoint_base_uri": "s3://deepfinery-training/runs",
  "dataset_uri": "s3://deepfinery-datasets/aml/v1/",
  "resume_from_checkpoint": false,
  "extra_args": {
    "max_steps": 6000,
    "lr": 2e-5,
    "gradient_accumulation": 16
  },
  "env": {
    "EXPERIMENT_NAME": "meta-demo"
  },
  "labels": {
    "deepfinery.com/team": "aml"
  }
}
```

### Response

`201 Created`

```json
{
  "name": "meta-meta-demo-001",
  "status": "submitted",
  "detail": "PyTorchJob created",
  "framework": "meta",
  "run_id": "meta_demo_001"
}
```

### Failure Modes

- `400 Bad Request` – schema validation failed or a job with the same name already exists.
- `401 Unauthorized` – missing/incorrect `X-TRAINER-TOKEN` header.
- `500` – unexpected error while creating the PyTorchJob.

## Get Training Job Status (`GET /train/{job_name}`)

Returns the same `TrainingJobStatus` shape. Values derive from PyTorchJob `.status.conditions`:

| Condition | API `status` | Detail |
| --- | --- | --- |
| `Succeeded=True` | `succeeded` | `Succeeded.message` or `"PyTorchJob completed"` |
| `Failed=True` | `failed` | `Failed.message` or `"PyTorchJob failed"` |
| `Running=True` or `status.active>0` | `running` | `Running.message` or `"PyTorchJob pods active"` |
| None of the above | `submitted` | `"PyTorchJob submitted"` |

`404 Not Found` when the resource does not exist.

## Authentication

Set `TRAINER_API_TOKEN=<token>` on the Deployment (or locally) and include `X-TRAINER-TOKEN: <token>` in every request. Omit the env var to disable auth during development.

## Checkpoint Layout

`train.py` standardizes checkpoint/log/config outputs:

```
s3://<checkpoint_base_uri>/<RUN_ID>/framework=<framework>/
  checkpoints/step-000001.json
  checkpoints/step-000002.json
  logs/
  config/run.json
```

Frameworks must save their native checkpoints (NeMo exp_manager, HF Trainer, Meta custom) into the `checkpoints/` folder following the `step-XXXXXX` convention so resume works across restarts. `CheckpointManager` handles filename generation and metadata, but framework code controls what gets written.
Typical `extra_args` include `dataset_text_field` (defaults to `"text"`), `batch_size`, `num_epochs`, `lr`, `max_seq_len`, `gradient_accumulation`, `save_steps`, and per-framework toggles such as `dataloader_workers` (NeMo) or `eval_steps` (HF Trainer).
