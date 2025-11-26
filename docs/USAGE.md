# Usage Guide

## Common Concepts

All APIs share the `/train` endpoint. POST a JSON payload describing the training job:

```json
{
  "job_id": "unique-train-id",
  "base_model": {
    "provider": "huggingface",
    "model_name": "meta-llama/Llama-2-7b",
    "revision": "main",
    "auth_token": "hf_xxx",
    "huggingface_token": "hf_launchpad_token",
    "weights_url": null
  },
  "customization": {
    "method": "lora",
    "rank": 16,
    "target_modules": ["q_proj", "v_proj"],
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "trainable_layers": ["decoder.layers.0"],
    "precision": "bf16",
    "qlora": {
      "use_double_quant": true,
      "bnb_4bit_compute_dtype": "bfloat16"
    },
    "peft": {
      "use": true,
      "config": {"task_type": "CAUSAL_LM"}
    }
  },
  "resources": {
    "gpus": 4,
    "gpu_type": "A100",
    "cpus": 16,
    "memory_gb": 128,
    "max_duration_minutes": 720
  },
  "datasets": [
    {
      "source": "s3://deepfinery-training-data-123456/users/e468b458-c061-70ca-966f-bb439ffde5e3/projects/6925ccd958905b1e58631d2c/ingestion/1764106730023-tx_aml_dataset.jsonl",
      "format": "jsonl",
      "auth": {"role_arn": "arn:aws:iam::123:role/FineTune"}
    }
  ],
  "artifacts": {
    "log_uri": "s3://deepfinery-training-data-123456/users/e468b458-c061-70ca-966f-bb439ffde5e3/projects/6925ccd958905b1e58631d2c/logs/job-id/",
    "status_stream_url": "https://events.internal/job-id",
    "output_uri": "s3://deepfinery-training-data-123456/users/e468b458-c061-70ca-966f-bb439ffde5e3/projects/6925ccd958905b1e58631d2c/results/"
  },
  "tuning_parameters": {
    "learning_rate": 2e-4,
    "batch_size": 64,
    "num_epochs": 3,
    "warmup_ratio": 0.1,
    "gradient_checkpointing": true,
    "max_sequence_length": 4096
  },
  "callbacks": {
    "webhook_url": "https://webhook.internal/train-status",
    "auth_header": "Bearer mytoken"
  }
}
```

Each service validates its backend-specific constraints. By default they submit to the in-memory mock scheduler
for local development, but when `TRAINER_JOB_RUNNER=kubernetes` is set the APIs create
[`trainer.kubeflow.org/v1alpha1` `TrainJob`](https://www.kubeflow.org/docs/components/trainer/) resources using
the runtime configured via `TRAINER_RUNTIME_NAME` (or `extra_parameters.runtime_ref` on a per-request basis).

When the job submission succeeds the API responds with status `submitted` and `detail: "Job accepted for processing"`. Follow-up status checks (described below) update `status` to `running`, `succeeded`, `failed`, or `cancelled` with a matching `detail` message, and—if you provided a callback—the service pushes the same payload to Training Studio automatically.

### S3 Layout

Training Studio organizes datasets and artifacts under a shared bucket prefix:

- Datasets uploaded by a user land at `s3://deepfinery-training-data-<account>/users/<userId>/projects/<projectId>/ingestion/<dataset>.jsonl`.
- Logs and checkpoints should be written back to `s3://deepfinery-training-data-<account>/users/<userId>/projects/<projectId>/logs/` and `.../results/`.

Reusing these conventions in `datasets[].source`, `artifacts.log_uri`, and `artifacts.output_uri` keeps IAM policies aligned with the launcher app.

### Checking Status or Cancelling

Every trainer service exposes:

- `GET /train/{job_id}` – returns the latest `TrainingResponse` object for that job (same schema as the submission response). Poll this endpoint from Launcher to surface `status` + `detail`.
- `POST /train/{job_id}/cancel` – transitions the job to `cancelled` when possible, or echoes the current terminal state with an informative `detail` string.

Remember to pass the `X-TRAINER-TOKEN` header on these requests as well.

### Callback Updates

Include a `callbacks` block in your request to have the trainer notify Training Studio automatically:

```json
"callbacks": {
  "webhook_url": "https://training-studio.internal/api/jobs/status",
  "auth_header": "Bearer studio-token"
}
```

When a job is pending or running, the API POSTs a JSON payload that mirrors `TrainingResponse` (`job_id`, `backend_job_id`, `status`, `detail`, `timestamp`) to the webhook every minute. Once the job reaches a terminal state (succeeded, failed, cancelled) the callback sends a final update and stops. Adjust the frequency by setting `TRAINER_CALLBACK_INTERVAL_SECONDS`.

## Running Locally

1. `cd services/<api>`
2. `python -m venv .venv && source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `uvicorn app:app --reload`

## API Tooling

- Full HTTP contract: `docs/API_SPEC.md`
- Postman collection: `docs/postman/trainer-apis.postman_collection.json` (variables default to localhost ports 8081/8082/8083; adjust for your cluster endpoints).

## Kubernetes

Each service exposes a minimal Deployment + Service manifest under `services/<api>/k8s`. Adjust requests/limits and config maps to supply credentials (tokens, storage URIs), then `kubectl apply -f k8s/`.

## Extending Backends

Implement `common.trainer.BaseTrainerBackend` and register it in the service-specific router. Use the shared Pydantic schemas in `common/schemas.py` to ensure compatibility across APIs.
