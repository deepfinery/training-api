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
      "source": "s3://bucket/dataset.jsonl",
      "format": "jsonl",
      "auth": {"role_arn": "arn:aws:iam::123:role/FineTune"}
    }
  ],
  "artifacts": {
    "log_uri": "s3://bucket/logs/job-id/",
    "status_stream_url": "https://events.internal/job-id",
    "output_uri": "s3://bucket/models/job-id/"
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

Each service validates its backend-specific constraints and ingests the job into the mock scheduler. Replace the `MockJobRunner` with your infrastructure (Kubernetes Jobs, Slurm, SageMaker, etc.).

## Running Locally

1. `cd services/<api>`
2. `python -m venv .venv && source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `uvicorn app:app --reload`

## Kubernetes

Each service exposes a minimal Deployment + Service manifest under `services/<api>/k8s`. Adjust requests/limits and config maps to supply credentials (tokens, storage URIs), then `kubectl apply -f k8s/`.

## Extending Backends

Implement `common.trainer.BaseTrainerBackend` and register it in the service-specific router. Use the shared Pydantic schemas in `common/schemas.py` to ensure compatibility across APIs.
