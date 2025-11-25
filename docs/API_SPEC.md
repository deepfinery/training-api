# Trainer API Specification

All services expose the same surface area so UI clients can reuse payload builders. Replace the hostname with the appropriate deployment (Nemo, Meta, HF+Unsloth).

## Health Check

`GET /healthz`

**Response**
```json
{
  "status": "ok"
}
```
Use for readiness probes.

## Submit Training Job

`POST /train`

### Request Body

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `job_id` | string | ✓ | Unique id you generate per request. |
| `base_model` | object | ✓ | Source info for the pretrained checkpoint. |
| `datasets[]` | array | ✓ | One or more dataset descriptors. |
| `customization` | object | ✓ | Defines tuning method (LoRA/QLoRA/PEFT/full) and precision knobs. |
| `resources` | object | ✓ | Hardware & duration requirements. |
| `artifacts` | object | ✓ | Where to store logs + final model checkpoints. |
| `tuning_parameters` | object | ✓ | Core hyperparameters (lr, batch size, epochs ...). |
| `callbacks` | object |  | Optional webhook notifications. |
| `extra_parameters` | object |  | Backend-specific overrides (container image, executor, unsloth tuning flags). |

### Schema Details

#### `base_model`
- `provider`: `"nemo" | "nemotron" | "bim" | "meta" | "huggingface" | "custom"`
- `model_name`: e.g., `"meta-llama/Llama-2-7b"`
- `revision`: optional git/tag revision
- `weights_url`: optional direct download URL
- `auth_token`: optional provider token
- `huggingface_token`: optional token coming from the Launcher/Training Studio when the provider is `huggingface`

#### `datasets[]`
- `source`: URI/path (S3/GCS/Azure/minio/local). For enterprise tenants the canonical S3 structure is
  `s3://deepfinery-training-data-<account>/users/<userId>/projects/<projectId>/ingestion/<dataset>.jsonl`.
- `format`: `"jsonl" | "parquet" | "csv" | "hf" | "nemo"`
- `split`: optional split name
- `auth`: `{ token | username/password | role_arn }`
- `streaming`: bool

#### `customization`
- `method`: `"full" | "lora" | "qlora" | "peft" | "adapter"`
- `trainable_layers`: optional layer list
- `precision`: `"fp32" | "fp16" | "bf16" | "int8"`
- `gradient_checkpointing`: bool
- `lora`: `{ rank, target_modules[], alpha, dropout }`
- `qlora`: `{ use_double_quant, bnb_4bit_compute_dtype, bnb_4bit_quant_type }`
- `peft`: `{ use, config }`

#### `resources`
- `gpus`, `gpu_type`
- `cpus`
- `memory_gb`
- `max_duration_minutes`

#### `artifacts`
- `log_uri`: e.g., `s3://bucket/logs/job-123/`
- `status_stream_url`: optional SSE/WS endpoint
- `output_uri`: e.g., `gs://bucket/models/job-123/`. For Deepfinery buckets, results typically go to
  `s3://deepfinery-training-data-<account>/users/<userId>/projects/<projectId>/results/`.

#### `tuning_parameters`
- `learning_rate`, `batch_size`, `micro_batch_size`, `num_epochs`, `warmup_ratio`, `max_sequence_length`, `weight_decay`

#### `callbacks`
- `webhook_url`, `auth_header`, `event_filter[]`

#### `extra_parameters`
- `container_image`, `executor`, `unsloth` etc.

### Example

```json
{
  "job_id": "demo-001",
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
    "log_uri": "s3://deepfinery-training-data-123456/users/e468b458-c061-70ca-966f-bb439ffde5e3/projects/6925ccd958905b1e58631d2c/logs/demo-001/",
    "output_uri": "s3://deepfinery-training-data-123456/users/e468b458-c061-70ca-966f-bb439ffde5e3/projects/6925ccd958905b1e58631d2c/results/"
  },
  "tuning_parameters": {
    "learning_rate": 0.0002,
    "batch_size": 64,
    "num_epochs": 3,
    "warmup_ratio": 0.1,
    "max_sequence_length": 4096
  },
  "callbacks": {
    "webhook_url": "https://webhooks.internal/train",
    "event_filter": ["submitted", "succeeded"]
  },
  "extra_parameters": {
    "container_image": "ghcr.io/your-org/hf-unsloth-trainer:latest",
    "unsloth": {
      "optimize_long_context": true,
      "quantization": "bnb-nf4",
      "gradient_accumulation": 8
    }
  }
}
```

### Response

```json
{
  "job_id": "demo-001",
  "backend_job_id": "hf-unsloth-demo-001",
  "status": "submitted",
  "detail": null,
  "dashboard_url": null,
  "metadata": {
    "job_spec": { "..." }
  }
}
```

- `status` mirrors the mock runner states: `submitted`, `running`, `failed`, `succeeded`.
- `metadata.job_spec` echoes what was handed to the backend/job runner.

### Backend-Specific Notes

| API | Endpoint | Extra Parameters |
| --- | --- | --- |
| Nemo | `https://nemo-trainer.yourdomain/train` | `extra_parameters.container_image` defaults to `nvcr.io/nvidia/nemo:24.03`. Requires provider `nemo | nemotron | bim`. |
| Meta | `https://meta-trainer.yourdomain/train` | `extra_parameters.executor` defaults to `torchrun`. Needs >=2 GPUs. |
| HF + Unsloth | `https://hf-unsloth-trainer.yourdomain/train` | `extra_parameters.unsloth` block controls quantization + long-context optimizations. |

### Error Handling

- 400 Bad Request: payload failed validation (`detail` contains message).
- 500 Internal Server Error: unexpected backend issue.

### Authentication

Add your desired auth layer (e.g., API Gateway, Service Mesh) in front of these services. Request schema already carries tokens for the *training providers*; they are not used for API auth.
