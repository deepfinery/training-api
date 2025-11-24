# Trainer APIs

This repository provides three training APIs that wrap modern LLM fine-tuning stacks:

1. **Nemo Customizer API** – exposes a Nemotron/NVIDIA Nemo style fine-tuning endpoint.
2. **Meta Adapter API** – wraps Meta-style tuning workflows (or proxies to Nemo-compatible flow when native tooling is unavailable).
3. **Hugging Face + Unsloth API** – accelerates Hugging Face fine-tuning with Unsloth adapters while matching the same request schema.

Each API accepts a superset of tuning parameters (model location, adapters, PEFT/LORA/QLoRA options, logging sinks, artifacts, etc.), executes validation, and orchestrates a background job runner. The APIs are intentionally pluggable: you can point them at real training backends, or use the included mock job runner for local development and CI.

Every service ships with a Dockerfile and Kubernetes manifests to help you deploy onto a cluster quickly.

See `docs/USAGE.md` for workflows, `docs/API_SPEC.md` for the HTTP contract, and the `services/*/k8s` folders for Kubernetes deployment assets. Import the Postman collection under `docs/postman/trainer-apis.postman_collection.json` to exercise each API quickly.
# training-api
