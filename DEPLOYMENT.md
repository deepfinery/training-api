# Deployment Checklist

## Prerequisites

1. GPU-enabled nodes exposing the `nvidia.com/gpu` resource (install the NVIDIA GPU Operator or equivalent drivers).
2. Kubeflow Training Operator with the `kubeflow.org/v1` `pytorchjobs` CRD installed.
3. Kubernetes namespace where the API runs and where PyTorchJobs should be created.

The provided manifests ship with a namespaced `ServiceAccount`, `Role`, and `RoleBinding` that grant the FastAPI
pod permission to `create/get/list/watch/delete` PyTorchJobs in-place.

You also need a trainer image (referenced by `TrainingJobRequest.image`) that bundles `python3`, `torch`, `transformers`, `datasets`, `pytorch-lightning`, and `boto3`. The `train.py` dispatcher expects these libraries to be present inside the pod launched by the PyTorchJob.

## Build & Push Image

```bash
export CONTAINER_REPO="docker.io/deepfinery"
export IMAGE_TAG="1.1"
docker build -t $CONTAINER_REPO/training-api:$IMAGE_TAG -f services/training_api/Dockerfile .
docker push $CONTAINER_REPO/training-api:$IMAGE_TAG
```

## Deploy the API

```bash
# Update services/training_api/k8s/deployment.yaml with your image + token.
kubectl apply -f services/training_api/k8s/

kubectl get deploy training-api
kubectl get svc training-api -o wide
```

This deploys the FastAPI service (port 8080) and exposes it via a LoadBalancer Service. `TRAINER_API_TOKEN`
controls authentication; match it with the `X-TRAINER-TOKEN` header in clients.

## Submit a Job

```bash
curl -H "X-TRAINER-TOKEN: trainer-demo-token" \
     -H "Content-Type: application/json" \
     -d @scripts/sample-unified-request.json \
     http://<lb-ip>/train
```

## Observe PyTorchJobs

Each submission creates a `kubeflow.org/v1` `PyTorchJob` named after `framework` + `run_id`.

```bash
kubectl get pytorchjobs
kubectl describe pytorchjob <name>
```

The API echoes this name back as `TrainingJobStatus.name`. Use standard `kubectl` commands to follow pods and logs.
