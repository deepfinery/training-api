## Prerequisites

1. Install the NVIDIA GPU Operator (or otherwise ensure GPU nodes expose the `nvidia.com/gpu` resource).
2. Deploy the [Kubeflow Trainer controller](https://www.kubeflow.org/docs/components/trainer/getting-started/) and
   apply the `ClusterTrainingRuntime` definitions that match your training stack (NeMo, Meta/Torch, Unsloth, etc.).
3. Create a namespace for the API pods. Unless `TRAINER_JOBS_NAMESPACE` is overridden, the APIs create `TrainJob`
   resources in the same namespace they run in.

Each service now comes with a dedicated `ServiceAccount`, `Role`, and `RoleBinding` that grant access to
`trainer.kubeflow.org/trainjobs`. Apply the manifest per namespace so the RBAC objects exist alongside the pods.

## Build and Push an API Image

```bash
export DOCKERHUB_USER="deepfinery"
docker build -t docker.io/$DOCKERHUB_USER/nemo-trainer:1.1 -f services/nemo_api/Dockerfile .
docker push docker.io/$DOCKERHUB_USER/nemo-trainer:1.1
```

Repeat for the Meta and HF + Unsloth services (`services/meta_api/Dockerfile` and `services/hf_unsloth_api/Dockerfile`),
adjusting the tags as necessary.

## Deploy the Service

```bash
kubectl apply -f services/nemo_api/k8s/
kubectl get service -l app=nemo-trainer
kubectl get all -l app=nemo-trainer
```

The manifest sets `TRAINER_JOB_RUNNER=kubernetes`, `TRAINER_RUNTIME_NAME=nemo-runtime`, and points the `TrainJob`
namespace to the pod namespace via a `fieldRef`. For Meta and HF deployments, the manifest sets the runtime name to
`meta-runtime` and `hf-unsloth-runtime` respectively. Override any of these by editing the deployment or injecting
`extra_parameters.runtime_ref` in the request body.

To redeploy cleanly:

```bash
kubectl delete -f services/nemo_api/k8s/
kubectl apply -f services/nemo_api/k8s/
kubectl get all -l app=nemo-trainer
```

## Observing TrainJobs

Submitting `POST /train` now creates a `trainer.kubeflow.org/v1alpha1` `TrainJob`. Inspect it with:

```bash
kubectl get trainjobs.trainer.kubeflow.org
kubectl describe trainjob <name>
```

The job name is derived from the API `job_id` and is stored in the API response as `backend_job_id`. Use `kubectl`
to follow the pods spawned by the referenced `ClusterTrainingRuntime`.
