export DOCKERHUB_USER="deepfinery"
docker build -t docker.io/$DOCKERHUB_USER/nemo-trainer:1.0 -f services/nemo_api/Dockerfile .
docker push docker.io/$DOCKERHUB_USER/nemo-trainer:1.0
kubectl apply -f services/nemo_api/k8s/
kubectl get service -l app=nemo-trainer
kubectl get all
# Optional clean redeploy
kubectl delete -f services/nemo_api/k8s/
kubectl apply -f services/nemo_api/k8s/
kubectl get all
