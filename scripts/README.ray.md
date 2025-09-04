helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
# Install both CRDs and KubeRay operator v1.4.0.
helm install kuberay-operator kuberay/kuberay-operator --version 1.4.0

# Deploy a sample RayCluster CR from the KubeRay Helm chart repo:
helm install raycluster kuberay/ray-cluster --version 1.4.0


kubectl get pods

Dashboard:
kubectl port-forward service/raycluster-head-svc 8265:8265

Ray client (for Python connection):
kubectl port-forward service/raycluster-head-svc 10001:10001
