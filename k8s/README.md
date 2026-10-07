# Kubernetes deployment (local cluster)

Runs the backend and frontend on a local Kubernetes cluster (Docker Desktop with the kind provisioner and
3 nodes, or minikube). It follows the structure of the course reference (`llmapp09`): one namespace, a
Deployment and a Service per component, probes, and `kubectl port-forward` to reach it.

| File | What it is |
|---|---|
| `namespace.yaml` | Namespace `aem` with the restricted Pod Security level |
| `configmap.yaml` | Non-secret settings (including `AUDIT_LOG_PER_INSTANCE`, see below) |
| `secret.example.yaml` | Example only. The real secret is created by `scripts/k8s_secret.py` |
| `backend.yaml` | Backend Deployment (2 replicas) and Service |
| `frontend.yaml` | Frontend Deployment (2 replicas) and Service |
| `hpa.yaml` | Autoscaler for the backend (needs metrics-server) |
| `kustomization.yaml` | Applies the files above (except the secret and the autoscaler) |

## Run it
1. Enable Kubernetes in Docker Desktop (Settings, Kubernetes) and check `kubectl get nodes`.
2. Build the two images. A local cluster uses them directly, so no registry is needed:
   ```
   docker build -f Dockerfile.backend  -t aem-backend:k8s .
   docker build -f Dockerfile.frontend -t aem-frontend:k8s .
   ```
3. Create the namespace and the secret (your `OPENAI_API_KEY` from `.env`; it is never printed or committed), then deploy:
   ```
   kubectl apply -f k8s/namespace.yaml
   python scripts/k8s_secret.py
   kubectl apply -k k8s
   kubectl -n aem rollout status deploy/backend      # the first start takes about a minute
   ```
4. Open the app:
   ```
   kubectl -n aem port-forward svc/frontend 5000:5000      # then http://localhost:5000
   ```
5. Watch it: `kubectl -n aem get pods -o wide`, `kubectl -n aem logs deploy/backend`.
6. Remove it: `kubectl delete namespace aem`.

## Scale it
```
kubectl -n aem scale deploy/backend --replicas=4
```
For autoscaling, install metrics-server (on kind it needs the `--kubelet-insecure-tls` flag), then `kubectl apply -f k8s/hpa.yaml`.

## Design notes
- **Probes.** The backend uses `/ready` (green only once the agents have loaded) for the startup and readiness probes, and `/health` for liveness. A pod that never becomes ready is restarted, and an unready pod gets no traffic.
- **Audit log with several replicas.** Several pods appending to one hash chain would corrupt it, so each pod writes its own file (`audit-<pod>.jsonl`) and `verify_audit_integrity()` checks every file. The files live in an `emptyDir`, so they last as long as the pod; a real deployment would ship them to a durable store.
- **No service-link variables.** `enableServiceLinks: false`. Otherwise Kubernetes injects `FRONTEND_PORT=tcp://...` for the Service named `frontend`, which collides with the app's own `FRONTEND_PORT` setting and crashes the backend.
- **Hardening.** Non-root user (UID 10001), no privilege escalation, all capabilities dropped, a seccomp profile, a read-only root filesystem on the frontend, resource requests and limits, and the restricted Pod Security level on the namespace.
- **Rolling updates** never drop below the desired ready pods (`maxUnavailable: 0`), and pods get a long termination grace period so slow agent requests can finish.
- **Langfuse** is not wired in here: the pods cannot reach a Langfuse running on your laptop by default. Leave the Langfuse keys out of the secret.
- **Real clusters.** Push the images to a registry and change `image:` and `imagePullPolicy`. The secret should come from a secrets manager, not from `.env`.
