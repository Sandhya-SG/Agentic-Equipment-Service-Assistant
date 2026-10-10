# Kubernetes deployment (local cluster)

Runs the backend and frontend on a local Kubernetes cluster (Docker Desktop with the kind provisioner and
3 nodes, or minikube). It follows the structure of the course reference (`llmapp09`): one namespace, a
Deployment and a Service per component, probes, and `kubectl port-forward` to reach it.

## Who uses this
**Only Sandhya runs the Kubernetes deployment** (on her laptop, for the demo and the scalability evidence). Teammates do not set it up: they run the application with Option A or B and use their own Langfuse, as described in the main [README](../README.md) ("Team quick start"). Ask Sandhya before changing anything under `k8s/` or the Dockerfiles; they carry the security settings (read-only filesystem, network policy, non-root user) that the report documents.

### After you change code (maintainer)
The cluster does not see code changes until the image is rebuilt **and** every kind node has it, because each node keeps its own copy of a tag (`IfNotPresent`):
```
docker build -f Dockerfile.backend -t aem-backend:k8s .
docker save aem-backend:k8s -o img.tar
docker exec -i desktop-control-plane ctr -n k8s.io images import --all-platforms - < img.tar
docker exec -i desktop-worker        ctr -n k8s.io images import --all-platforms - < img.tar
docker exec -i desktop-worker2       ctr -n k8s.io images import --all-platforms - < img.tar
kubectl -n aem rollout restart deploy/backend
```
Same for the frontend with `aem-frontend:k8s`. The `<` redirect works in Git Bash, not in plain PowerShell.

### If something goes wrong
| Symptom | Cause and fix |
|---|---|
| Backend pod `0/1`, restarting | Read `kubectl -n aem logs <pod>`. "attempt to write a readonly database": the `seed-index` init container did not run; reapply with `kubectl apply -k k8s`. Name-resolution errors: see Troubleshooting below |
| Every answer says "temporarily unavailable" | The secret is missing or wrong: rerun `python scripts/k8s_secret.py`, then `kubectl -n aem rollout restart deploy/backend` |
| Pods stay `Pending` | Not enough memory in Docker Desktop: raise it, or `kubectl -n aem scale deploy/backend --replicas=1` |
| A change is not visible | The image was not imported into the nodes, see above |

## What is in this folder

| File | What it is |
|---|---|
| `namespace.yaml` | Namespace `aem` with the restricted Pod Security level |
| `configmap.yaml` | Non-secret settings (including `AUDIT_LOG_PER_INSTANCE`, see below) |
| `secret.example.yaml` | Example only. The real secret is created by `scripts/k8s_secret.py` |
| `backend.yaml` | Backend Deployment (2 replicas) and Service |
| `frontend.yaml` | Frontend Deployment (2 replicas) and Service |
| `networkpolicy.yaml` | Default-deny network rules: only the frontend reaches the backend, the backend may reach only DNS and port 443 |
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
- **Hardening.** Non-root user (UID 10001), no privilege escalation, all capabilities dropped, a seccomp profile, a read-only root filesystem on both pods, no service-account token (`automountServiceAccountToken: false`), resource requests and limits (including ephemeral storage), size limits on every `emptyDir`, and the restricted Pod Security level on the namespace.
- **Read-only backend and Chroma.** Chroma's SQLite store needs write access even to read, so an init container (`seed-index`) copies the index baked into the image (3.4 MB) into a writable `emptyDir` mounted at `/app/chroma_store`. Without it the agents fail to load ("attempt to write a readonly database") and `/ready` stays 503.
- **Network policy** (`networkpolicy.yaml`): default deny, DNS allowed, the frontend may only be reached on 5000 and may only call the backend on 8000, and the backend accepts traffic only from the frontend and may only reach port 443 outside (OpenAI). Checked on the kind cluster: frontend to backend works, frontend to the internet and backend to a peer backend time out, backend to api.openai.com connects.
- **After rebuilding an image** with the same tag, the kind nodes keep their old copy (`IfNotPresent`). Import it into each node before `rollout restart`: `docker save aem-backend:k8s aem-frontend:k8s -o img.tar`, then `docker exec -i <node> ctr -n k8s.io images import --all-platforms - < img.tar` for `desktop-control-plane`, `desktop-worker` and `desktop-worker2`.
- **metrics-server (for the autoscaler).** Install a pinned release and add `--kubelet-insecure-tls` for kind: `kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/download/v0.9.0/components.yaml`, then `kubectl -n kube-system patch deploy metrics-server --type=json -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'`, then `kubectl apply -f k8s/hpa.yaml`.
- **Troubleshooting.** After a Docker Desktop restart the cluster may have no DNS (`Temporary failure in name resolution`, CoreDNS logs `Plugins not ready: "kubernetes"`): `kubectl -n kube-system rollout restart deploy/coredns`. The ConfigMap sets `HF_HUB_OFFLINE=1` so a missing network cannot stall the backend start (the model is baked into the image).
- **Rolling updates** never drop below the desired ready pods (`maxUnavailable: 0`), and pods get a long termination grace period so slow agent requests can finish.
- **Langfuse** is not wired in here: the pods cannot reach a Langfuse running on your laptop by default. Leave the Langfuse keys out of the secret.
- **Real clusters.** Push the images to a registry and change `image:` and `imagePullPolicy`. The secret should come from a secrets manager, not from `.env`.
