# Multi-pod session check

Pending clarification state lives in the memory of one backend process (`app/services/clarification_store.py`). With
two backend pods, a follow-up that reaches the other pod loses it. This check measures how often, on the cluster, with
no OpenAI call and no cost.

How it works: a stand-in for the OpenAI API (`stub_openai.py`) always answers "CLARIFY". The backend escalates a
conversation on its third message (`MAX_CLARIFICATIONS = 2`) only if one pod saw all of its messages. The client
(`session_client.py`) sends 20 conversations of 4 messages, each message on a new connection, and counts at which
message each conversation escalated. On one pod that is message 3 every time.

```
kubectl -n aem create configmap mp-test --from-file=scripts/multipod_session_test/stub_openai.py --from-file=scripts/multipod_session_test/session_client.py
kubectl apply -f scripts/multipod_session_test/stub.yaml
kubectl -n aem set env deploy/backend OPENAI_BASE_URL=http://stub-openai:8080/v1
kubectl -n aem rollout status deploy/backend
kubectl apply -f scripts/multipod_session_test/client-pod.yaml
kubectl -n aem logs mp-client                         # the SUMMARY line

# clean up (important: the backend must not keep pointing at the stub)
kubectl -n aem delete pod mp-client stub-openai; kubectl -n aem delete svc stub-openai; kubectl -n aem delete configmap mp-test
kubectl -n aem delete networkpolicy mp-test-stub mp-test-backend-to-stub
kubectl -n aem set env deploy/backend OPENAI_BASE_URL-
```

The pods in the two YAML files use the backend image tag of `k8s/kustomization.yaml` (`aem-backend:2f0cc72`); change it
to the tag you deployed. `session_client.py` carries the signed state like the browser does; run it with `CARRY_STATE=0` (an env entry in
`client-pod.yaml`) to reproduce the original problem. In Git Bash prefix paths like `/app/logs` with `MSYS_NO_PATHCONV=1`. To test another setting, change the Service or the
replica count and delete `mp-client` before running it again. Results: `docs/deployment-verification.md`.
