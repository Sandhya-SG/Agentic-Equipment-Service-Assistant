"""The Kubernetes manifests: the backend keeps its logs on a persistent volume per replica, and nothing points at the old
Deployment. (Parsed as YAML; no cluster needed.)"""

from pathlib import Path

import pytest
import yaml

K8S = Path(__file__).resolve().parents[2] / "k8s"


def load(name):
    return list(yaml.safe_load_all((K8S / name).read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def backend():
    return next(doc for doc in load("backend.yaml") if doc["kind"] == "StatefulSet")


def test_the_backend_is_a_statefulset_so_each_pod_keeps_its_own_logs(backend):
    assert backend["metadata"]["name"] == "backend"
    assert backend["spec"]["serviceName"] == "backend"
    assert backend["spec"]["replicas"] >= 2
    assert backend["spec"]["updateStrategy"]["type"] == "RollingUpdate"


def test_every_replica_gets_a_persistent_volume_for_its_logs(backend):
    [claim] = backend["spec"]["volumeClaimTemplates"]
    assert claim["metadata"]["name"] == "logs"
    assert claim["spec"]["accessModes"] == ["ReadWriteOnce"]
    assert claim["spec"]["resources"]["requests"]["storage"]
    retention = backend["spec"]["persistentVolumeClaimRetentionPolicy"]
    assert retention == {"whenDeleted": "Retain", "whenScaled": "Retain"}


def test_the_logs_are_mounted_from_that_volume_and_not_from_an_emptydir(backend):
    pod = backend["spec"]["template"]["spec"]
    container = next(c for c in pod["containers"] if c["name"] == "backend")
    assert {"name": "logs", "mountPath": "/app/logs"} in container["volumeMounts"]
    assert "logs" not in {volume["name"] for volume in pod["volumes"]}  # it is the claim template, not an emptyDir


def test_the_non_root_user_can_write_to_the_volume(backend):
    context = backend["spec"]["template"]["spec"]["securityContext"]
    assert context["runAsNonRoot"] is True and context["fsGroup"] == context["runAsUser"] == 10001


def test_the_audit_log_is_still_one_file_per_replica():
    config = load("configmap.yaml")[0]["data"]
    assert config["AUDIT_LOG_PER_INSTANCE"] == "true"  # two pods never write the same hash chain


def test_the_autoscaler_targets_the_statefulset_and_nothing_still_names_a_backend_deployment():
    target = load("hpa.yaml")[0]["spec"]["scaleTargetRef"]
    assert (target["kind"], target["name"]) == ("StatefulSet", "backend")
    for name in ("backend.yaml", "frontend.yaml", "kustomization.yaml", "hpa.yaml"):
        for doc in load(name):
            if doc.get("kind") == "Deployment":
                assert doc["metadata"]["name"] != "backend"


def test_the_backend_service_still_selects_the_backend_pods():
    service = next(doc for doc in load("backend.yaml") if doc["kind"] == "Service")
    assert service["spec"]["selector"] == {"app": "backend"}
    assert service["metadata"]["name"] == "backend"


def test_the_review_token_comes_from_the_secret_and_is_optional(backend):
    container = next(c for c in backend["spec"]["template"]["spec"]["containers"] if c["name"] == "backend")
    env = {e["name"]: e for e in container["env"]}
    assert env["REVIEWER_TOKEN"]["valueFrom"]["secretKeyRef"]["key"] == "REVIEWER_TOKEN"
    assert env["REVIEWER_TOKEN"]["valueFrom"]["secretKeyRef"]["optional"] is True  # unset: the review API answers 503


def test_the_review_token_is_never_in_a_manifest_or_the_compose_file_as_a_value():
    root = K8S.parent
    for name in ("k8s/backend.yaml", "k8s/configmap.yaml", "docker-compose.yml"):
        text = (root / name).read_text(encoding="utf-8")
        for line in text.splitlines():
            if "REVIEWER_TOKEN" in line and ":" in line and "${REVIEWER_TOKEN" not in line:
                assert "secretKeyRef" in line or line.strip().startswith(("#", "- name:")), line
    example = (K8S / "secret.example.yaml").read_text(encoding="utf-8")
    assert 'REVIEWER_TOKEN: "replace-me"' in example


def test_the_cluster_has_the_cost_controls_and_the_ledger_is_on_the_writable_log_volume():
    config = load("configmap.yaml")[0]["data"]
    assert float(config["OPENAI_SPEND_CAP_USD"]) > 0  # an estimated-spend cap is on by default
    assert config["OPENAI_SPEND_LEDGER"].startswith("/app/logs/")  # the root filesystem is read-only
    assert int(config["RATE_LIMIT_PER_MINUTE"]) > 0
    assert config["TRUST_FORWARDED_FOR"] == "true"  # safe only because the network policy admits just the frontend
    ingress = [rule for doc in load("networkpolicy.yaml") for rule in doc.get("spec", {}).get("ingress", [])]
    from_frontend = {"podSelector": {"matchLabels": {"app": "frontend"}}}
    assert any(from_frontend in rule.get("from", []) for rule in ingress)


def test_the_secret_script_can_store_the_review_token():
    script = (K8S.parent / "scripts" / "k8s_secret.py").read_text(encoding="utf-8")
    assert "REVIEWER_TOKEN" in script
