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
