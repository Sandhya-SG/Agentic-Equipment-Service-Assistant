"""The cluster manifests as a whole: every container hardened and bounded, probes that hit real routes, Services that
select the right pods, the network closed except for what the app needs, and settings the code actually reads.
(Parsed as YAML; no cluster is needed. `kubectl kustomize` is used when it is installed.)"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
K8S = ROOT / "k8s"
NAMESPACE = "aem"


def load(name):
    return [doc for doc in yaml.safe_load_all((K8S / name).read_text(encoding="utf-8")) if doc]


def everything():
    kustomization = yaml.safe_load((K8S / "kustomization.yaml").read_text(encoding="utf-8"))
    docs = []
    for resource in kustomization["resources"] + ["hpa.yaml"]:
        docs += load(resource)
    return docs


def of_kind(kind):
    return [doc for doc in everything() if doc["kind"] == kind]


def workloads():
    return of_kind("StatefulSet") + of_kind("Deployment")


def containers(workload):
    spec = workload["spec"]["template"]["spec"]
    return [(c, "init") for c in spec.get("initContainers", [])] + [(c, "main") for c in spec["containers"]]


ALL_CONTAINERS = [(w["metadata"]["name"], c["name"], c, kind) for w in workloads() for c, kind in containers(w)]
IDS = [f"{workload}/{name}" for workload, name, _, _ in ALL_CONTAINERS]


# --- every container ---


@pytest.mark.parametrize("workload, name, container, kind", ALL_CONTAINERS, ids=IDS)
def test_every_container_is_hardened(workload, name, container, kind):
    context = container["securityContext"]
    assert context["allowPrivilegeEscalation"] is False
    assert context["readOnlyRootFilesystem"] is True
    assert context["capabilities"]["drop"] == ["ALL"]
    assert not context.get("privileged") and "add" not in context["capabilities"]


@pytest.mark.parametrize("workload, name, container, kind", ALL_CONTAINERS, ids=IDS)
def test_every_container_has_requests_and_limits(workload, name, container, kind):
    resources = container["resources"]
    assert resources["requests"]["cpu"] and resources["requests"]["memory"]
    assert resources["limits"]["memory"], "a memory limit stops one pod taking the node"
    assert resources["limits"]["ephemeral-storage"], "a disk limit stops a runaway log or temp file"


@pytest.mark.parametrize("workload, name, container, kind", ALL_CONTAINERS, ids=IDS)
def test_no_container_uses_a_moving_image_tag(workload, name, container, kind):
    image = container["image"]
    assert ":" in image and not image.endswith(":latest"), image
    assert container["imagePullPolicy"] == "IfNotPresent"


@pytest.mark.parametrize("workload, name, container, kind", ALL_CONTAINERS, ids=IDS)
def test_no_container_carries_a_secret_as_a_literal_value(workload, name, container, kind):
    for entry in container.get("env", []):
        if any(word in entry["name"] for word in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            assert "value" not in entry and "secretKeyRef" in entry["valueFrom"], entry["name"]


# --- every pod ---


@pytest.mark.parametrize("workload", workloads(), ids=lambda w: w["metadata"]["name"])
def test_every_pod_runs_non_root_with_the_default_seccomp_profile_and_no_api_token(workload):
    pod = workload["spec"]["template"]["spec"]
    assert pod["securityContext"]["runAsNonRoot"] is True and pod["securityContext"]["runAsUser"] == 10001
    assert pod["securityContext"]["seccompProfile"] == {"type": "RuntimeDefault"}
    assert pod["automountServiceAccountToken"] is False  # the app never calls the Kubernetes API
    assert (
        pod["enableServiceLinks"] is False
    )  # injected FRONTEND_PORT / BACKEND_PORT would collide with the app's settings
    assert "hostNetwork" not in pod and "hostPID" not in pod and "hostPath" not in str(pod.get("volumes", []))


@pytest.mark.parametrize("workload", workloads(), ids=lambda w: w["metadata"]["name"])
def test_every_workload_has_two_replicas_a_rolling_update_and_matching_labels(workload):
    assert workload["spec"]["replicas"] >= 2
    template_labels = workload["spec"]["template"]["metadata"]["labels"]
    assert workload["spec"]["selector"]["matchLabels"].items() <= template_labels.items()
    strategy = workload["spec"].get("strategy") or workload["spec"]["updateStrategy"]
    assert strategy["type"] == "RollingUpdate"


def test_the_frontend_rollout_never_takes_a_pod_away_before_a_new_one_is_ready():
    [frontend] = [w for w in workloads() if w["metadata"]["name"] == "frontend"]
    rolling = frontend["spec"]["strategy"]["rollingUpdate"]
    assert rolling["maxUnavailable"] == 0 and rolling["maxSurge"] >= 1


# --- probes ---


def declared_routes():
    """Every route path the backend declares, read from the source."""
    text = " ".join(
        p.read_text(encoding="utf-8") for p in [ROOT / "app" / "main.py", *(ROOT / "app" / "routes").glob("*.py")]
    )
    return set(re.findall(r'@(?:app|router)\.(?:get|post)\("([^"]+)"', text))


def test_the_backend_probes_hit_routes_the_backend_really_has_and_the_right_kind():
    [backend] = [w for w in workloads() if w["metadata"]["name"] == "backend"]
    main = next(c for c, kind in containers(backend) if c["name"] == "backend")
    routes = declared_routes()
    assert main["startupProbe"]["httpGet"]["path"] == "/ready" and main["readinessProbe"]["httpGet"]["path"] == "/ready"
    assert main["livenessProbe"]["httpGet"]["path"] == "/health"  # liveness must not depend on the slow agents
    for probe in ("startupProbe", "readinessProbe", "livenessProbe"):
        assert main[probe]["httpGet"]["path"] in routes, probe
        assert main[probe]["httpGet"]["port"] == main["ports"][0]["name"]


def test_the_backend_may_take_up_to_a_few_minutes_to_load_before_it_is_restarted():
    [backend] = [w for w in workloads() if w["metadata"]["name"] == "backend"]
    main = next(c for c, kind in containers(backend) if c["name"] == "backend")
    startup = main["startupProbe"]
    budget = startup["periodSeconds"] * startup["failureThreshold"]
    assert 120 <= budget <= 400, budget  # torch, the model and the index take a while; a hang must still be restarted


def test_the_frontend_has_readiness_and_liveness_on_the_page_it_serves():
    [frontend] = [w for w in workloads() if w["metadata"]["name"] == "frontend"]
    main = next(c for c, kind in containers(frontend) if c["name"] == "frontend")
    assert main["readinessProbe"]["httpGet"]["path"] == "/" and main["livenessProbe"]["httpGet"]["path"] == "/"
    assert main["readinessProbe"]["httpGet"]["port"] == main["ports"][0]["name"]


def test_the_backend_pod_outlives_a_slow_request_when_it_is_stopped():
    [backend] = [w for w in workloads() if w["metadata"]["name"] == "backend"]
    assert backend["spec"]["template"]["spec"]["terminationGracePeriodSeconds"] >= 60


# --- services and the connections between them ---


def test_each_service_selects_its_pods_and_points_at_a_named_container_port():
    for service in of_kind("Service"):
        name = service["metadata"]["name"]
        [workload] = [w for w in workloads() if w["metadata"]["name"] == name]
        assert service["spec"]["selector"].items() <= workload["spec"]["template"]["metadata"]["labels"].items()
        ports = {p["name"]: p["containerPort"] for c, _ in containers(workload) for p in c.get("ports", [])}
        assert service["spec"]["ports"][0]["targetPort"] in ports
        assert service["spec"]["type"] == "ClusterIP"  # nothing is exposed outside the cluster by default


def test_the_frontend_is_configured_to_call_the_backend_service_on_its_real_port():
    config = load("configmap.yaml")[0]["data"]
    [service] = [s for s in of_kind("Service") if s["metadata"]["name"] == "backend"]
    assert config["BACKEND_URL"] == f"http://backend:{service['spec']['ports'][0]['port']}"
    assert float(config["BACKEND_TIMEOUT"]) >= 60  # the agents are slow


def test_the_statefulset_names_the_service_that_governs_it():
    [backend] = of_kind("StatefulSet")
    assert backend["spec"]["serviceName"] in {s["metadata"]["name"] for s in of_kind("Service")}


# --- network ---


def policies():
    return {p["metadata"]["name"]: p for p in of_kind("NetworkPolicy")}


def test_all_traffic_is_denied_unless_a_policy_allows_it():
    deny = policies()["default-deny"]["spec"]
    assert deny["podSelector"] == {} and set(deny["policyTypes"]) == {"Ingress", "Egress"}
    assert "ingress" not in deny and "egress" not in deny


def test_the_only_ways_in_are_the_browser_to_the_frontend_and_the_frontend_to_the_backend():
    frontend = policies()["frontend"]["spec"]
    backend = policies()["backend"]["spec"]
    assert frontend["ingress"] == [{"ports": [{"protocol": "TCP", "port": 5000}]}]
    assert backend["ingress"][0]["from"] == [{"podSelector": {"matchLabels": {"app": "frontend"}}}]
    assert backend["ingress"][0]["ports"] == [{"protocol": "TCP", "port": 8000}]


def test_the_frontend_can_reach_only_the_backend_and_the_backend_only_port_443_and_dns():
    frontend = policies()["frontend"]["spec"]["egress"]
    assert frontend == [
        {"to": [{"podSelector": {"matchLabels": {"app": "backend"}}}], "ports": [{"protocol": "TCP", "port": 8000}]}
    ]
    backend = policies()["backend"]["spec"]["egress"]
    assert backend == [
        {"ports": [{"protocol": "TCP", "port": 443}]}
    ]  # the model provider (and Langfuse Cloud if enabled)
    dns = policies()["allow-dns"]["spec"]["egress"][0]
    assert {p["port"] for p in dns["ports"]} == {53}


def test_the_ports_in_the_policies_match_the_ports_the_containers_listen_on():
    ports = {
        w["metadata"]["name"]: [p["containerPort"] for c, _ in containers(w) for p in c.get("ports", [])]
        for w in workloads()
    }
    assert ports == {"backend": [8000], "frontend": [5000]}


# --- namespace and autoscaler ---


def test_everything_is_in_one_namespace_that_enforces_the_restricted_pod_standard():
    for doc in everything():
        if doc["kind"] != "Namespace":
            assert doc["metadata"]["namespace"] == NAMESPACE, doc["metadata"]["name"]
    [namespace] = of_kind("Namespace")
    assert namespace["metadata"]["labels"]["pod-security.kubernetes.io/enforce"] == "restricted"


def test_the_autoscaler_has_sane_bounds_and_scales_a_workload_that_exists():
    [hpa] = of_kind("HorizontalPodAutoscaler")
    spec = hpa["spec"]
    target = spec["scaleTargetRef"]
    assert any(w["kind"] == target["kind"] and w["metadata"]["name"] == target["name"] for w in workloads())
    assert 2 <= spec["minReplicas"] <= spec["maxReplicas"] <= 10
    assert spec["metrics"][0]["resource"]["target"]["averageUtilization"] <= 80
    assert spec["behavior"]["scaleDown"]["stabilizationWindowSeconds"] >= 60  # no flapping while slow requests finish
    [backend] = of_kind("StatefulSet")
    assert backend["spec"]["replicas"] >= spec["minReplicas"]


def test_the_cpu_request_exists_so_the_autoscaler_has_something_to_measure():
    [backend] = of_kind("StatefulSet")
    main = next(c for c, kind in containers(backend) if c["name"] == "backend")
    assert main["resources"]["requests"]["cpu"]


# --- settings: names the code really reads ---


def names_read_by_the_code():
    found = set()
    pattern = re.compile(
        r'(?:getenv|environ\.get|environ\[)\(?\s*["\']([A-Z][A-Z0-9_]+)["\']|_env_(?:float|int|bool)\(\s*["\']([A-Z][A-Z0-9_]+)["\']'
    )
    for folder in ("app", "src/asa", "frontend"):
        for path in (ROOT / folder).rglob("*.py"):
            for match in pattern.finditer(path.read_text(encoding="utf-8")):
                found.add(match.group(1) or match.group(2))
    return found


LIBRARY_SETTINGS = {"HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HOME"}  # read by Hugging Face, not by this code


def test_every_setting_in_the_configmap_is_one_the_code_or_a_library_reads():
    read = names_read_by_the_code() | LIBRARY_SETTINGS
    unknown = set(load("configmap.yaml")[0]["data"]) - read
    assert not unknown, f"settings nobody reads (a typo?): {sorted(unknown)}"


def test_every_environment_variable_the_pods_set_is_one_the_code_reads():
    read = names_read_by_the_code() | LIBRARY_SETTINGS
    for workload in workloads():
        for container, _ in containers(workload):
            names = {entry["name"] for entry in container.get("env", [])}
            assert not names - read, f"{container['name']}: {sorted(names - read)}"


def test_the_secret_keys_the_pods_ask_for_are_the_ones_the_example_and_the_script_provide():
    asked = set()
    for workload in workloads():
        for container, _ in containers(workload):
            for entry in container.get("env", []):
                reference = entry.get("valueFrom", {}).get("secretKeyRef")
                if reference:
                    assert reference["name"] == "aem-secrets"
                    asked.add(reference["key"])
    example = yaml.safe_load((K8S / "secret.example.yaml").read_text(encoding="utf-8"))
    assert asked <= set(example["stringData"])
    assert all(value == "replace-me" for value in example["stringData"].values())  # the example holds no real value
    script = (ROOT / "scripts" / "k8s_secret.py").read_text(encoding="utf-8")
    assert all(key in script for key in asked)


def test_a_secret_is_never_part_of_what_is_applied_from_git():
    kinds = {doc["kind"] for doc in everything()}
    assert "Secret" not in kinds  # created by scripts/k8s_secret.py, so no key is in the repository


# --- kustomize ---


def test_every_resource_the_kustomization_lists_exists_and_its_images_are_the_ones_the_manifests_use():
    kustomization = yaml.safe_load((K8S / "kustomization.yaml").read_text(encoding="utf-8"))
    for resource in kustomization["resources"]:
        assert (K8S / resource).is_file(), resource
    used = {c["image"].split(":")[0] for w in workloads() for c, _ in containers(w)}
    assert {image["name"] for image in kustomization["images"]} == used


@pytest.mark.skipif(shutil.which("kubectl") is None, reason="kubectl is not installed")
def test_the_kustomization_renders_and_every_rendered_object_is_in_the_namespace():
    rendered = subprocess.run(["kubectl", "kustomize", str(K8S)], capture_output=True, text=True, check=True).stdout
    docs = [doc for doc in yaml.safe_load_all(rendered) if doc]
    assert {doc["kind"] for doc in docs} >= {
        "Namespace",
        "ConfigMap",
        "StatefulSet",
        "Deployment",
        "Service",
        "NetworkPolicy",
    }
    for doc in docs:
        if doc["kind"] != "Namespace":
            assert doc["metadata"]["namespace"] == NAMESPACE
