"""Langfuse on the cluster: the optional network rule is as narrow as it claims, the secret keys are optional so tracing is
off by default, and scripts/k8s_langfuse.py enables and disables it safely. (Everything outside the script is faked.)"""

import base64
import importlib.util
import ipaddress
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
K8S = ROOT / "k8s"
spec = importlib.util.spec_from_file_location("k8s_langfuse", ROOT / "scripts" / "k8s_langfuse.py")
kl = importlib.util.module_from_spec(spec)
sys.modules["k8s_langfuse"] = kl
spec.loader.exec_module(kl)

PUBLIC, SECRET = "pk-lf-test-public-111", "sk-lf-test-secret-222"
KEYS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST")


# --- the manifests ---


def test_the_network_rule_lets_only_the_backend_reach_one_address_on_one_port():
    policy = yaml.safe_load((K8S / "optional" / "networkpolicy-langfuse-local.yaml").read_text(encoding="utf-8"))
    spec = policy["spec"]
    assert policy["metadata"]["namespace"] == "aem" and spec["podSelector"] == {"matchLabels": {"app": "backend"}}
    assert spec["policyTypes"] == ["Egress"] and "ingress" not in spec
    [rule] = spec["egress"]
    [target] = rule["to"]
    network = ipaddress.ip_network(target["ipBlock"]["cidr"])
    assert network.prefixlen == 32 and network.is_private  # one machine, on a private address
    assert rule["ports"] == [{"protocol": "TCP", "port": 3000}]


def test_the_optional_rule_is_not_applied_by_default():
    kustomization = yaml.safe_load((K8S / "kustomization.yaml").read_text(encoding="utf-8"))
    assert not any("optional" in resource or "langfuse" in resource for resource in kustomization["resources"])


def test_the_default_network_policies_still_allow_no_langfuse_address():
    text = (K8S / "networkpolicy.yaml").read_text(encoding="utf-8")
    assert "3000" not in text and "192.168" not in text  # the opt-in is the only place that names it


def test_tracing_is_off_unless_the_secret_holds_the_keys():
    backend = next(
        d for d in yaml.safe_load_all((K8S / "backend.yaml").read_text(encoding="utf-8")) if d["kind"] == "StatefulSet"
    )
    container = next(c for c in backend["spec"]["template"]["spec"]["containers"] if c["name"] == "backend")
    env = {entry["name"]: entry for entry in container["env"]}
    for name in KEYS:
        reference = env[name]["valueFrom"]["secretKeyRef"]
        assert reference["name"] == "aem-secrets" and reference["key"] == name and reference["optional"] is True
        assert "value" not in env[name]
    assert "LANGFUSE_CAPTURE_CONTENT" not in env  # question and answer text is never switched on by the manifest


def test_the_example_secret_documents_the_langfuse_keys_without_a_real_value():
    example = yaml.safe_load((K8S / "secret.example.yaml").read_text(encoding="utf-8"))["stringData"]
    assert all(example[name] == "replace-me" for name in KEYS)


def test_the_general_secret_script_does_not_copy_the_localhost_address_into_the_cluster():
    script = (ROOT / "scripts" / "k8s_secret.py").read_text(encoding="utf-8")
    assert (
        "LANGFUSE_HOST" not in script.split("OPTIONAL_KEYS")[1].split(")")[0]
    )  # inside a pod, localhost is the pod itself


def test_the_policy_address_is_read_from_the_manifest():
    assert kl.policy_address() == "192.168.65.254"


# --- the script ---


class Cluster:
    """Records every command and answers the ones the script depends on."""

    def __init__(self, **broken):
        self.broken, self.commands, self.patches, self.statuses = broken, [], [], {"http": 200}

    def kubectl(self, *args, check=True):
        self.commands.append(("kubectl",) + args)
        if "patch" in args:
            self.patches.append(json.loads(Path(args[args.index("--patch-file") + 1]).read_text(encoding="utf-8")))
        if args[:2] == ("get", "nodes"):
            return "desktop-control-plane desktop-worker desktop-worker2"
        if args[:2] == ("get", "secret"):
            return json.dumps({"data": {k: "eA==" for k in self.broken.get("present", ())}})
        if args[:2] == ("get", "networkpolicy"):
            return "networkpolicy.networking.k8s.io/backend-to-local-langfuse" if self.broken.get("policy") else ""
        return "applied"

    def sh(self, *command, check=True):
        self.commands.append(command)
        if command[:2] == ("docker", "exec") and "getent" in command:
            return self.broken.get("resolves", "192.168.65.254  host.docker.internal")
        if command[:2] == ("docker", "exec") and "curl" in command:
            return self.broken.get("node_http", "200")
        return ""


@pytest.fixture
def cluster(monkeypatch, tmp_path):
    state = Cluster()
    (tmp_path / ".env").write_text(
        f"LANGFUSE_PUBLIC_KEY={PUBLIC}\nLANGFUSE_SECRET_KEY={SECRET}\nLANGFUSE_HOST=http://localhost:3000\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(kl, "ROOT", tmp_path)
    monkeypatch.setattr(kl, "kubectl", lambda *a, check=True: state.kubectl(*a, check=check))
    monkeypatch.setattr(kl, "sh", lambda *c, check=True: state.sh(*c, check=check))
    monkeypatch.setattr(kl, "http_status", lambda url: state.statuses["http"])
    for name in KEYS:
        monkeypatch.setenv(name, "")
    return state


def order(state):
    """The kinds of step the script took, in order."""
    steps = []
    for command in state.commands:
        text = " ".join(command)
        for label, marker in (
            ("node-check", "getent"),
            ("node-curl", "curl"),
            ("apply", " apply "),
            ("patch", " patch "),
            ("restart", "rollout restart"),
            ("wait", "rollout status"),
            ("delete", " delete "),
        ):
            if marker in f" {text} ":
                steps.append(label)
    return steps


def test_enable_checks_everything_before_it_changes_the_cluster_then_restarts_the_backend(cluster, capsys):
    assert kl.main(["enable"]) == 0
    assert order(cluster) == ["node-check", "node-curl", "apply", "patch", "restart", "wait"]


def test_enable_stores_the_keys_and_an_address_the_pods_can_reach_never_localhost(cluster, capsys):
    kl.main(["enable"])
    [patch] = cluster.patches
    stored = {key: base64.b64decode(value).decode() for key, value in patch["data"].items()}
    assert stored == {
        "LANGFUSE_PUBLIC_KEY": PUBLIC,
        "LANGFUSE_SECRET_KEY": SECRET,
        "LANGFUSE_HOST": "http://host.docker.internal:3000",
    }


def test_enable_never_prints_a_key(cluster, capsys):
    kl.main(["enable"])
    out = capsys.readouterr().out
    assert PUBLIC not in out and SECRET not in out and "lf-test" not in out


def test_enable_refuses_when_langfuse_is_not_running_and_changes_nothing(cluster, capsys):
    cluster.statuses["http"] = 0
    assert kl.main(["enable"]) == 1
    assert "docker compose" in capsys.readouterr().out and order(cluster) == []


def test_enable_refuses_when_the_nodes_see_a_different_host_address_than_the_rule_allows(cluster, capsys):
    cluster.broken["resolves"] = "172.17.0.1  host.docker.internal"
    assert kl.main(["enable"]) == 1
    out = capsys.readouterr().out
    assert "172.17.0.1" in out and "192.168.65.254" in out
    assert "apply" not in order(cluster) and "patch" not in order(cluster)


def test_enable_refuses_when_a_node_cannot_reach_langfuse(cluster, capsys):
    cluster.broken["node_http"] = "000"
    assert kl.main(["enable"]) == 1
    assert "patch" not in order(cluster) and "restart" not in order(cluster)


def test_enable_refuses_without_keys(cluster, tmp_path, capsys):
    (tmp_path / ".env").write_text("LANGFUSE_HOST=http://localhost:3000\n", encoding="utf-8")
    assert kl.main(["enable"]) == 1
    assert "LANGFUSE_PUBLIC_KEY" in capsys.readouterr().out and cluster.commands == []


def test_a_hosted_langfuse_needs_no_network_rule_and_no_node_check(cluster, capsys):
    assert kl.main(["enable", "--pod-host", "https://cloud.example.org"]) == 0
    assert order(cluster) == ["patch", "restart", "wait"]
    assert base64.b64decode(cluster.patches[0]["data"]["LANGFUSE_HOST"]).decode() == "https://cloud.example.org"


def test_a_plain_http_address_outside_this_machine_is_refused(cluster, capsys):
    assert kl.main(["enable", "--pod-host", "http://example.org:3000"]) == 1
    assert cluster.commands == []


def test_disable_removes_the_rule_and_the_keys_and_restarts(cluster, capsys):
    assert kl.main(["disable"]) == 0
    assert order(cluster) == ["delete", "patch", "restart", "wait"]
    assert cluster.patches[0] == {"data": {name: None for name in KEYS}}  # null removes a key in a merge patch


def test_status_reports_whether_tracing_is_on_without_changing_anything(cluster, capsys):
    cluster.broken.update(present=KEYS, policy=True)
    assert kl.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "tracing is ON" in out and "applied" in out and "running" in out
    assert order(cluster) == []
    cluster.broken.update(present=())
    kl.main(["status"])
    assert "tracing is off" in capsys.readouterr().out


def test_the_patch_file_with_the_keys_is_removed_afterwards(cluster):
    kl.main(["enable"])
    patch_files = [c[c.index("--patch-file") + 1] for c in cluster.commands if "--patch-file" in c]
    assert patch_files and not any(Path(p).exists() for p in patch_files)


def test_local_addresses_are_recognised():
    assert kl.is_local("http://host.docker.internal:3000") and kl.is_local("http://localhost:3000")
    assert not kl.is_local("https://cloud.langfuse.com")
