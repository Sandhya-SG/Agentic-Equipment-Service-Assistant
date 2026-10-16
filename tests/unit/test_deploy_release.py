"""The deploy script: approved tags only, digests not tags, every pod checked, free smoke tests. No cluster is used."""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("deploy_release", ROOT / "scripts" / "deploy_release.py")
dr = importlib.util.module_from_spec(spec)
sys.modules["deploy_release"] = dr
spec.loader.exec_module(dr)

BACKEND = "sha256:" + "a" * 64
FRONTEND = "sha256:" + "b" * 64
DIGESTS = {"backend": BACKEND, "frontend": FRONTEND}


# --- which tags are accepted ---


def test_an_approved_release_tag_is_accepted():
    assert dr.parse_ref("release-f049973") == ("release-f049973", "f049973")
    assert dr.parse_ref("release-" + "a" * 40)[1] == "a" * 40


def test_an_unapproved_sha_tag_needs_the_explicit_flag():
    with pytest.raises(dr.DeployError, match="has not been approved"):
        dr.parse_ref("sha-f049973")
    assert dr.parse_ref("sha-f049973", unapproved=True) == ("sha-f049973", "f049973")


@pytest.mark.parametrize(
    "ref",
    [
        "latest",
        "main",
        "v0.10",
        "release-",
        "release-XYZ",
        "release-abc",
        "sha-f049973; rm -rf",
        "",
        None,
        "RELEASE-f049973",
    ],
)
def test_a_moving_or_malformed_tag_is_always_refused(ref):
    for unapproved in (False, True):
        with pytest.raises(dr.DeployError):
            dr.parse_ref(ref, unapproved)


# --- digests and the pinned manifest ---


def test_the_digest_is_read_from_the_registry_output():
    output = f"Name:      ghcr.io/o/asa-backend:sha-abc1234\nMediaType: application/vnd.docker.distribution.manifest.v2+json\nDigest:    {BACKEND}\n"
    assert dr.parse_digest(output) == BACKEND
    with pytest.raises(dr.DeployError):
        dr.parse_digest("unauthorized")


def test_image_names_are_lowercase_registry_paths():
    assert dr.image_name("Sandhya-SG", "backend") == "ghcr.io/sandhya-sg/asa-backend"


def test_a_value_that_is_not_a_digest_is_never_written_into_the_manifest():
    for bad in ("latest", "sha256:short", BACKEND + "\n  - name: evil"):
        with pytest.raises(dr.DeployError):
            dr.render_images_block("o", {"backend": bad, "frontend": FRONTEND})


def test_the_images_block_is_replaced_and_the_rest_of_the_file_is_kept():
    original = (ROOT / "k8s" / "kustomization.yaml").read_text(encoding="utf-8")
    pinned = dr.pinned_kustomization(original, "Sandhya-SG", DIGESTS)
    assert "newTag:" not in pinned and f"digest: {BACKEND}" in pinned and f"digest: {FRONTEND}" in pinned
    assert "newName: ghcr.io/sandhya-sg/asa-backend" in pinned
    assert pinned.startswith(original[: original.rfind("\nimages:") + 1])
    assert "resources:" in pinned and "networkpolicy.yaml" in pinned


def test_a_kustomization_without_an_images_block_is_refused():
    with pytest.raises(dr.DeployError):
        dr.pinned_kustomization("resources: []\n", "o", DIGESTS)


@pytest.mark.skipif(shutil.which("kubectl") is None, reason="kubectl is not installed")
def test_the_pinned_manifest_renders_the_digest_on_every_container_including_the_init_container(tmp_path):
    target = tmp_path / "k8s"
    shutil.copytree(ROOT / "k8s", target)
    kustomization = target / "kustomization.yaml"
    kustomization.write_text(
        dr.pinned_kustomization(kustomization.read_text(encoding="utf-8"), "Sandhya-SG", DIGESTS), encoding="utf-8"
    )
    rendered = subprocess.run(["kubectl", "kustomize", str(target)], capture_output=True, text=True, check=True).stdout
    images = [line.split("image:", 1)[1].strip() for line in rendered.splitlines() if line.strip().startswith("image:")]
    assert images.count(f"ghcr.io/sandhya-sg/asa-backend@{BACKEND}") == 2  # the init container uses the same image
    assert images.count(f"ghcr.io/sandhya-sg/asa-frontend@{FRONTEND}") == 1
    assert not any(":k8s" in image or ":latest" in image for image in images)


# --- what the pods really run ---


def pod(name, service, image_id):
    return {"metadata": {"name": name}, "status": {"containerStatuses": [{"name": service, "imageID": image_id}]}}


def test_every_pod_must_run_the_approved_digest():
    good = {
        "items": [
            pod("backend-0", "backend", f"ghcr.io/o/asa-backend@{BACKEND}"),
            pod("backend-1", "backend", f"ghcr.io/o/asa-backend@{BACKEND}"),
        ]
    }
    assert dr.digest_mismatches(good, "backend", BACKEND) == []


def test_one_pod_on_an_older_image_is_reported():
    mixed = {
        "items": [
            pod("backend-0", "backend", f"ghcr.io/o/asa-backend@{BACKEND}"),
            pod("backend-1", "backend", "ghcr.io/o/asa-backend@sha256:" + "c" * 64),
        ]
    }
    [problem] = dr.digest_mismatches(mixed, "backend", BACKEND)
    assert "backend" in problem and "c" * 64 in problem


def test_a_locally_built_image_has_no_registry_digest_and_does_not_pass():
    local = {
        "items": [
            pod("backend-0", "backend", "sha256:2f3e545142cf7115c023bfbdffe8f79f2a1ade07505dd7b262df6aafca3678b6")
        ]
    }
    [problem] = dr.digest_mismatches(local, "backend", BACKEND)
    assert "without a registry digest" in problem


def test_no_pods_at_all_is_a_failure_not_a_pass():
    assert dr.digest_mismatches({"items": []}, "backend", BACKEND) == ["backend: no running pods found"]


# --- the smoke checks ---


def fake_http(monkeypatch, answers):
    calls = []

    def http(method, url, body=None, headers=None):
        calls.append((method, url.split("18000")[-1] if "18000" in url else url, body, headers))
        return answers[(method, url.rsplit("localhost", 1)[-1].split(":", 1)[-1].split("/", 1)[-1])]

    monkeypatch.setattr(dr, "http", http)
    return calls


GOOD = {
    ("GET", "health"): (200, {}),
    ("GET", "ready"): (200, {}),
    ("GET", "version"): (200, {"git": {"commit": "f049973d375ef72e2f4b941b951eeba3293225ca"}}),
    ("POST", "api/chat"): (400, {"status": "blocked"}),
}


def sequenced(monkeypatch, version=None, injection=(400, {"status": "blocked"}), empty=(422, {}), ready=(200, {})):
    chat = iter([injection, empty])

    def http(method, url, body=None, headers=None):
        path = url.split("://", 1)[1].split("/", 1)[1] if "/" in url.split("://", 1)[1] else ""
        if path == "health":
            return 200, {}
        if path == "ready":
            return ready
        if path == "version":
            return version or (200, {"git": {"commit": "f049973d375ef72e2f4b941b951eeba3293225ca"}})
        return next(chat)

    monkeypatch.setattr(dr, "http", http)


def test_the_smoke_checks_pass_on_a_healthy_matching_backend(monkeypatch):
    sequenced(monkeypatch)
    assert dr.smoke_checks("http://localhost:18000", "f049973", "tok") == []


def test_a_wrong_commit_is_a_failure(monkeypatch):
    sequenced(monkeypatch, version=(200, {"git": {"commit": "0123456789abcdef"}}))
    assert any("expected f049973" in p for p in dr.smoke_checks("http://x", "f049973", None))


def test_an_unblocked_injection_is_a_failure(monkeypatch):
    sequenced(monkeypatch, injection=(200, {"status": "ok"}))
    assert any("not blocked" in p for p in dr.smoke_checks("http://x", "f049973", None))


def test_a_503_is_not_taken_as_a_blocked_injection(monkeypatch):
    sequenced(monkeypatch, injection=(503, {"status": "unavailable"}))
    assert any("not blocked" in p for p in dr.smoke_checks("http://x", "f049973", None))


def test_a_backend_that_is_not_ready_is_a_failure(monkeypatch):
    sequenced(monkeypatch, ready=(503, {}))
    assert any("/ready" in p for p in dr.smoke_checks("http://x", "f049973", None))


def test_the_smoke_checks_make_no_model_call(monkeypatch):
    """Only the blocked injection and an empty message post to the chat endpoint: both are decided before any model."""
    posted = []

    def http(method, url, body=None, headers=None):
        if method == "POST":
            posted.append(body)
        return 400, {"status": "blocked"}

    monkeypatch.setattr(dr, "http", http)
    dr.smoke_checks("http://x", "f049973", None)
    assert posted == [{"message": dr.INJECTION_PROBE, "equipment_model": "thermal_station"}, {"message": ""}]


# --- the command line ---


def test_without_apply_nothing_is_changed(monkeypatch, capsys):
    commands = []

    def fake_run(command, check=True):
        commands.append(command)
        if command[:3] == ["docker", "buildx", "imagetools"]:
            return f"Digest:    {BACKEND}\n"
        return "aem-backend:216440f"

    monkeypatch.setattr(dr, "run", fake_run)
    monkeypatch.setattr(dr, "find_kubectl", lambda: "kubectl")
    assert dr.main(["release-f049973", "--owner", "Sandhya-SG"]) == 0
    assert "dry run" in capsys.readouterr().out
    assert not any("apply" in c or "undo" in c for command in commands for c in command)


def test_an_unapproved_tag_is_refused_before_anything_is_contacted(monkeypatch, capsys):
    monkeypatch.setattr(dr, "run", lambda *a, **k: pytest.fail("nothing may run"))
    assert dr.main(["sha-f049973", "--owner", "o"]) == 1
    assert "has not been approved" in capsys.readouterr().out


def test_latest_is_refused(monkeypatch, capsys):
    monkeypatch.setattr(dr, "run", lambda *a, **k: pytest.fail("nothing may run"))
    assert dr.main(["latest", "--apply"]) == 1


def test_a_tag_is_required_unless_rolling_back():
    with pytest.raises(SystemExit):
        dr.main([])


# --- applying, recording and rolling back (the cluster and the registry are faked) ---


@pytest.fixture
def cluster(monkeypatch, tmp_path):
    """Fake everything outside the script: the registry, kubectl, the pods and the smoke test."""
    state = {
        "commands": [],
        "applied_kustomization": None,
        "recorded": [],
        "order": [],
        "pods": None,
        "smoke": [],
        "wait": None,
    }

    def run(command, check=True):
        state["commands"].append(command)
        if command[:2] == ["kubectl", "apply"]:
            state["order"].append("apply")
            state["applied_kustomization"] = (Path(command[-1]) / "kustomization.yaml").read_text(encoding="utf-8")
        if "undo" in command:
            state["order"].append("undo " + command[-1])
        return "applied"

    def wait(kubectl):
        state["order"].append("wait")
        if state["wait"]:
            raise dr.DeployError(state["wait"])

    def pods(kubectl, service):
        state["order"].append(f"pods {service}")
        digest = DIGESTS[service]
        return state["pods"] or {"items": [pod("p-0", service, f"ghcr.io/o/asa-{service}@{digest}")]}

    def smoke(kubectl, commit):
        state["order"].append("smoke")
        return state["smoke"]

    monkeypatch.setattr(dr, "run", run)
    monkeypatch.setattr(dr, "find_kubectl", lambda: "kubectl")
    monkeypatch.setattr(dr, "resolve_digest", lambda owner, service, tag: DIGESTS[service])
    monkeypatch.setattr(dr, "current_images", lambda kubectl: {"backend": "old-backend", "frontend": "old-frontend"})
    monkeypatch.setattr(dr, "wait_for_rollouts", wait)
    monkeypatch.setattr(dr, "pods", pods)
    monkeypatch.setattr(dr, "smoke_test", smoke)
    monkeypatch.setattr(dr, "record", lambda entry: state["recorded"].append(entry))
    return state


def test_a_good_deployment_applies_then_waits_then_checks_the_digests_then_smoke_tests_then_records_ok(cluster, capsys):
    assert dr.main(["release-f049973", "--owner", "Sandhya-SG", "--apply"]) == 0
    assert cluster["order"] == ["apply", "wait", "pods backend", "pods frontend", "smoke"]
    [entry] = cluster["recorded"]
    assert entry["result"] == "ok" and entry["approved"] is True and entry["digests"] == DIGESTS
    assert entry["previous"] == {"backend": "old-backend", "frontend": "old-frontend"} and entry["problems"] == []
    assert "OK:" in capsys.readouterr().out


def test_what_is_applied_is_a_copy_pinned_to_the_digests_and_the_repository_file_is_untouched(cluster):
    before = (ROOT / "k8s" / "kustomization.yaml").read_text(encoding="utf-8")
    dr.main(["release-f049973", "--owner", "Sandhya-SG", "--apply"])
    applied = cluster["applied_kustomization"]
    assert f"digest: {BACKEND}" in applied and f"digest: {FRONTEND}" in applied and "newTag:" not in applied
    assert (ROOT / "k8s" / "kustomization.yaml").read_text(encoding="utf-8") == before


def test_a_pod_on_the_wrong_image_fails_the_deployment_records_it_and_names_the_rollback(cluster, capsys):
    cluster["pods"] = {"items": [pod("backend-0", "backend", "ghcr.io/o/asa-backend@sha256:" + "d" * 64)]}
    assert dr.main(["release-f049973", "--owner", "Sandhya-SG", "--apply"]) == 1
    assert "smoke" not in cluster["order"]  # a wrong image is not smoke tested as if it were right
    assert cluster["recorded"][0]["result"] == "failed" and "expected" in cluster["recorded"][0]["problems"][0]
    assert "--rollback" in capsys.readouterr().out


def test_a_failed_smoke_check_fails_the_deployment_and_is_recorded(cluster, capsys):
    cluster["smoke"] = ["the injection probe was not blocked (HTTP 200, status 'ok')"]
    assert dr.main(["release-f049973", "--owner", "Sandhya-SG", "--apply"]) == 1
    assert cluster["recorded"][0]["result"] == "failed" and "not blocked" in cluster["recorded"][0]["problems"][0]
    assert "--rollback" in capsys.readouterr().out


def test_a_rollout_that_never_completes_is_recorded_and_answered_with_the_rollback_not_lost(cluster, capsys):
    cluster["wait"] = "timed out waiting for the condition"
    assert dr.main(["release-f049973", "--owner", "Sandhya-SG", "--apply"]) == 1
    [entry] = cluster["recorded"]  # the failure is on record
    assert entry["result"] == "failed" and "timed out" in entry["problems"][0]
    assert "--rollback" in capsys.readouterr().out


def test_an_unapproved_deployment_is_recorded_as_such(cluster):
    dr.main(["sha-f049973", "--unapproved", "--owner", "Sandhya-SG", "--apply"])
    assert cluster["recorded"][0]["approved"] is False


def test_rollback_undoes_the_backend_then_the_frontend_waits_and_records(cluster, monkeypatch, capsys):
    assert dr.main(["--rollback"]) == 0
    assert cluster["order"] == ["undo statefulset/backend", "undo deployment/frontend", "wait"]
    assert cluster["recorded"][0]["ref"] == "rollback" and cluster["recorded"][0]["result"] == "ok"
    assert "rolled back" in capsys.readouterr().out


def test_a_rollback_that_fails_is_reported_not_hidden(cluster, monkeypatch, capsys):
    def failing_run(command, check=True):
        raise dr.DeployError("rollout undo failed: no previous revision")

    monkeypatch.setattr(dr, "run", failing_run)
    assert dr.main(["--rollback"]) == 1
    assert "no previous revision" in capsys.readouterr().out


def test_the_deployment_record_is_one_json_line_per_event_in_the_logs_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(dr, "ROOT", tmp_path)
    dr.record({"ref": "release-a", "result": "ok"})
    dr.record({"ref": "rollback", "result": "ok"})
    written = (tmp_path / "logs" / "deployments.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["ref"] for line in written] == ["release-a", "rollback"]


def test_the_version_token_is_read_from_the_secret_and_a_missing_one_is_none(monkeypatch):
    import base64

    class Result:
        def __init__(self, code, out):
            self.returncode, self.stdout = code, out

    monkeypatch.setattr(dr.subprocess, "run", lambda *a, **k: Result(0, base64.b64encode(b"tok-123").decode()))
    assert dr.version_token("kubectl") == "tok-123"
    monkeypatch.setattr(dr.subprocess, "run", lambda *a, **k: Result(1, ""))
    assert dr.version_token("kubectl") is None
