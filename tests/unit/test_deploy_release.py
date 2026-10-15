"""The deploy script: approved tags only, digests not tags, every pod checked, free smoke tests. No cluster is used."""

import importlib.util
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
