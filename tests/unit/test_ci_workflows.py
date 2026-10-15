"""The release gate: the image build needs the Promptfoo security evaluation, so a failing one cannot publish."""

from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"


def load(name):
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def triggers(workflow):
    return workflow.get("on", workflow.get(True))  # YAML reads a bare `on` as the boolean True


@pytest.fixture(scope="module")
def ci():
    return load("ci.yml")["jobs"]


def test_the_image_build_needs_every_quality_and_security_job(ci):
    needs = set(ci["build"]["needs"])
    assert {"unit-tests", "security-tests", "promptfoo", "dependency-scan"} <= needs


def test_the_promptfoo_job_calls_the_reusable_workflow_and_is_not_optional(ci):
    job = ci["promptfoo"]
    assert job["uses"] == "./.github/workflows/promptfoo-tests-ci.yml"
    assert "continue-on-error" not in job and "if" not in job  # it cannot be skipped or ignored


def test_the_promptfoo_workflow_can_be_called_and_does_not_also_run_on_its_own_pushes():
    on = triggers(load("promptfoo-tests-ci.yml"))
    assert "workflow_call" in on and "workflow_dispatch" in on
    assert "push" not in on and "pull_request" not in on  # otherwise it runs twice and, path-filtered, can be skipped


def test_the_gating_suite_fails_the_job_and_the_known_gaps_suite_does_not():
    steps = {s.get("name"): s for s in load("promptfoo-tests-ci.yml")["jobs"]["promptfoo-free"]["steps"]}
    gating = steps["Run the Promptfoo free suite"]
    assert "continue-on-error" not in gating and "free-api.yaml" in gating["run"]
    known = steps["Report the known gaps (not gating)"]
    assert known["continue-on-error"] is True and "known-gaps.yaml" in known["run"]


def test_the_test_backend_has_no_request_limit_because_the_suite_sends_many_requests():
    steps = load("promptfoo-tests-ci.yml")["jobs"]["promptfoo-free"]["steps"]
    start = next(s for s in steps if s.get("name") == "Start the backend with a fake key")
    assert start["env"]["RATE_LIMIT_PER_MINUTE"] == "0"  # with the default 30 a run of 86 requests fails halfway


def test_the_html_report_is_uploaded_with_the_json_results():
    steps = load("promptfoo-tests-ci.yml")["jobs"]["promptfoo-free"]["steps"]
    run = next(s for s in steps if s.get("name") == "Run the Promptfoo free suite")["run"]
    upload = next(s for s in steps if s.get("name") == "Upload Promptfoo results")["with"]["path"]
    assert "free-report.html" in run and "free-report.html" in upload and "free-results.json" in upload


def test_images_are_published_only_after_the_gate_and_only_from_main(ci):
    build = ci["build"]
    assert "github.ref == 'refs/heads/main'" in build["env"]["PUBLISH"]
    assert "promptfoo" in build["needs"]


# --- the release approval ---


def test_the_release_job_waits_for_the_production_environment_after_the_build(ci):
    release = ci["release"]
    assert release["environment"] == "production"  # the required reviewers live on this environment
    assert release["needs"] == ["build"]
    assert "github.ref == 'refs/heads/main'" in release["if"] and "push" in release["if"]


def test_the_release_job_adds_a_tag_to_the_published_image_and_never_rebuilds_or_moves_latest(ci):
    text = " ".join(step.get("run", "") for step in ci["release"]["steps"])
    assert "imagetools create --tag" in text and ":sha-" in text and ":release-" in text
    assert "docker build " not in text and ":latest" not in text
    assert "digest mismatch" in text  # the approved digest must equal the built one


def test_the_release_job_can_write_packages_and_nothing_else_it_does_not_need(ci):
    assert ci["release"]["permissions"] == {"contents": "read", "packages": "write"}
