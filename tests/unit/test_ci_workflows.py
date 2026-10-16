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
    assert "imagetools create --prefer-index=false --tag" in text  # an exact copy: same manifest, same digest
    assert ":sha-" in text and ":release-" in text
    assert "docker build " not in text and ":latest" not in text
    assert "digest mismatch" in text  # the approved digest must equal the built one


def test_the_release_job_can_write_packages_and_nothing_else_it_does_not_need(ci):
    assert ci["release"]["permissions"] == {"contents": "read", "packages": "write"}


# --- the offline evaluation gate and the retrieval job ---


def test_the_offline_evaluation_fails_the_job_when_a_target_is_missed(ci):
    steps = ci["offline-evals"]["steps"]
    run = next(s for s in steps if s.get("name") == "Run the offline evaluation")["run"]
    assert "run_eval.py --enforce" in run
    assert "continue-on-error" not in ci["offline-evals"]


def test_the_retrieval_job_builds_the_index_before_it_measures_and_never_blocks_the_release(ci):
    job = ci["retrieval-eval"]
    names = [s.get("name") for s in job["steps"]]
    assert names.index("Build the manual index") < names.index("Run the retrieval evaluation (report only)")
    ingest = next(s for s in job["steps"] if s.get("name") == "Build the manual index")
    assert ingest["run"].strip() == "python scripts/ingest.py" and "continue-on-error" not in ingest
    measure = next(s for s in job["steps"] if s.get("name") == "Run the retrieval evaluation (report only)")
    assert measure["continue-on-error"] is True and "retrieval_eval.py" in measure["run"]
    assert "--enforce" not in measure["run"]  # report only until a few green runs show the numbers are stable
    assert "retrieval-eval" not in ci["build"]["needs"] and "retrieval-eval" not in ci["release"]["needs"]


def test_the_retrieval_job_needs_no_key_and_uploads_its_reports(ci):
    job = ci["retrieval-eval"]
    assert "OPENAI_API_KEY" not in str(job) and "secrets." not in str(job).replace("secrets.GITHUB_TOKEN", "")
    upload = next(s for s in job["steps"] if s.get("name") == "Upload the retrieval report")
    assert upload["if"] == "always()" and "retrieval-report.json" in upload["with"]["path"]


def test_a_suite_that_cannot_run_is_skipped_not_failed_so_enforce_is_safe_without_an_index():
    import sys

    sys.path.insert(0, str(WORKFLOWS.parents[1] / "eval"))
    import run_eval

    report = {"retrieval": {"available": False, "reason": "no index"}, "injection": {"block_rate_all": 1.0}}
    targets = {"retrieval": {"area_hit_at_5": {"min": 0.85}}, "injection": {"block_rate_all": {"min": 0.9}}}
    rows = run_eval.check_targets(report, targets)
    assert [(row["suite"], row["met"]) for row in rows] == [("injection", True)]


# --- the manifest scan ---


def test_the_manifests_are_scanned_and_a_high_or_critical_finding_fails_the_job(ci):
    job = ci["manifest-scan"]
    scan = next(step for step in job["steps"] if "trivy-action" in step.get("uses", ""))
    options = scan["with"]
    assert options["scan-type"] == "config" and options["scan-ref"] == "k8s"
    assert options["exit-code"] == "1" and set(options["severity"].split(",")) == {"CRITICAL", "HIGH"}
    assert "continue-on-error" not in job and "continue-on-error" not in scan


def test_the_image_build_needs_the_manifest_scan_so_a_misconfiguration_cannot_release(ci):
    assert "manifest-scan" in ci["build"]["needs"]
