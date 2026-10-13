"""Version-consistency tests: the metadata cannot go missing or drift without a test failing.

These guard the wiring of version tracking (see `test_versioning.py` for the facts themselves): every agent with a
prompt or a model is covered, every result writer stamps the build, the container receives the commit, and a running
backend can be compared with a checkout.
"""

import importlib.util
import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from asa import versioning as V
from asa.components import tracing

ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- nothing is missed ---


def test_every_agent_with_a_system_prompt_is_in_the_prompt_hash():
    """A new agent file with a system prompt must show up in the hash, or a prompt change would go unrecorded."""
    with_prompt = {
        path.stem
        for path in (ROOT / "src" / "asa" / "agents").glob("*.py")
        if '"role": "system"' in path.read_text("utf-8")
    }
    assert with_prompt, "the test found no agents: the check itself is broken"
    assert with_prompt == set(V.prompt_hashes()["agents"])


def test_every_model_constant_in_the_source_is_recorded():
    found = set()
    for path in (ROOT / "src" / "asa" / "agents").glob("*.py"):
        found |= {
            f"{path.stem}.{name}"
            for name in re.findall(r"^\s*(?:self\.)?(\w*_?model)\s*=\s*\"", path.read_text("utf-8"), re.M | re.I)
        }
    assert found, "the test found no models: the check itself is broken"
    assert found <= set(V.model_ids())


def test_the_version_facts_are_complete_in_this_checkout():
    info = V.build_info(include_datasets=True)
    assert info["git"]["commit"] != "unknown"
    assert info["corpus"] and info["corpus"]["files"], "the manual corpus is missing or unreadable"
    assert info["datasets"] and info["datasets"]["files"]
    assert len(info["models"]) >= 6


# --- the writers stamp the build ---


@pytest.mark.parametrize(
    "path, marker",
    [
        ("eval/run_eval.py", "build_info("),
        ("deepeval-tests/test_grounding.py", "build_info("),
        ("deepeval-tests/test_grounding.py", '"rubric": rubric()'),
        ("scripts/stamp_report.py", "build_info("),
        ("src/asa/components/logging_sub.py", "audit_stamp"),
    ],
)
def test_the_result_writers_record_the_build(path, marker):
    assert marker in (ROOT / path).read_text(encoding="utf-8"), f"{path} no longer records the build"


def test_the_container_receives_the_commit():
    assert "ARG GIT_COMMIT" in (ROOT / "Dockerfile.backend").read_text(encoding="utf-8")
    assert "GIT_COMMIT" in (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")


def test_ingestion_writes_the_manifest():
    assert "write_corpus_manifest" in (ROOT / "scripts" / "ingest.py").read_text(encoding="utf-8")


# --- the rubric (the judge's definition of grounded) ---


def test_the_rubric_hash_changes_with_the_definition():
    source = (ROOT / "deepeval-tests" / "test_grounding.py").read_text(encoding="utf-8")
    assert 'RUBRIC_VERSION = "' in source
    assert (
        "INVENTED_CRITERIA" in source
        and "FAITHFULNESS_THRESHOLD" in source.split("def rubric()")[1].split("def _case")[0]
    )


# --- stamping a result file and comparing with a running backend ---


def test_a_result_file_is_stamped(tmp_path):
    stamp_report = load_script("stamp_report")
    result = tmp_path / "results.json"
    result.write_text(json.dumps({"results": {"stats": {"successes": 3}}}), encoding="utf-8")
    assert stamp_report.main([str(result)]) == 0
    data = json.loads(result.read_text(encoding="utf-8"))
    assert data["results"]["stats"]["successes"] == 3  # nothing else was changed
    assert {"git", "models", "prompts", "corpus", "datasets", "promptfoo"} <= set(data["versions"])
    assert data["versions"]["promptfoo"]["graders"], "the grader model of the Promptfoo suites is recorded"
    assert stamp_report.main([str(result)]) == 0  # stamping again replaces the stamp


def test_the_promptfoo_config_hash_changes_with_a_suite(tmp_path):
    folder = tmp_path / "promptfoo-tests"
    folder.mkdir()
    (folder / "live.yaml").write_text("provider: openai:gpt-4o-mini\n", encoding="utf-8")
    first = V.promptfoo_config(tmp_path)
    (folder / "live.yaml").write_text("provider: openai:gpt-4.1\n", encoding="utf-8")
    second = V.promptfoo_config(tmp_path)
    assert first["sha256"] != second["sha256"]
    assert first["graders"] == ["openai:gpt-4o-mini"] and second["graders"] == ["openai:gpt-4.1"]
    assert V.promptfoo_config(tmp_path / "nowhere") is None


def test_the_same_build_compares_equal():
    info = V.build_info()
    assert V.compare(info, json.loads(json.dumps(info))) == []


def test_every_difference_is_reported():
    info = V.build_info()
    other = json.loads(json.dumps(info))
    other["git"]["commit"] = "f" * 40
    other["prompts"]["sha256"] = "0" * 64
    other["corpus"] = {"sha256": "1" * 64, "files": {}}
    other["models"] = {**info["models"], "rag.senior_model": "another-model"}
    problems = V.compare(info, other)
    assert len(problems) == 4
    assert any("commit differs" in p for p in problems)
    assert any("prompts" in p for p in problems) and any("corpus" in p for p in problems)


def test_an_abbreviated_commit_matches_the_full_one():
    info = V.build_info()
    other = json.loads(json.dumps(info))
    other["git"]["commit"] = info["git"]["commit"][:12]
    assert V.compare(info, other) == []


def test_an_unknown_commit_is_not_a_match():
    info = V.build_info()
    other = json.loads(json.dumps(info))
    other["git"]["commit"] = "unknown"
    assert any("cannot be compared" in p for p in V.compare(info, other))


def test_verify_version_reports_ok_and_mismatch(monkeypatch, capsys):
    verify = load_script("verify_version")
    info = V.build_info()
    monkeypatch.setattr(verify, "fetch", lambda url, token: json.loads(json.dumps(info)))
    assert verify.main(["http://backend:8000"]) == 0
    assert "OK" in capsys.readouterr().out
    monkeypatch.setattr(verify, "fetch", lambda url, token: {**info, "prompts": {"sha256": "0" * 64, "agents": {}}})
    assert verify.main(["http://backend:8000"]) == 1
    assert "MISMATCH" in capsys.readouterr().out


def test_verify_version_fails_when_the_backend_cannot_be_read(monkeypatch, capsys):
    verify = load_script("verify_version")

    def down(url, token):
        raise OSError("connection refused")

    monkeypatch.setattr(verify, "fetch", down)
    assert verify.main(["http://backend:8000"]) == 1
    assert "cannot read /version" in capsys.readouterr().out


def test_verify_version_uses_the_expected_commit(monkeypatch):
    verify = load_script("verify_version")
    info = V.build_info()
    running = json.loads(json.dumps(info))
    running["git"]["commit"] = "a" * 40
    monkeypatch.setattr(verify, "fetch", lambda url, token: running)
    assert verify.main(["http://x", "--expect-commit", "a" * 40]) == 0
    assert verify.main(["http://x", "--expect-commit", "b" * 40]) == 1


# --- /version is protected ---


def test_version_is_open_in_development(monkeypatch):
    from app.main import app

    monkeypatch.delenv("VERSION_TOKEN", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "development")
    assert TestClient(app).get("/version").status_code == 200


def test_version_needs_the_token_when_one_is_set(monkeypatch):
    from app.main import app

    monkeypatch.setenv("VERSION_TOKEN", "s3cret")
    client = TestClient(app)
    assert client.get("/version").status_code == 401
    assert client.get("/version", headers={"X-Version-Token": "wrong"}).status_code == 401
    assert client.get("/version", headers={"X-Version-Token": "s3cret"}).status_code == 200


@pytest.mark.parametrize("environment", ["production", "kubernetes", "staging"])
def test_version_is_off_in_a_deployed_environment_without_a_token(monkeypatch, environment):
    from app.main import app

    monkeypatch.delenv("VERSION_TOKEN", raising=False)
    monkeypatch.setenv("ENVIRONMENT", environment)
    assert TestClient(app).get("/version").status_code == 404


def test_version_is_readable_in_kubernetes_with_the_token(monkeypatch):
    from app.main import app

    monkeypatch.setenv("ENVIRONMENT", "kubernetes")
    monkeypatch.setenv("VERSION_TOKEN", "s3cret")
    assert TestClient(app).get("/version", headers={"X-Version-Token": "s3cret"}).status_code == 200


def test_the_kubernetes_backend_reads_the_token_as_an_optional_secret():
    manifest = (ROOT / "k8s" / "backend.yaml").read_text(encoding="utf-8")
    assert "name: VERSION_TOKEN" in manifest and "optional: true" in manifest
    assert "VERSION_TOKEN" in (ROOT / "k8s" / "secret.example.yaml").read_text(encoding="utf-8")
    assert "VERSION_TOKEN" in (ROOT / "scripts" / "k8s_secret.py").read_text(encoding="utf-8")


# --- Langfuse traces carry the build ---


def test_a_trace_is_versioned_with_the_commit_and_build_metadata(monkeypatch):
    monkeypatch.setenv("GIT_COMMIT", "0123456789abcdef0123")
    V.git_commit.cache_clear()
    V.audit_stamp.cache_clear()
    meta = tracing._request_metadata("thermal_station")
    assert tracing._build_version() == "0123456789ab"
    assert meta["equipment_model"] == "thermal_station"
    assert {"commit", "prompts", "models"} <= set(meta)
    assert all(isinstance(value, str) and len(value) <= 200 for value in meta.values())
    V.git_commit.cache_clear()
    V.audit_stamp.cache_clear()


def test_the_trace_metadata_never_raises(monkeypatch):
    monkeypatch.setattr(V, "audit_stamp", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert tracing._build_metadata() == {}
    assert tracing._request_metadata(None) == {}


# --- MLflow logging (the pure parts; mlflow itself is optional) ---


def test_mlflow_metrics_and_params_are_extracted():
    log = load_script("log_to_mlflow")
    deepeval = {
        "cases": {"g01": {"grounded": True}},
        "summary": {
            "judge_model": "gpt-4.1",
            "repeats": 3,
            "grounding_rate": 0.9,
            "mean_scores": {"FaithfulnessMetric": 0.95},
            "rubric": {"version": "2", "sha256": "ab" * 32},
            "versions": V.build_info(),
        },
    }
    metrics = log.flatten_metrics(deepeval["summary"])
    assert metrics["grounding_rate"] == 0.9 and metrics["mean_scores.FaithfulnessMetric"] == 0.95
    assert not any(key.startswith(("versions", "rubric")) for key in metrics)
    params = log.build_params(deepeval)
    assert params["judge_model"] == "gpt-4.1" and params["rubric_version"] == "2"
    assert {"git_commit", "prompts_sha256", "corpus_sha256", "models"} <= set(params)


def test_mlflow_flatten_ignores_booleans_and_lists():
    log = load_script("log_to_mlflow")
    assert log.flatten_metrics({"ok": True, "rows": [1, 2], "n": 3, "nested": {"x": 1.5}}) == {
        "n": 3.0,
        "nested.x": 1.5,
    }


def test_mlflow_script_reports_when_mlflow_is_missing(monkeypatch, capsys, tmp_path):
    log = load_script("log_to_mlflow")
    monkeypatch.setitem(__import__("sys").modules, "mlflow", None)  # makes `import mlflow` fail
    result = tmp_path / "r.json"
    result.write_text("{}", encoding="utf-8")
    assert log.main([str(result)]) == 1
    assert "mlflow is not installed" in capsys.readouterr().out


def test_mlflow_logs_one_run_per_file(monkeypatch, tmp_path, capsys):
    """The whole flow against a stand-in for mlflow: one run per file, with the build as parameters."""
    import contextlib
    import sys
    import types

    log = load_script("log_to_mlflow")
    calls = {"runs": [], "params": [], "metrics": {}, "artifacts": []}
    fake = types.SimpleNamespace(
        set_experiment=lambda name: calls.update(experiment=name),
        start_run=lambda run_name=None: calls["runs"].append(run_name) or contextlib.nullcontext(),
        log_param=lambda key, value: calls["params"].append((key, value)),
        log_params=lambda params: calls["params"].extend(params.items()),
        log_metric=lambda key, value: calls["metrics"].__setitem__(key, value),
        log_artifact=lambda path: calls["artifacts"].append(path),
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    result = tmp_path / "deepeval-results.json"
    result.write_text(
        json.dumps({"summary": {"judge_model": "gpt-4.1", "grounding_rate": 0.9, "versions": V.build_info()}}),
        encoding="utf-8",
    )
    assert log.main([str(result), "--experiment", "demo"]) == 0
    assert calls["experiment"] == "demo" and calls["runs"] == ["deepeval-results"]
    assert dict(calls["params"])["judge_model"] == "gpt-4.1"
    assert calls["metrics"]["grounding_rate"] == 0.9
    assert calls["artifacts"] == [str(result)]
