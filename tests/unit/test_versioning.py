"""Tests for version tracking: commit, model ids, prompt hashes, corpus hash, dataset version."""

import json

import pytest
from fastapi.testclient import TestClient

from asa import versioning as V
from asa.components import logging_sub as L

CACHED = ("git_commit", "model_ids", "prompt_hashes", "audit_stamp")


def clear_caches():
    for name in CACHED:
        clear = getattr(getattr(V, name), "cache_clear", None)  # a test may have replaced the function
        if clear:
            clear()


@pytest.fixture(autouse=True)
def fresh_caches():
    """The facts are cached for the life of the process; each test starts from a clean cache."""
    clear_caches()
    yield
    clear_caches()


def agents_dir(tmp_path, files):
    folder = tmp_path / "agents"
    folder.mkdir()
    for name, text in files.items():
        (folder / name).write_text(text, encoding="utf-8")
    return folder


# --- code ---


def test_the_build_commit_wins_over_git(monkeypatch):
    monkeypatch.setenv("GIT_COMMIT", "abc123def456")
    assert V.git_commit() == {"commit": "abc123def456", "dirty": None, "source": "build"}


def test_the_commit_comes_from_git_without_a_build_argument(monkeypatch):
    monkeypatch.delenv("GIT_COMMIT", raising=False)
    info = V.git_commit()
    assert info["source"] in ("git", "none")
    assert len(info["commit"]) == 40 or info["commit"] == "unknown"


def test_an_unset_build_argument_is_not_a_commit(monkeypatch):
    """The Dockerfile default is the word 'unknown'; that must not be reported as a commit."""
    monkeypatch.setenv("GIT_COMMIT", "unknown")
    assert V.git_commit()["source"] != "build"


def test_no_git_and_no_argument_is_reported_as_unknown(monkeypatch):
    monkeypatch.delenv("GIT_COMMIT", raising=False)

    def broken(*args, **kwargs):
        raise OSError("git is not installed")

    monkeypatch.setattr(V.subprocess, "run", broken)
    assert V.git_commit() == {"commit": "unknown", "dirty": None, "source": "none"}


# --- models ---


def test_the_models_of_the_agents_are_read_from_the_source(monkeypatch, tmp_path):
    folder = agents_dir(
        tmp_path,
        {
            "alpha.py": 'ALPHA_MODEL = "model-a"\n\nclass A:\n    def __init__(self):\n        self.junior_model = "model-j"\n',
            "beta.py": 'NOT_A_NAME = "ignored"\nBETA_MODEL: str = "model-b"\nTIMEOUT = 5\n',
        },
    )
    monkeypatch.setattr(V, "AGENTS_DIR", folder)
    monkeypatch.setattr(V, "EMBED_FILE", tmp_path / "missing.py")
    assert V.model_ids() == {
        "alpha.ALPHA_MODEL": "model-a",
        "alpha.junior_model": "model-j",
        "beta.BETA_MODEL": "model-b",
    }


def test_the_real_agents_and_the_embedding_model_are_found():
    models = V.model_ids()
    assert models["safety.SAFETY_MODEL"].startswith("gpt-")
    assert models["rag.senior_model"].startswith("gpt-")
    assert models["embedding.EMBED_MODEL"] == "all-MiniLM-L6-v2"


def test_a_file_that_does_not_parse_is_skipped(monkeypatch, tmp_path):
    folder = agents_dir(tmp_path, {"broken.py": "def (:\n", "ok.py": 'OK_MODEL = "m"\n'})
    monkeypatch.setattr(V, "AGENTS_DIR", folder)
    monkeypatch.setattr(V, "EMBED_FILE", tmp_path / "missing.py")
    assert V.model_ids() == {"ok.OK_MODEL": "m"}


# --- prompts ---

PROMPT_A = 'def call():\n    return [{"role": "system", "content": "You are a safety agent."}, {"role": "user", "content": x}]\n'


def prompts_of(monkeypatch, tmp_path, text):
    """The prompt hashes of a folder holding one agent file with this text."""
    V.prompt_hashes.cache_clear()
    folder = tmp_path / f"agents{len(list(tmp_path.iterdir()))}"
    monkeypatch.setattr(V, "AGENTS_DIR", agents_dir_at(folder, {"agent.py": text}))
    return V.prompt_hashes()


def agents_dir_at(folder, files):
    folder.mkdir()
    for name, text in files.items():
        (folder / name).write_text(text, encoding="utf-8")
    return folder


def test_a_changed_prompt_changes_the_hash(monkeypatch, tmp_path):
    first = prompts_of(monkeypatch, tmp_path, PROMPT_A)
    second = prompts_of(monkeypatch, tmp_path, PROMPT_A.replace("safety agent", "safety officer"))
    assert first["sha256"] != second["sha256"]
    assert first["agents"]["agent"]["count"] == 1


def test_a_change_outside_the_prompts_does_not_change_the_hash(monkeypatch, tmp_path):
    first = prompts_of(monkeypatch, tmp_path, PROMPT_A)
    second = prompts_of(monkeypatch, tmp_path, "# a comment\nTIMEOUT = 9\n" + PROMPT_A + "\nOTHER = 1\n")
    assert first["sha256"] == second["sha256"]


def test_user_messages_are_not_prompts(monkeypatch, tmp_path):
    info = prompts_of(monkeypatch, tmp_path, 'def f():\n    return {"role": "user", "content": "hello"}\n')
    assert info["agents"] == {}


def test_the_hash_is_the_same_twice():
    assert V.prompt_hashes() == V.prompt_hashes()


def test_every_agent_with_a_system_prompt_is_hashed():
    agents = V.prompt_hashes()["agents"]
    assert {"rag", "safety", "diagnostic", "planning", "request"} <= set(agents)
    assert all(len(a["sha256"]) == 64 and a["count"] >= 1 for a in agents.values())


# --- manuals ---


def make_manuals(folder, contents):
    folder.mkdir()
    for name, data in contents.items():
        (folder / name).write_bytes(data)
    return folder


def test_the_corpus_hash_covers_names_and_contents(tmp_path):
    one = V.compute_corpus_manifest(make_manuals(tmp_path / "one", {"a.pdf": b"AAA", "b.pdf": b"BBB"}))
    changed = V.compute_corpus_manifest(make_manuals(tmp_path / "two", {"a.pdf": b"AAA", "b.pdf": b"BBX"}))
    renamed = V.compute_corpus_manifest(make_manuals(tmp_path / "three", {"a.pdf": b"AAA", "c.pdf": b"BBB"}))
    assert set(one["files"]) == {"a.pdf", "b.pdf"}
    assert len({one["sha256"], changed["sha256"], renamed["sha256"]}) == 3


def test_only_pdfs_are_part_of_the_corpus(tmp_path):
    manifest = V.compute_corpus_manifest(make_manuals(tmp_path / "m", {"a.pdf": b"AAA", "notes.txt": b"x"}))
    assert list(manifest["files"]) == ["a.pdf"]


def test_the_manifest_is_written_next_to_the_index_and_read_back(tmp_path, monkeypatch):
    manuals = make_manuals(tmp_path / "manuals", {"a.pdf": b"AAA"})
    path = V.write_corpus_manifest(manuals, tmp_path / "chroma_store")
    assert path.name == "corpus_manifest.json"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CHROMA_DIR", "chroma_store")
    assert V.corpus_manifest() == json.loads(path.read_text(encoding="utf-8"))


def test_the_manifest_survives_deleted_manuals(tmp_path, monkeypatch):
    """A container deletes the PDFs after ingestion: the manifest written then is still the record."""
    manuals = make_manuals(tmp_path / "manuals", {"a.pdf": b"AAA"})
    V.write_corpus_manifest(manuals, tmp_path / "chroma_store")
    (manuals / "a.pdf").unlink()
    manuals.rmdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CHROMA_DIR", "chroma_store")
    assert V.corpus_manifest()["files"] == {"a.pdf": V._sha256_bytes(b"AAA")}


def test_without_a_manifest_the_manuals_are_hashed(tmp_path, monkeypatch):
    make_manuals(tmp_path / "aem_documents", {"a.pdf": b"AAA"})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(V, "REPO_ROOT", tmp_path / "elsewhere")
    monkeypatch.delenv("CHROMA_DIR", raising=False)
    monkeypatch.delenv("AEM_DOCUMENT_PATH", raising=False)
    assert V.corpus_manifest()["files"] == {"a.pdf": V._sha256_bytes(b"AAA")}


def test_no_manifest_and_no_manuals_is_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(V, "REPO_ROOT", tmp_path)
    assert V.corpus_manifest() is None


def test_a_broken_manifest_is_ignored(tmp_path, monkeypatch):
    (tmp_path / "chroma_store").mkdir()
    (tmp_path / "chroma_store" / "corpus_manifest.json").write_text("{not json", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(V, "REPO_ROOT", tmp_path)
    assert V.corpus_manifest() is None


# --- evaluation datasets ---


def test_the_dataset_version_changes_with_any_dataset(tmp_path):
    folder = tmp_path / "eval" / "datasets"
    folder.mkdir(parents=True)
    (folder / "a.jsonl").write_text('{"id": 1}\n', encoding="utf-8")
    first = V.dataset_version(tmp_path)
    (folder / "a.jsonl").write_text('{"id": 2}\n', encoding="utf-8")
    assert first["sha256"] != V.dataset_version(tmp_path)["sha256"]
    assert list(first["files"]) == ["eval/datasets/a.jsonl"]


def test_no_datasets_is_none(tmp_path):
    assert V.dataset_version(tmp_path) is None


def test_the_real_datasets_are_versioned():
    info = V.dataset_version()
    assert info and any(name.endswith("adversarial.jsonl") or "datasets" in name for name in info["files"])


# --- everything together, the audit log and the endpoint ---


def test_build_info_has_every_fact():
    info = V.build_info()
    assert set(info) == {"git", "models", "prompts", "corpus"}
    assert "datasets" in V.build_info(include_datasets=True)


def test_the_audit_stamp_is_short_and_has_no_text_of_the_prompts():
    stamp = V.audit_stamp()
    assert {"commit", "dirty", "prompts", "corpus", "models"} <= set(stamp)
    assert len(stamp["prompts"]) == V.SHORT
    assert "You are" not in json.dumps(stamp)


def test_the_audit_stamp_never_raises(monkeypatch):
    def broken():
        raise RuntimeError("boom")

    monkeypatch.setattr(V, "git_commit", broken)
    assert V.audit_stamp() == {"commit": "unknown"}


def test_a_run_start_record_carries_the_build(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "LOG_DIR", tmp_path)
    monkeypatch.setattr(L, "AUDIT_LOG", tmp_path / "audit.jsonl")
    monkeypatch.setenv("GIT_COMMIT", "0123456789abcdef0123")
    L.start_run("handler fault")
    L.log_blocked("conv-1", ["prompt_injection"], 40)
    starts = [json.loads(line) for line in L.AUDIT_LOG.read_text().splitlines()]
    starts = [r for r in starts if r["type"] == "run_start"]
    assert len(starts) == 2
    assert all(r["build"]["commit"] == "0123456789ab" for r in starts)


def test_the_version_endpoint_reports_the_build(monkeypatch):
    from app.main import app

    monkeypatch.setenv("GIT_COMMIT", "feedfacefeedface")
    body = TestClient(app).get("/version").json()
    assert body["git"]["commit"] == "feedfacefeedface"
    assert body["models"]["embedding.EMBED_MODEL"] == "all-MiniLM-L6-v2"
    assert len(body["prompts"]["sha256"]) == 64
