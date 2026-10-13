"""Version tracking: which code, models, prompts, manuals and evaluation data produced a result.

An answer, an audit record or an evaluation report is only reproducible if it says what it was made with. This
module collects five facts without importing the agents or loading any model, so it is cheap and safe to call at
start-up, from the API and from the evaluation scripts:

  * git commit      the code (the `GIT_COMMIT` build argument in a container, `git` on a developer machine)
  * model ids       every chat model and the embedding model named in the source
  * prompt hashes   a hash of every system prompt, per agent file and overall
  * corpus hash     a hash of the manual PDFs the index was built from (a manifest written at ingestion)
  * dataset version a hash of the evaluation datasets (only where the `eval` folder exists)

The agents are read as text (the `ast` module), never edited or imported, so adding this changes no behaviour.
Every function returns a value, never raises: a missing fact is reported as None or "unknown".
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent  # .../src/asa
REPO_ROOT = PACKAGE_DIR.parent.parent
AGENTS_DIR = PACKAGE_DIR / "agents"
EMBED_FILE = PACKAGE_DIR / "ingestion" / "embed.py"
MANIFEST_NAME = "corpus_manifest.json"
DATASET_DIR_NAMES = ("eval/datasets", "promptfoo-tests/datasets")
SHORT = 12  # characters of a hash shown in the audit log and on the version endpoint


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Code                                                                        #
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def git_commit() -> dict:
    """The commit the code was built from: `GIT_COMMIT` if set (containers have no .git), else ask git."""
    env_commit = os.getenv("GIT_COMMIT", "").strip()
    if env_commit and env_commit != "unknown":
        return {"commit": env_commit, "dirty": None, "source": "build"}
    try:
        run = lambda *args: subprocess.run(  # noqa: E731
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
        commit = run("rev-parse", "HEAD")
        return {"commit": commit, "dirty": bool(run("status", "--porcelain", "--untracked-files=no")), "source": "git"}
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unknown", "dirty": None, "source": "none"}


# --------------------------------------------------------------------------- #
# Models and prompts (read from the source, not imported)                      #
# --------------------------------------------------------------------------- #


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None


def _is_model_name(name: str) -> bool:
    return name.lower().endswith("_model") or name.lower() == "model"


def _target_name(target: ast.expr) -> str | None:
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return None


def _model_assignments(tree: ast.Module) -> dict[str, str]:
    """`SAFETY_MODEL = "gpt-4o-mini"` and `self.senior_model = "..."`: name to string value."""
    found: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
            continue
        for target in targets:
            name = _target_name(target)
            if name and _is_model_name(name):
                found[name] = value.value
    return found


@lru_cache(maxsize=1)
def model_ids() -> dict[str, str]:
    """Every model named in the agents, as `agent.NAME`, plus the embedding model."""
    models: dict[str, str] = {}
    for path in sorted(AGENTS_DIR.glob("*.py")):
        tree = _parse(path)
        if tree:
            for name, value in _model_assignments(tree).items():
                models[f"{path.stem}.{name}"] = value
    tree = _parse(EMBED_FILE)
    if tree:
        for name, value in _model_assignments(tree).items():
            models[f"embedding.{name}"] = value
    return models


def _is_system_message(node: ast.Dict) -> ast.expr | None:
    """The `content` of a `{"role": "system", "content": ...}` message, if this dictionary is one."""
    pairs = {k.value: v for k, v in zip(node.keys, node.values) if isinstance(k, ast.Constant)}
    role = pairs.get("role")
    if isinstance(role, ast.Constant) and role.value == "system" and "content" in pairs:
        return pairs["content"]
    return None


@lru_cache(maxsize=1)
def prompt_hashes() -> dict:
    """A hash of the system prompts of each agent file, and one over all of them.

    The prompt is hashed as it is written in the source, so a changed word changes the hash and a change anywhere
    else in the file does not.
    """
    per_agent: dict[str, dict] = {}
    for path in sorted(AGENTS_DIR.glob("*.py")):
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        tree = _parse(path)
        if not tree:
            continue
        segments = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                content = _is_system_message(node)
                if content is not None:
                    segments.append(ast.get_source_segment(source, content) or "")
        if segments:
            per_agent[path.stem] = {"count": len(segments), "sha256": _sha256_bytes("\n".join(segments).encode())}
    overall = _sha256_bytes("\n".join(f"{k}:{v['sha256']}" for k, v in sorted(per_agent.items())).encode())
    return {"sha256": overall, "agents": per_agent}


# --------------------------------------------------------------------------- #
# Manuals                                                                     #
# --------------------------------------------------------------------------- #


def compute_corpus_manifest(document_dir: Path) -> dict:
    """The PDFs in a folder with their sha256, and one hash over all of them (names and contents)."""
    files = {path.name: _sha256_file(path) for path in sorted(Path(document_dir).glob("*.pdf"))}
    combined = _sha256_bytes("\n".join(f"{name}:{sha}" for name, sha in files.items()).encode())
    return {"sha256": combined, "files": files}


def write_corpus_manifest(document_dir: Path, out_dir: Path) -> Path:
    """Record the manuals an index was built from, next to the index. Called by `scripts/ingest.py`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = compute_corpus_manifest(document_dir)
    path = out_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def corpus_manifest() -> dict | None:
    """The manuals the running index was built from (read from the manifest ingestion wrote), else None.

    A container deletes the PDFs after ingestion, so the manifest next to the index is the record; on a developer
    machine without a manifest the PDFs themselves are hashed. The index folder is `CHROMA_DIR` (default
    `chroma_store`), looked up in the working directory and in the repository root.
    """
    chroma_dir = os.getenv("CHROMA_DIR", "chroma_store")
    for base in (Path.cwd(), REPO_ROOT):
        manifest = base / chroma_dir / MANIFEST_NAME
        try:
            if manifest.is_file():
                return json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    document_dir = Path(os.getenv("AEM_DOCUMENT_PATH", "aem_documents"))
    for base in (Path.cwd(), REPO_ROOT):
        folder = document_dir if document_dir.is_absolute() else base / document_dir
        if folder.is_dir() and any(folder.glob("*.pdf")):
            return compute_corpus_manifest(folder)
    return None


# --------------------------------------------------------------------------- #
# Evaluation data                                                             #
# --------------------------------------------------------------------------- #


def dataset_version(root: Path | None = None) -> dict | None:
    """A hash of the evaluation datasets, per file and overall; None where the datasets are not present."""
    root = Path(root) if root else REPO_ROOT
    files: dict[str, str] = {}
    for name in DATASET_DIR_NAMES:
        folder = root / name
        if folder.is_dir():
            for path in sorted(p for p in folder.rglob("*") if p.is_file()):
                files[path.relative_to(root).as_posix()] = _sha256_file(path)
    if not files:
        return None
    combined = _sha256_bytes("\n".join(f"{name}:{sha}" for name, sha in files.items()).encode())
    return {"sha256": combined, "files": files}


PROMPTFOO_DIR = "promptfoo-tests"
PROMPTFOO_FILES = ("*.yaml", "*.js")
_GRADER = re.compile(r"provider:\s*(openai:[\w.\-]+)")


def promptfoo_config(root: Path | None = None) -> dict | None:
    """A hash of the Promptfoo suites (configs, assertion library, generators, provider) and the grader models they
    name; None where the folder is not present."""
    root = Path(root) if root else REPO_ROOT
    folder = root / PROMPTFOO_DIR
    if not folder.is_dir():
        return None
    files = {}
    for pattern in PROMPTFOO_FILES:
        for path in sorted(folder.glob(pattern)):
            files[path.name] = _sha256_file(path)
    graders = sorted(
        {
            match
            for path in folder.glob("*.yaml")
            for match in _GRADER.findall(path.read_text(encoding="utf-8", errors="ignore"))
        }
    )
    combined = _sha256_bytes("\n".join(f"{name}:{sha}" for name, sha in files.items()).encode())
    return {"sha256": combined, "files": files, "graders": graders}


def compare(expected: dict, actual: dict) -> list[str]:
    """What differs between two `build_info()` results (for example this checkout against a running backend).
    An empty list means the same build. Compares the commit, the prompts, the corpus and the models."""
    problems = []
    commit_a, commit_b = expected["git"]["commit"], actual["git"]["commit"]
    if "unknown" in (commit_a, commit_b):
        problems.append(f"commit cannot be compared: expected {commit_a[:12]}, running {commit_b[:12]}")
    elif not (commit_a.startswith(commit_b) or commit_b.startswith(commit_a)):
        problems.append(f"commit differs: expected {commit_a[:12]}, running {commit_b[:12]}")
    if expected["prompts"]["sha256"] != actual["prompts"]["sha256"]:
        problems.append("system prompts differ")
    if ((expected.get("corpus") or {}).get("sha256")) != ((actual.get("corpus") or {}).get("sha256")):
        problems.append("manual corpus differs (the index was built from other PDFs)")
    if expected["models"] != actual["models"]:
        problems.append("models differ")
    return problems


# --------------------------------------------------------------------------- #
# Everything, and the short form for the audit log                            #
# --------------------------------------------------------------------------- #


def build_info(include_datasets: bool = False) -> dict:
    """All five facts. `include_datasets` adds the evaluation datasets (evaluation reports; not needed at run time)."""
    corpus = corpus_manifest()
    info = {
        "git": git_commit(),
        "models": model_ids(),
        "prompts": prompt_hashes(),
        "corpus": corpus,
    }
    if include_datasets:
        info["datasets"] = dataset_version()
    return info


@lru_cache(maxsize=1)
def audit_stamp() -> dict:
    """The short form written into every audit `run_start` record: enough to tell two runs apart."""
    try:
        commit = git_commit()
        corpus = corpus_manifest()
        return {
            "commit": commit["commit"][:SHORT] if commit["commit"] != "unknown" else "unknown",
            "dirty": commit["dirty"],
            "prompts": prompt_hashes()["sha256"][:SHORT],
            "corpus": corpus["sha256"][:SHORT] if corpus else None,
            "models": sorted(set(model_ids().values())),
        }
    except Exception:  # noqa: BLE001 - versioning must never break a run
        return {"commit": "unknown"}
