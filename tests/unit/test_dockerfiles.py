"""The container images: non-root, no secrets baked in, only what is named is copied, and the .dockerignore never removes
something a Dockerfile needs. (Parsed as text; no Docker is needed.)"""

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
K8S = ROOT / "k8s"
FILES = {"backend": ROOT / "Dockerfile.backend", "frontend": ROOT / "Dockerfile.frontend"}
SECRET_WORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL")


def lines(name):
    """The instructions of a Dockerfile, with comments removed and continued lines joined."""
    text = FILES[name].read_text(encoding="utf-8")
    joined = re.sub(r"\\\s*\n", " ", text)
    return [line.strip() for line in joined.splitlines() if line.strip() and not line.strip().startswith("#")]


def instructions(name, keyword):
    return [line.split(None, 1)[1] for line in lines(name) if line.split(None, 1)[0].upper() == keyword]


def load(path):
    return list(yaml.safe_load_all(path.read_text(encoding="utf-8")))


# --- the images ---


@pytest.mark.parametrize("name", FILES)
def test_the_image_starts_from_a_pinned_python_slim_base(name):
    [base] = instructions(name, "FROM")
    assert re.fullmatch(r"python:3\.\d+-slim", base), base  # a minor version, not latest, and no full OS image


@pytest.mark.parametrize("name", FILES)
def test_the_container_runs_as_a_numeric_non_root_user(name):
    [user] = instructions(name, "USER")
    assert user == "10001"  # numeric, so Kubernetes can verify runAsNonRoot
    assert any("useradd" in run and "--uid 10001" in run for run in instructions(name, "RUN"))
    assert lines(name).index(f"USER {user}") > max(i for i, line in enumerate(lines(name)) if line.startswith("RUN"))


@pytest.mark.parametrize("name", FILES)
def test_the_runtime_user_matches_the_cluster_security_context(name):
    workload = next(doc for doc in load(K8S / f"{name}.yaml") if doc["kind"] in ("Deployment", "StatefulSet"))
    context = workload["spec"]["template"]["spec"]["securityContext"]
    assert context["runAsUser"] == 10001 and context["runAsNonRoot"] is True


@pytest.mark.parametrize("name", FILES)
def test_no_secret_is_set_in_the_image(name):
    for keyword in ("ENV", "ARG"):
        for entry in instructions(name, keyword):
            variable = re.split(r"[=\s]", entry.strip(), maxsplit=1)[0].upper()
            assert not any(word in variable for word in SECRET_WORDS), f"{keyword} {variable}"


@pytest.mark.parametrize("name", FILES)
def test_the_whole_build_context_is_never_copied(name):
    for entry in instructions(name, "COPY"):
        sources = [part for part in entry.split() if not part.startswith("--")][:-1]
        assert all(source not in (".", "./", "*", "./*") for source in sources), entry
        assert not any(".env" in source or ".git" in source for source in sources), entry


@pytest.mark.parametrize("name", FILES)
def test_the_operating_system_is_patched_and_the_package_lists_are_removed(name):
    runs = " ".join(instructions(name, "RUN"))
    assert "apt-get upgrade" in runs and "rm -rf /var/lib/apt/lists" in runs
    assert "pip uninstall -y setuptools wheel" in runs  # build tools the app does not need at run time


@pytest.mark.parametrize("name", FILES)
def test_the_command_is_in_exec_form_so_signals_reach_the_server(name):
    [command] = instructions(name, "CMD")
    assert command.startswith("[") and command.endswith("]")


def test_each_image_installs_its_own_slim_requirements_not_the_full_list():
    assert any("requirements-backend.txt" in entry for entry in instructions("backend", "COPY"))
    assert any("requirements-frontend.txt" in entry for entry in instructions("frontend", "COPY"))
    for name in FILES:
        assert not any(re.search(r"requirements\.txt", entry) for entry in instructions(name, "COPY"))
    frontend = (ROOT / "requirements-frontend.txt").read_text(encoding="utf-8").lower()
    for heavy in ("torch", "chromadb", "sentence-transformers", "langgraph", "openai"):
        assert heavy not in frontend  # the frontend must stay small and free of the ML stack


# --- the backend image ---


def test_the_backend_image_records_the_commit_and_builds_its_own_index():
    args = instructions("backend", "ARG")
    envs = instructions("backend", "ENV")
    assert "GIT_COMMIT=unknown" in args and any(e.startswith("GIT_COMMIT=") for e in envs)
    runs = " ".join(instructions("backend", "RUN"))
    assert (
        "scripts/ingest.py" in runs and "rm -rf aem_documents scripts" in runs
    )  # manuals and script are not left behind
    assert (
        "SentenceTransformer('all-MiniLM-L6-v2')" in runs
    )  # the embedding model is baked in: the container starts offline
    assert "HF_HOME=/app/.cache/huggingface" in " ".join(envs)


def test_the_backend_image_serves_the_app_on_the_port_the_cluster_expects():
    [command] = instructions("backend", "CMD")
    assert '"app.main:app"' in command and '"8000"' in command and '"0.0.0.0"' in command
    assert instructions("backend", "EXPOSE") == ["8000"]
    backend = next(doc for doc in load(K8S / "backend.yaml") if doc["kind"] == "StatefulSet")
    container = next(c for c in backend["spec"]["template"]["spec"]["containers"] if c["name"] == "backend")
    assert container["ports"][0]["containerPort"] == 8000


def test_the_backend_image_makes_its_writable_folders_owned_by_the_app_user():
    runs = " ".join(instructions("backend", "RUN"))
    assert "chown -R 10001:10001 /app" in runs  # logs and the index live under /app


# --- the frontend image ---


def test_the_frontend_image_uses_gunicorn_not_the_development_server():
    [command] = instructions("frontend", "CMD")
    assert "gunicorn" in command and '"frontend.app:app"' in command and '"0.0.0.0:5000"' in command
    assert "flask run" not in command and "app.run" not in command
    assert instructions("frontend", "EXPOSE") == ["5000"]


def test_the_frontend_waits_longer_than_the_backend_and_the_pod_outlives_a_request():
    [command] = instructions("frontend", "CMD")
    timeout = int(re.search(r'"--timeout",\s*"(\d+)"', command).group(1))
    configmap = load(K8S / "configmap.yaml")[0]["data"]
    frontend = next(doc for doc in load(K8S / "frontend.yaml") if doc["kind"] == "Deployment")
    grace = frontend["spec"]["template"]["spec"]["terminationGracePeriodSeconds"]
    assert timeout > int(configmap["BACKEND_TIMEOUT"])  # a slow backend answer is not cut off by the server
    assert grace >= timeout  # a stopping pod can finish the request it is serving


# --- the .dockerignore ---


def _pattern(rule):
    out, i = "", 0
    while i < len(rule):
        if rule.startswith("**/", i):
            out += "(?:.*/)?"
            i += 3
        elif rule.startswith("**", i):
            out += ".*"
            i += 2
        elif rule[i] == "*":
            out += "[^/]*"
            i += 1
        elif rule[i] == "?":
            out += "[^/]"
            i += 1
        else:
            out += re.escape(rule[i])
            i += 1
    return re.compile(out + r"\Z")


def ignored(path):
    """Whether the build context leaves `path` out: the last matching rule wins, and a rule also covers what is inside it."""
    state = False
    rules = [r.strip() for r in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()]
    for rule in (r for r in rules if r and not r.startswith("#")):
        negate = rule.startswith("!")
        regex = _pattern(rule.lstrip("!").strip("/"))
        parts = path.split("/")
        if any(regex.match("/".join(parts[: n + 1])) for n in range(len(parts))):
            state = not negate
    return state


def test_a_dockerignore_exists_and_keeps_secrets_and_history_out_of_the_build():
    assert (ROOT / ".dockerignore").is_file()
    for secret_or_history in (".env", ".env.local", ".git/config", ".venv/lib/x.py", "server.pem", "api.key"):
        assert ignored(secret_or_history), secret_or_history
    assert not ignored(".env.example")  # the template is not a secret


def test_large_local_folders_and_test_tooling_stay_out_of_the_build():
    for path in (
        "chroma_store/chroma.sqlite3",
        "logs/audit.jsonl",
        "docs/runbook.md",
        "tests/test_chat.py",
        "promptfoo-tests/node_modules/x.js",
        "mlruns/0/meta.yaml",
        ".openai_spend.json",
        "k8s/backend.yaml",
    ):
        assert ignored(path), path


@pytest.mark.parametrize("name", FILES)
def test_nothing_a_dockerfile_copies_is_excluded_by_the_dockerignore(name):
    for entry in instructions(name, "COPY"):
        sources = [part for part in entry.split() if not part.startswith("--")][:-1]
        for source in sources:
            target = ROOT / source
            assert target.exists(), f"{source} is copied but does not exist"
            probe = (
                next((p for p in sorted(target.rglob("*")) if p.is_file() and "__pycache__" not in p.parts), target)
                if target.is_dir()
                else target
            )
            assert not ignored(source), f"{source} is excluded by .dockerignore"
            assert not ignored(probe.relative_to(ROOT).as_posix()), (
                f"{probe.relative_to(ROOT)} is excluded by .dockerignore"
            )


def test_only_the_ingestion_script_is_let_through_from_the_scripts_folder():
    assert not ignored("scripts/ingest.py")
    for other in ("scripts/deploy_release.py", "scripts/k8s_secret.py", "scripts/stamp_report.py"):
        assert ignored(other), other
