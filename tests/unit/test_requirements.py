"""The slim requirements lists must cover what each image imports: a missing package only shows up when the container
starts, so it is checked here. (Reads the source with `ast`; needs the packages installed to map imports to names.)"""

import ast
import re
import sys
from importlib import metadata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LOCAL = {"app", "asa", "frontend", "security", "scripts", "eval", "tests"}  # this repository's own packages
# Imported by the code but installed as a dependency of a listed package, and stable enough to rely on.
TRANSITIVE_OK = {
    "starlette",  # FastAPI is built on it
    "httpx",  # a dependency of the openai client, used for its timeout settings
    "numpy",  # a dependency of chromadb and sentence-transformers
    "pydantic_core",
}


def normal(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_names(path):
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.add(normal(re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0]))
    return names


def imported_modules(folder):
    """The top-level modules imported anywhere in the Python files of `folder` (also inside functions and try blocks)."""
    found = {}
    for path in sorted(folder.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                found.setdefault(name.split(".")[0], path.relative_to(ROOT).as_posix())
    return found


def third_party(folder):
    """{module: first file that imports it} for the modules that are neither standard library nor this repository's own."""
    return {m: f for m, f in imported_modules(folder).items() if m not in sys.stdlib_module_names and m not in LOCAL}


def distribution_of(module):
    owners = metadata.packages_distributions().get(module)
    return normal(owners[0]) if owners else None


BACKEND_FOLDERS = ("app", "src/asa")


@pytest.mark.parametrize("folder", BACKEND_FOLDERS)
def test_every_package_the_backend_imports_is_in_the_backend_requirements(folder):
    listed = requirement_names(ROOT / "requirements-backend.txt")
    missing = []
    for module, where in third_party(ROOT / folder).items():
        owner = distribution_of(module)
        if owner is None:
            missing.append(f"{module} (in {where}) is not installed here, so it cannot be checked")
        elif owner not in listed and module not in TRANSITIVE_OK:
            missing.append(f"{module} (distribution {owner}, imported in {where}) is not in requirements-backend.txt")
    assert not missing, "the container would fail at import:\n  " + "\n  ".join(missing)


def test_every_package_the_frontend_imports_or_runs_is_in_the_frontend_requirements():
    listed = requirement_names(ROOT / "requirements-frontend.txt")
    needed = {distribution_of(m) for m in third_party(ROOT / "frontend")} | {"gunicorn"}  # the CMD starts gunicorn
    assert needed - {None} <= listed, sorted((needed - {None}) - listed)


def test_the_frontend_does_not_import_the_backend_or_the_agents():
    modules = imported_modules(ROOT / "frontend")
    assert not ({"app", "asa", "openai", "chromadb", "langgraph", "torch"} & set(modules)), modules


def test_the_backend_does_not_import_the_frontend_or_test_tools():
    modules = imported_modules(ROOT / "app") | imported_modules(ROOT / "src" / "asa")
    assert not ({"frontend", "pytest", "deepeval", "promptfoo", "mlflow"} & set(modules)), sorted(modules)


CONTAINER_ONLY = {"gunicorn"}  # the production web server: the images need it, the tests and CI do not


def test_the_slim_lists_are_part_of_the_full_list_so_ci_and_the_images_agree():
    full = requirement_names(ROOT / "requirements.txt")
    for slim in ("requirements-backend.txt", "requirements-frontend.txt"):
        extra = requirement_names(ROOT / slim) - full - CONTAINER_ONLY
        assert not extra, (
            f"{slim} lists {sorted(extra)}, which CI does not install: tests could pass on a different set"
        )


def test_every_requirement_has_a_lower_bound_and_no_secret_or_url():
    for name in ("requirements-backend.txt", "requirements-frontend.txt"):
        for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                assert re.search(r">=|==|~=", line), f"{name}: {line} has no version bound"
                assert "://" not in line and "@" not in line, f"{name}: {line}"
