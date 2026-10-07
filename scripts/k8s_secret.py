"""Create or update the Kubernetes secret `aem-secrets` from your local .env.

    python scripts/k8s_secret.py

Reads OPENAI_API_KEY from .env (or the environment) and applies it to the `aem` namespace. The key is
passed to kubectl through a private temporary file, never on the command line, and is never printed or
written into the repository. Run it after `kubectl apply -k k8s` has created the namespace.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

NAMESPACE = "aem"
SECRET_NAME = "aem-secrets"
KEYS = ("OPENAI_API_KEY",)


def find_kubectl() -> str:
    found = shutil.which("kubectl")
    if found:
        return found
    docker_bin = (
        Path(os.getenv("LOCALAPPDATA", "")) / "Programs" / "DockerDesktop" / "resources" / "bin" / "kubectl.exe"
    )
    if docker_bin.exists():
        return str(docker_bin)
    sys.exit("kubectl not found. Install it or add Docker Desktop's bin folder to PATH.")


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(".env"))
    except ImportError:
        pass

    values = {key: os.getenv(key, "").strip() for key in KEYS}
    missing = [key for key, value in values.items() if not value]
    if missing:
        sys.exit(f"Missing in .env or the environment: {', '.join(missing)}")

    kubectl = find_kubectl()
    fd, temp_path = tempfile.mkstemp(suffix=".env")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.writelines(f"{key}={value}\n" for key, value in values.items())
        os.chmod(temp_path, 0o600)
        rendered = subprocess.run(
            [
                kubectl,
                "create",
                "secret",
                "generic",
                SECRET_NAME,
                "-n",
                NAMESPACE,
                f"--from-env-file={temp_path}",
                "--dry-run=client",
                "-o",
                "yaml",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        applied = subprocess.run([kubectl, "apply", "-f", "-"], input=rendered.stdout, capture_output=True, text=True)
    finally:
        os.remove(temp_path)

    if applied.returncode != 0:
        sys.exit(f"kubectl apply failed: {applied.stderr.strip()}")
    print(f"secret/{SECRET_NAME} configured in namespace {NAMESPACE} (keys: {', '.join(KEYS)})")


if __name__ == "__main__":
    main()
