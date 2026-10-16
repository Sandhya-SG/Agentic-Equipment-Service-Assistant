"""Turn Langfuse tracing on or off for the backend pods of the local Kubernetes cluster.

    python scripts/k8s_langfuse.py status
    python scripts/k8s_langfuse.py enable      # local Langfuse (docker-compose.langfuse.yml) on this PC
    python scripts/k8s_langfuse.py disable

`enable` does, in order: checks that Langfuse answers on this PC; checks from a cluster node that the host address the pods
will use (host.docker.internal) is the one the network rule allows; applies k8s/optional/networkpolicy-langfuse-local.yaml
(backend pods may reach that one address on port 3000, nothing else); stores the keys from .env, with an address the pods can
reach, in the secret; and restarts the backend so it picks them up. Tracing stays metadata only (LANGFUSE_CAPTURE_CONTENT is not
set), so no question or answer text is sent. The keys are never printed.

Why not k8s_secret.py: LANGFUSE_HOST in .env is http://localhost:3000, which inside a pod is the pod itself.
For a hosted Langfuse over HTTPS pass --pod-host https://... : no network rule is needed (port 443 is allowed) and `enable`
skips the policy and the node check.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAMESPACE = "aem"
POLICY = ROOT / "k8s" / "optional" / "networkpolicy-langfuse-local.yaml"
LOCAL_HEALTH = "http://localhost:3000/api/public/health"
DEFAULT_POD_HOST = "http://host.docker.internal:3000"
KEYS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST")


class LangfuseSetupError(RuntimeError):
    pass


def sh(*command: str, check: bool = True) -> str:
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)  # noqa: S603 - fixed argument lists
    if check and result.returncode != 0:
        raise LangfuseSetupError(f"{' '.join(command[:4])} failed: {(result.stderr or result.stdout).strip()[:300]}")
    return result.stdout


def kubectl(*args: str, check: bool = True) -> str:
    return sh("kubectl", "-n", NAMESPACE, *args, check=check)


def http_status(url: str) -> int:
    try:
        with urllib.request.urlopen(url, timeout=8) as response:  # noqa: S310 - the operator's own Langfuse
            return response.status
    except urllib.error.HTTPError as error:
        return error.code
    except OSError:
        return 0


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip("\"'")
    return values


def policy_address() -> str:
    """The single address the network rule allows (read from the manifest, so the two can never disagree)."""
    match = re.search(r"cidr:\s*(\d+\.\d+\.\d+\.\d+)/32", POLICY.read_text(encoding="utf-8"))
    if not match:
        raise LangfuseSetupError(f"{POLICY.name} has no single-address (/32) rule")
    return match.group(1)


def is_local(host: str) -> bool:
    return "host.docker.internal" in host or host.startswith(("http://localhost", "http://127."))


def check_local_langfuse() -> None:
    if http_status(LOCAL_HEALTH) != 200:
        raise LangfuseSetupError(
            "Langfuse does not answer on http://localhost:3000. Start it: "
            "docker compose -p aem-langfuse -f docker-compose.langfuse.yml up -d"
        )


def check_node_reaches_host(expected: str) -> None:
    nodes = kubectl("get", "nodes", "-o", "jsonpath={.items[*].metadata.name}").split()
    if not nodes:
        raise LangfuseSetupError("no cluster nodes found")
    node = nodes[-1]
    resolved = sh("docker", "exec", node, "getent", "hosts", "host.docker.internal", check=False).split()
    if not resolved or resolved[0] != expected:
        raise LangfuseSetupError(
            f"from node {node}, host.docker.internal is {resolved[0] if resolved else 'not resolvable'}, but the network rule "
            f"allows {expected}. Update k8s/optional/networkpolicy-langfuse-local.yaml to the address above first."
        )
    code = sh(
        "docker",
        "exec",
        node,
        "curl",
        "-s",
        "-o",
        "/dev/null",
        "-w",
        "%{http_code}",
        "--max-time",
        "6",
        "http://host.docker.internal:3000/api/public/health",
        check=False,
    ).strip()
    if code != "200":
        raise LangfuseSetupError(
            f"node {node} cannot reach Langfuse through host.docker.internal:3000 (HTTP {code or 'none'})"
        )


def patch_secret(data: dict[str, str | None]) -> None:
    """Merge-patch the secret. A value of None removes the key. The patch file is private and removed afterwards."""
    body = {"data": {k: (base64.b64encode(v.encode()).decode() if v is not None else None) for k, v in data.items()}}
    handle, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            json.dump(body, out)
        os.chmod(path, 0o600)
        kubectl("patch", "secret", "aem-secrets", "--type", "merge", "--patch-file", path)
    finally:
        os.remove(path)


def restart_backend() -> None:
    kubectl("rollout", "restart", "statefulset/backend")
    kubectl("rollout", "status", "statefulset/backend", "--timeout=300s")


def secret_keys() -> set[str]:
    data = json.loads(kubectl("get", "secret", "aem-secrets", "-o", "json") or "{}").get("data") or {}
    return {key for key in data if key in KEYS}


def enable(pod_host: str) -> int:
    env = {**read_env(ROOT / ".env"), **{k: v for k, v in os.environ.items() if k.startswith("LANGFUSE_") and v}}
    if not (env.get("LANGFUSE_PUBLIC_KEY") and env.get("LANGFUSE_SECRET_KEY")):
        raise LangfuseSetupError("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are needed (environment or .env)")
    if is_local(pod_host):
        check_local_langfuse()
        check_node_reaches_host(policy_address())
        print(kubectl("apply", "-f", str(POLICY)).strip())
    elif not pod_host.startswith("https://"):
        raise LangfuseSetupError("--pod-host must be https:// for a hosted Langfuse (or the local address)")
    patch_secret(
        {
            "LANGFUSE_PUBLIC_KEY": env["LANGFUSE_PUBLIC_KEY"],
            "LANGFUSE_SECRET_KEY": env["LANGFUSE_SECRET_KEY"],
            "LANGFUSE_HOST": pod_host,
        }
    )
    print(f"keys stored in the secret; the pods will send traces to {pod_host}")
    restart_backend()
    print(
        "backend restarted with tracing on. Send a few questions, then: python scripts/verify_langfuse.py --expect-commit <commit>"
    )
    return 0


def disable() -> int:
    kubectl("delete", "-f", str(POLICY), "--ignore-not-found")
    patch_secret({key: None for key in KEYS})
    restart_backend()
    print("tracing off: the network rule and the keys are removed and the backend restarted")
    return 0


def status() -> int:
    present = secret_keys()
    policy = bool(
        kubectl("get", "networkpolicy", "backend-to-local-langfuse", "-o", "name", "--ignore-not-found").strip()
    )
    local = http_status(LOCAL_HEALTH) == 200
    print(f"keys in the secret: {sorted(present) or 'none'}")
    print(
        f"tracing is {'ON' if {'LANGFUSE_PUBLIC_KEY', 'LANGFUSE_SECRET_KEY'} <= present else 'off'} for the backend pods"
    )
    print(f"network rule to local Langfuse: {'applied' if policy else 'not applied'}")
    print(f"local Langfuse on this PC: {'running' if local else 'not running'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("status", "enable", "disable"))
    parser.add_argument(
        "--pod-host", default=DEFAULT_POD_HOST, help=f"where the pods send traces (default {DEFAULT_POD_HOST})"
    )
    args = parser.parse_args(argv)
    try:
        return {"status": status, "disable": disable}.get(args.action, lambda: enable(args.pod_host))()
    except LangfuseSetupError as error:
        print(f"error: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
