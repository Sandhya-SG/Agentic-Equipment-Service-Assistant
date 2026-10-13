"""Check that a running backend (Docker, Kubernetes, AWS) is the build you think it is.

    python scripts/verify_version.py http://localhost:8000
    python scripts/verify_version.py https://aem.example.com --token $VERSION_TOKEN --expect-commit <sha>

It reads GET /version from the backend and compares it with this checkout (or with `--expect-commit`): the commit, the
system prompts, the manual corpus hash and the models. Exit code 0 means they match, 1 means something differs.
In Kubernetes: `kubectl port-forward svc/<backend> 8000:8000` first, then use http://localhost:8000.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from asa.versioning import build_info, compare  # noqa: E402


def fetch(url: str, token: str | None) -> dict:
    headers = {"X-Version-Token": token} if token else {}
    request = urllib.request.Request(url.rstrip("/") + "/version", headers=headers)
    with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310 - the URL is the operator's own
        return json.loads(response.read())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url", help="base URL of the backend")
    parser.add_argument("--token", default=None, help="the VERSION_TOKEN of the backend, if it has one")
    parser.add_argument("--expect-commit", default=None, help="the commit the deployment should be (default: HEAD)")
    args = parser.parse_args(argv)

    try:
        running = fetch(args.url, args.token)
    except urllib.error.HTTPError as error:
        reason = "token missing or wrong" if error.code == 401 else "endpoint disabled"
        print(f"cannot read /version: HTTP {error.code} ({reason})")
        return 1
    except (urllib.error.URLError, OSError, ValueError) as error:
        print(f"cannot read /version: {error}")
        return 1

    expected = build_info()
    if args.expect_commit:
        expected["git"] = {"commit": args.expect_commit, "dirty": None, "source": "argument"}
    problems = compare(expected, running)
    corpus = ((running.get("corpus") or {}).get("sha256") or "none")[:12]
    print(
        f"running: commit {running['git']['commit'][:12]}, prompts {running['prompts']['sha256'][:12]}, corpus {corpus}"
    )
    if problems:
        print("MISMATCH:\n  - " + "\n  - ".join(problems))
        return 1
    print("OK: the running backend matches the expected build")
    return 0


if __name__ == "__main__":
    sys.exit(main())
