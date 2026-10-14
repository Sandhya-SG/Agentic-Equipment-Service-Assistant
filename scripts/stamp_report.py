"""Add the version information to an evaluation result file, so a result says what it was measured on.

    python scripts/stamp_report.py promptfoo-tests/live-results.json [more files ...]

Adds a top-level "versions" key (git commit, models, prompt hashes, manual corpus hash, evaluation datasets and the
Promptfoo suite hash with its grader models). Run it after `promptfoo eval -o <file>`; running it again replaces the
stamp. DeepEval and `eval/run_eval.py` write their own version block.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from asa.versioning import build_info, promptfoo_config  # noqa: E402


def stamp(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    versions = build_info(include_datasets=True)
    versions["promptfoo"] = promptfoo_config()
    data["versions"] = versions
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return versions


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for name in argv:
        versions = stamp(Path(name))
        print(f"{name}: stamped (commit {versions['git']['commit'][:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
