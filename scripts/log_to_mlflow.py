"""Log evaluation results to MLflow, so runs can be compared side by side (before and after a change).

    python scripts/log_to_mlflow.py promptfoo-tests/free-results.json deepeval-tests/deepeval-results.json report.json

Each file becomes one MLflow run in the experiment `aem-evaluation` (override with `--experiment`). The run carries
the build as parameters (git commit, prompt hash, corpus hash, models, judge model, rubric hash, dataset hash) and every
numeric summary value as a metric. MLflow is optional and not a backend dependency: `pip install mlflow` first. With no
`MLFLOW_TRACKING_URI` the runs are written to a local `mlruns/` folder (view them with `mlflow ui`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

PARAM_LIMIT = 250  # MLflow parameter values are limited to 500 characters; keep them short


def flatten_metrics(data: object, prefix: str = "", depth: int = 0) -> dict[str, float]:
    """Every number in a (nested) summary as `a.b.c: value`. Lists and booleans are not metrics."""
    metrics: dict[str, float] = {}
    if depth > 3 or isinstance(data, bool):
        return metrics
    if isinstance(data, (int, float)):
        metrics[prefix] = float(data)
    elif isinstance(data, dict):
        for key, value in data.items():
            if key in ("versions", "rubric", "cases", "rows", "failures", "skipped", "errors"):
                continue
            metrics.update(flatten_metrics(value, f"{prefix}.{key}" if prefix else str(key), depth + 1))
    return metrics


def build_params(data: dict) -> dict[str, str]:
    """The build and the evaluation settings of a result file, as short strings."""
    versions = data.get("versions") or (data.get("summary") or {}).get("versions") or {}
    summary = data.get("summary") or data
    params: dict[str, str] = {}
    git = versions.get("git") or {}
    if git.get("commit"):
        params["git_commit"] = git["commit"][:12]
        params["git_dirty"] = str(git.get("dirty"))
    if versions.get("prompts"):
        params["prompts_sha256"] = versions["prompts"]["sha256"][:12]
    if versions.get("corpus"):
        params["corpus_sha256"] = versions["corpus"]["sha256"][:12]
    if versions.get("datasets"):
        params["datasets_sha256"] = versions["datasets"]["sha256"][:12]
    if versions.get("models"):
        params["models"] = ",".join(sorted(set(versions["models"].values())))[:PARAM_LIMIT]
    for key in ("judge_model", "fallback_judge_model", "repeats"):
        if summary.get(key) is not None:
            params[key] = str(summary[key])
    rubric = summary.get("rubric") or {}
    if rubric:
        params["rubric_version"] = str(rubric.get("version"))
        params["rubric_sha256"] = str(rubric.get("sha256", ""))[:12]
    return params


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="result files (JSON)")
    parser.add_argument("--experiment", default="aem-evaluation")
    args = parser.parse_args(argv)
    try:
        import mlflow
    except ImportError:
        print("mlflow is not installed: pip install mlflow")
        return 1
    mlflow.set_experiment(args.experiment)
    for name in args.files:
        path = Path(name)
        data = json.loads(path.read_text(encoding="utf-8"))
        with mlflow.start_run(run_name=path.stem):
            mlflow.log_param("result_file", path.name)
            mlflow.log_params(build_params(data))
            for key, value in flatten_metrics(data.get("summary") or data.get("report") or data).items():
                mlflow.log_metric(key.replace(" ", "_")[:240], value)
            mlflow.log_artifact(str(path))
        print(f"{name}: logged to MLflow experiment {args.experiment}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
