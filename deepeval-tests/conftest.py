"""Shared setup for the DeepEval grounding tests.

The tests run the real pipeline (request -> planning -> specialist) in the same process, not through HTTP,
because DeepEval's faithfulness metric needs the retrieved passages as context and the API only returns their
labels. Needs the real OPENAI_API_KEY (the agents and the judge model both use it) and the manuals in aem_documents.
Run from the repository root with the venv created in deepeval-tests/README (see DEEPEVAL_INSTRUCTIONS.md).
"""

import json
import os
import sys
from pathlib import Path

import pytest

# DeepEval gives each judge call 88 seconds by default. gpt-4.1 judging long answers three times at once can need
# longer, and a timeout is an infrastructure error, not a score, so the allowance is raised (override to change).
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "300")

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

try:  # the key lives in .env for local runs
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:  # pragma: no cover
    pass

GOLD_SET = REPO_ROOT / "eval" / "datasets" / "gold_set.jsonl"
RESULTS_FILE = Path(__file__).resolve().parent / "deepeval-results.json"

# The judge. gpt-4.1 is much stronger than the gpt-4o-mini the agents use, runs at temperature 0 and returns the
# log-probabilities the G-Eval score is weighted by. (DeepEval's own default, a gpt-5 reasoning model, samples at
# temperature 1, which is noisier.) Override with DEEPEVAL_JUDGE_MODEL.
JUDGE_MODEL = os.getenv("DEEPEVAL_JUDGE_MODEL", "gpt-4.1")

# Every metric is measured this many times per answer and the MEDIAN is used, so one odd verdict from the judge
# cannot flip a question. The spread between the runs is reported as the measured judge noise.
REPEATS = max(1, int(os.getenv("DEEPEVAL_REPEATS", "3")))

# Definitions. The first version of this suite required all three metrics to pass for a question to count as
# grounded. That mixed two different things, so it was split. This is recorded openly because the split was proposed
# after seeing results; the old combined rate is still reported (strict_rate) so runs stay comparable.
#   grounded  = faithfulness >= 0.7 AND no-invented-specifics >= 0.7   (does the answer agree with the manual?)
#   relevant  = answer relevancy >= 0.7                                (does the answer address the question?)
# PRD: grounding of at least 90%. Relevancy is a separate quality with its own 90% target.
GROUNDING_TARGET = 0.90
RELEVANCY_TARGET = 0.90
FAITHFULNESS_THRESHOLD = 0.7
INVENTED_THRESHOLD = 0.7
RELEVANCY_THRESHOLD = 0.7


def load_gold_set() -> list[dict]:
    with GOLD_SET.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@pytest.fixture(scope="session", autouse=True)
def _require_key_and_index():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set: these tests call the real model", allow_module_level=True)
    from app.services.graph_runner import is_ready, warm_up

    if not (is_ready() or warm_up()):
        pytest.skip("the agents or the manual index could not be loaded", allow_module_level=True)


_pipeline_cache: dict[str, dict] = {}


@pytest.fixture(scope="session")
def pipeline():
    """Run the pipeline once per question and reuse the result for every metric."""
    from app.services.graph_runner import run_graph

    def run(case: dict) -> dict:
        if case["id"] not in _pipeline_cache:
            _pipeline_cache[case["id"]] = run_graph(case["question"], case["equipment_model"])
        return _pipeline_cache[case["id"]]

    return run


def pytest_sessionfinish(session, exitstatus):
    """Write the scores gathered by the tests, so the report can quote them."""
    from test_grounding import SCORES

    if SCORES:
        RESULTS_FILE.write_text(json.dumps(SCORES, indent=2), encoding="utf-8")
