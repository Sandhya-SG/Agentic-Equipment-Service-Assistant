# DeepEval grounding tests

Scored evaluation of whether the answers agree with the manual (PRD target: grounding at least 90%). Modelled on llmapp09's `deepeval-tests`, but the pipeline is called in the same process, because DeepEval's faithfulness metric needs the retrieved passages and the API only returns their labels.

## What it measures

For each question in `eval/datasets/gold_set.jsonl` (16 questions today, 10 for the Thermal Station and 6 for the 1kW retrofit; the plan asks for 30, and an SME should review them) the real pipeline answers, then a judge model scores three metrics:

| Metric | Question it answers | Threshold |
|---|---|---|
| Faithfulness | Is every claim in the answer supported by the retrieved manual passages? | 0.7 |
| No invented specifics (G-Eval) | Does any number, part, step or warning appear that the passages do not contain? | 0.7 |
| Answer relevancy | Does the answer address the question? | 0.7 |

## Two rates, each with a 90% target

| Rate | Counts a question when | Question it answers |
|---|---|---|
| **Grounding** | faithfulness and no-invented-specifics both pass | Does the answer agree with the manual? (the PRD's "grounding") |
| **Relevancy** | answer relevancy passes | Does the answer address the question? |

The first version of this suite required all three metrics to pass for one combined rate. Splitting grounding from relevancy was proposed after seeing results, so the old combined rate is still written to the results as `strict_rate`, and the change is recorded in `conftest.py`.

## How the numbers are made steadier

- **A stronger judge.** `gpt-4.1` at temperature 0 (`DEEPEVAL_JUDGE_MODEL` to change). The agents use the smaller `gpt-4o-mini`; DeepEval's own default is a reasoning model that samples at temperature 1, which is noisier.
- **Three repeats and the median.** Every metric is measured `DEEPEVAL_REPEATS` times (default 3) per answer, concurrently, and the median decides. One odd verdict can no longer flip a question.
- **Noise is measured.** The spread (largest minus smallest score across the repeats) is stored per metric and summarised as `mean_judge_spread` and `max_judge_spread`. A spread of 0 means the repeats agreed.
- **Errors are not scores.** A judge timeout or rate limit is retried; if it keeps failing the question is reported as *unscored*, the aggregate tests fail ("the rate is not valid"), and it is never dropped from the denominator. The per-call allowance is raised to 300 s (`DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE`).

With the earlier judge (`gpt-4o-mini`, one measurement per metric) the same answers moved by about 20 points between runs (72.7% to 93.8%). With `gpt-4.1` and the median the repeats agree almost exactly (mean spread about 0.01).

Questions the Request agent sends back for clarification, or that end in an escalation, have no answer to score. They are skipped, counted and listed with the reason in `deepeval-results.json` (the over-ask rate is a finding in itself).

## Set up (once)

The DeepEval dependencies are large, so they live in their own venv and stay out of `requirements.txt` and the container images.

```bash
python -m venv --system-site-packages deepeval-tests/.venv
deepeval-tests/.venv/Scripts/python -m pip install -r deepeval-tests/requirements.txt
# make the project's own packages visible (appended last, so the venv's versions win):
echo "<repo>\.venv\Lib\site-packages" > deepeval-tests/.venv/Lib/site-packages/project_venv.pth
```

## Run (from the repository root)

```bash
# real OPENAI_API_KEY in .env or the environment; the ChromaDB index must be built (scripts/ingest.py)
PYTHONUTF8=1 DEEPEVAL_TELEMETRY_OPT_OUT=YES deepeval-tests/.venv/Scripts/python -m pytest deepeval-tests -v -rs
```

Results are written to `deepeval-tests/deepeval-results.json`: per question the median scores, every repeat, the spread, whether it is grounded and relevant and the judge's reason for any low score; the skipped and unscored questions; and the summary (judge, repeats, grounding and relevancy rates, mean scores, judge noise).

## Cost and limits

About 16 pipeline runs plus about 16 x 3 x 10 judge calls on `gpt-4.1`: a few dollars at most, and about 25 minutes. It is a real-key run, so keep it out of CI (CI has no ChromaDB index or key). The faithfulness check compares the answer with the passages the agent retrieved, not with the whole manual: if retrieval misses the relevant page, a faithful answer to the wrong passages still scores well. The judge is a model, so a disputed verdict should be checked by a person; the judge's reason is saved with every low score for that purpose.
