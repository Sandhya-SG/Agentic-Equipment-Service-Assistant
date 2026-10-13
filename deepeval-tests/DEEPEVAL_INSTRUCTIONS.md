# DeepEval grounding tests

Scored evaluation of whether the answers agree with the manual (PRD target: grounding at least 90%). Modelled on llmapp09's `deepeval-tests`, but the pipeline is called in the same process, because DeepEval's faithfulness metric needs the retrieved passages and the API only returns their labels.

## What it measures

For each question in `eval/datasets/gold_set.jsonl` (16 questions today, 10 for the Thermal Station and 6 for the 1kW retrofit; the plan asks for 30, and an SME should review them) the real pipeline answers, then a judge model (`gpt-4o-mini`, override with `DEEPEVAL_JUDGE_MODEL`) scores:

| Metric | Question it answers | Threshold |
|---|---|---|
| Faithfulness | Is every claim in the answer supported by the retrieved manual passages? | 0.7 |
| Answer relevancy | Does the answer address the question? | 0.7 |
| No invented specifics (G-Eval) | Does any number, part, step or warning appear that the passages do not contain? | 0.7 |

A question counts as grounded when all three pass. `test_zz_grounding_rate_meets_the_prd_target` fails if fewer than 90% of the gradable answers are grounded. Questions the Request agent sends back for clarification, or that end in an escalation, have no answer to score: they are skipped, counted and listed in `deepeval-results.json` (the over-ask rate is a finding in itself).

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
# real OPENAI_API_KEY in .env or the environment; the manuals must be in aem_documents/
PYTHONUTF8=1 DEEPEVAL_TELEMETRY_OPT_OUT=YES deepeval-tests/.venv/Scripts/python -m pytest deepeval-tests -v -rs
```

Results are written to `deepeval-tests/deepeval-results.json` (scores per question, skipped questions and their reasons, the grounding rate).

## Cost and limits

About 16 pipeline runs (several model calls each) plus about 50 judge calls: cents, but it is a real-key run, so keep it out of CI (CI also has no manuals). The judge is a model, so scores vary slightly between runs and a borderline question may flip. Treat the rate as an estimate and keep the per-question scores in the report.
