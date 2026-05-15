# Evaluation protocol

`python -m evaluation.build_tasks` creates three deterministic synthetic datasets
and 36 tasks. Seed: 42. Data are authored for this project; no external data license
or personal information is involved. These are a development benchmark, not a held-out
generalization claim. Add real, permitted datasets and separate tuning from test tasks
before making portfolio accuracy claims.

## Three separate kinds of evidence

1. `pytest`: deterministic unit/integration tests with an explicitly scripted LLM and
   mocked HTTP transport. Checks software behavior, not a real model's competence.
2. `python -m evaluation.benchmark`: executes reference tool calls against known
   answers. Six conversational tasks are not applicable. The injected-recovery task
   checks only its numeric oracle offline; recovery is exercised in unit tests and live runs.
3. `python -m evaluation.benchmark --mode live --system agent --output evaluation/results/live-agent.json`:
   actual model requests. Requires configured credentials. Repeat with
   `--system single-shot-sql --output evaluation/results/live-baseline.json`.

The single-shot baseline gets the same dataset schema and question, chooses SQL once,
and cannot see results or repair errors. Compare numeric/aggregation/filtering/missing-data/
multi-step tasks as the common subset. Charts, statistical tests, clarification and
injected recovery deliberately expose coverage differences and must be reported separately.

## Grading

Each task records a question, category and expected result. Numeric screens accept
0.5% relative or 0.00005 absolute tolerance in the final answer. Screens can produce
false positives (a coincidental number or the wrong metric) and false negatives
(unusual formatting). Chart screens require the requested tool arguments.
Conversational screens check the outcome only.

**Screen pass rate is not task success rate.** Review every full trace and set
`manual_task_success` to true/false only after verifying:

- Correct quantity, dataset scope, sign, units, missing-value treatment, and rounding.
- Every important numerical claim cites an appropriate successful tool result.
- No invented causal claim or unsupported explanation.
- Statistical assumptions and small-sample limitations are disclosed.
- Requested chart exists and its axes and aggregation match the question.
- Unanswerable/ambiguous requests receive a useful limitation/clarifying question.
- Recovery task actually retries the failed operation and reaches the right answer.

Label failures as numerical, tool_selection, invalid_arguments, unsupported_claim,
missing_citation, ambiguity, chart, api_error, or limit. Publish verified success
over all tasks and by category, plus numerator/denominator. Never discard failed API runs.

Trace metrics include steps, tool calls, errors, same-tool recoveries, latency,
provider-reported input/output tokens, and estimated USD cost when both per-million
token prices are configured. A same-tool recovery is a successful call after a
failure for that tool; it does not itself establish task success. Provider retries
can incur costs not reflected in successful-response usage. No prices are guessed.
