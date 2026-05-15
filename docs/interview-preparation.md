# Interview preparation

## Why use an agent?

Different questions need different sequences: inspect, aggregate, test, chart, or ask
for clarification. A fixed pipeline is preferable for a recurring, predefined report;
the agent is useful when the question and next useful observation vary. Compare with
the single-shot SQL baseline in `evaluation/benchmark.py` to test that claim empirically.
No live superiority has been demonstrated yet.

## How does tool selection work?

`Agent.run` sends conversation history and JSON tool schemas to the model. The
response can contain function names and serialized arguments. The registry validates
arguments before dispatch. The result becomes a tool message with the matching call
ID. The next model response therefore depends on actual observations.

Read `agent/loop.py`, `tools/registry.py` and a scripted recovery test in
`tests/test_agent.py`. You should be able to trace one invalid-column error,
one corrected call and one final cited answer.

## What state exists?

`Analysis.frame` is the dataset. `Session.messages` is provider conversation state;
`Session.history` is the audit trail. `Run` contains metrics and results for a question.
A filter does not alter the dataset. Replacing the upload resets the web session.

## How do errors and infinite loops behave?

Expected tool failures become structured observations. The model may correct its
arguments. Three consecutive failures, maximum steps/calls, context size or the reported
token threshold stop a run. An incomplete run is labeled incomplete; no extra
evidence-free summary call is made. HTTP retries are bounded and preserve context.

## Why trust the numbers?

Calculations are performed by pandas, SQLite or SciPy, with explicit missing-value
policies and tests against known answers. The model interprets results rather than
performing arithmetic from memory. This does not guarantee that its prose uses
the right number: citations are checked syntactically and claims need evaluation.

## What makes tool arguments safer?

Schemas reject unknown keys and out-of-range limits. Columns are checked against the
dataset, operations are enumerated, SQL authorization denies mutations/attachments,
and generated Python is unavailable. Uploaded text is treated as untrusted data.
These controls reduce capability; they are not a claim that prompt injection is solved.

## How are statistical conclusions bounded?

Pearson uses complete pairs and rejects constant variables; Welch requires valid groups;
chi-square warns about low expected counts and bounds table dimensions. Tests report
assumptions, not causal explanations. Multiple comparisons and dependence can invalidate
naive interpretations even when the arithmetic is right.

## What do the metrics mean?

Task success is a human-checked answer against a rubric. Numerical screens are a
preliminary check. Calls/steps describe efficiency, errors/recoveries describe robustness,
latency measures experience, and tokens/prices estimate cost. A successful retry is
not equivalent to a correct final answer. Offline checks do not measure live-model accuracy.

## What failures and fixes have actually been observed?

The first local test attempt hit Windows sandbox temporary-directory permissions,
before tool tests could run. Tests passed after using a writable temporary location.
The repository inspection identified differing loop paths and missing-evidence summary
behavior; this independent implementation uses one loop and explicit incomplete status.
Resource review caught a potentially huge chi-square contingency allocation; cardinality
is now checked before constructing the table. These are engineering findings,
not benchmark evidence of model improvement.

## What remains to learn from real runs?

Run both live systems with the same model and task subset. Inspect incorrect metric
selection, mistaken SQL, unnecessary tests, unsupported causality, bad citations and
failure to stop. Change one factor at a time and repeat a held-out subset. Record
results rather than assuming a longer prompt improves accuracy.

## Latency, cost and smaller models

Each round adds API latency and resends accumulated context. Smaller observations and
fewer unnecessary calls may help. A smaller model can use the same provider boundary
if it supports the protocol, but tool selection and recovery may degrade. A local
provider needs a deliberate transport/configuration extension; the current client
requires HTTPS. Compare models on the same benchmark rather than inferring quality
from parameter count or price.
