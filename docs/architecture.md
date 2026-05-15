# Architecture decisions and reference study

## What the inspected reference actually does

Reference: https://github.com/frzzzing/Data-analysis-agent
Commit: `7bef491009b1a600587ebde32f86ec71cde00156`.

Files inspected include `agent/core/react_loop.py`, `agent/tools/registry.py`,
`agent/tools/base.py`, `agent/llm/client.py`, `agent/memory/context.py`,
`agent/tools/builtin/{sql_executor,hypothesis_test,python_executor}.py`,
`cli.py`, `requirements.txt` and the README.

The loop calls an OpenAI-compatible client, appends the assistant's tool calls to
session messages, dispatches calls through a registry, and appends tool observations.
A `finish` call or limits ends the run. Its streaming and nonstreaming paths differ,
and a separate server/WebUI stack exposes events, sessions, uploads and reports.
Tools include file reading, Python execution, SQL, hypothesis tests and skill loading.

Relevant findings from the source, not just feature claims:

- Tool argument JSON parsing falls back to an empty object on malformed input.
- SQL execution supports commits for non-SELECT statements; it is not a read-only boundary.
- Hypothesis-test arguments can pass through `eval`; this rebuild does not reuse that approach.
- The forced-summary path sends system/user messages without accumulated observations.
- The requirements file inspected did not list SciPy even though the hypothesis tool imports it.
- The README states MIT; no standalone license file was found. We do not copy its code.

The retained subset is dynamic tool selection, observation feedback, simple session
history, error correction, evidence-linked reporting, and measurable execution.
Excluded: Next.js, separate FastAPI service, MCP, generated Python execution,
long-term stores and template/skill machinery.

## 1. A single synchronous loop

Problem: the core behavior must be visible and testable. Choice: a `Client` protocol
plus a small Python loop (`agent/loop.py`). UI progress is a callback on actual
tool results. This is simpler than maintaining independent streaming/nonstreaming
engines. Tradeoff: the UI receives completed tool events, not token streaming.
Interview point: the model selects actions; Python controls validation, execution,
state transitions and limits.

## 2. Bounded operations instead of generated code

Problem: a general Python executor grants much more capability than the analysis needs.
Choice: seven specific functions and strict schemas (`tools/registry.py`,
`tools/analysis.py`). SQL's authorizer and query-only mode enforce read-only access;
a progress handler bounds long SQL execution. Every query gets a fresh in-memory
database. Tradeoff: rebuilding SQLite costs latency and large joins are out of scope.
Interview point: JSON schemas constrain arguments; they do not establish data correctness
or implement a general sandbox.

## 3. Separate data from conversation

Problem: conversational state and the dataset have different lifetimes and meanings.
Choice: an analysis object owns copies of named DataFrames; `Session` owns messages
and evidence events. Tools never mutate the canonical datasets. The Streamlit UI
reuses conversation state for the same filenames, uploaded bytes and schema.
Tradeoff: history grows, so a context limit ends the run instead of clever compression.
Interview point: filters return subsets, not hidden mutations affecting later operations.

The multiple-CSV extension loads an optional JSON schema before parsing so declared
string identifiers preserve leading zeros. It validates keys and many-to-one links.
SQL builds all tables inside one temporary database; table-bound operations delegate
to the same single-frame functions. Charts/tests may operate on a bounded read-only
source query. Source table names or SQL remain attached to tool arguments/results.

## 4. Explicit finish with citations

Problem: an apparently plausible final answer can refer to nonexistent calculations.
Choice: `finish` requires an outcome, report and evidence IDs; answered reports require
at least one valid citation. Earlier-turn evidence remains exportable.
Tradeoff: this validates provenance IDs, not whether every sentence correctly interprets
the numbers. Human benchmark grading remains necessary.
Interview point: distinguish syntactic provenance, numerical verification and semantic truth.

## 5. Recoverable errors as observations

Problem: malformed arguments and unknown columns are expected model behavior.
Choice: return bounded error messages to the model; allow another action. Count same-tool
success after failure as a recovery. Stop after three consecutive errors.
API retries preserve complete context and only retry transient connection/HTTP errors.
Tradeoff: there is no guarantee another call fixes the original analytical task.
Interview point: measure recovery separately from task success.

## 6. Evaluation before scope growth

Problem: code running is different from a reliable agent.
Choice: deterministic software tests, offline tool-oracle checks, and a separate live
36-task benchmark with a one-request SQL baseline.
Tradeoff: synthetic development data are narrow and task screening is not semantic grading.
Interview point: live results, provider/model versions, cost and human-reviewed failures
must exist before making accuracy claims. No training or fine-tuning is involved.

## Milestones

1. Study/architecture: reference inspected and decisions documented.
2. Foundations: client, loop, registry, CLI and scripted integration tests implemented.
3. Tools: seven bounded analytical tools implemented.
4. Evaluation: 36 tasks, offline checks and live comparison runner implemented;
   real-model runs pending credentials.
5. Deployment/polish: Streamlit interface, configuration and documentation implemented;
   public deployment pending hosting/repository access.
