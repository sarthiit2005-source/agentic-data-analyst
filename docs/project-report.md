# Project report

## Delivered

Independent Python implementation, custom dynamic tool-calling loop, API client,
seven analytical tools, CLI, Streamlit interface, session history, Markdown reports,
charts, JSON traces, tests, synthetic data, 36 benchmark tasks, a live comparison runner,
architecture decisions, deployment instructions and interview notes.

No scikit-learn dependency, training, fine-tuning or predictive modeling was added.

## Evidence and current limitations

The completed suite passed 31 tests, including schema validation, multiple CSVs,
joined charts and a scripted cross-table evidence flow; the verification record is in
`evaluation/results/verification.txt`. The local health endpoint returned HTTP 200,
and the sample sales-data view was inspected in a browser; its screenshot is
`docs/images/app-preview.png`. Offline benchmark: 30 applicable tool checks
passed; six conversational tasks not applicable offline. These checks include the
numerical oracle for the recovery task, not a live recovery measurement.

Live LLM analysis and baseline comparison remain pending because no API key/model
configuration was supplied. Public deployment remains pending user-owned GitHub/
hosting access. No task accuracy, live cost or public URL is invented.

## Next concrete experiment

Configure a compatible model and key locally. Run a two-task live smoke test, inspect
the reports, then run the full agent and SQL baseline benchmark. Grade traces using
`evaluation/README.md`. Fix the most frequent demonstrated failure, and test on new
held-out questions before reporting improvements.
