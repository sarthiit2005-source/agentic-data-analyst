# Deployment

## Run locally

Install `requirements.txt` in a virtual environment, then:

```powershell
.\.venv\Scripts\python -m streamlit run app.py
```

The app asks each visitor for their own API key and model. The owner does not
need to publish a shared key. For another compatible provider, the administrator
sets `LLM_BASE_URL` in the server environment; visitors cannot choose arbitrary
server request destinations. The CLI reads `.env`; the web app uses environment
variables and its session-only key field.

## Streamlit Community Cloud

1. Push the project source to a GitHub repository you control. Exclude `.env`,
   `.venv`, `.reference`, user datasets, outputs and local secrets.
2. Create an app in Community Cloud. Select that repository/branch and `app.py`
   as the entrypoint.
3. Select Python 3.13 in advanced settings to match the local verification runtime.
4. Keep the root `requirements.txt` and `.streamlit/config.toml`. No OS packages
   or separate server are required.
5. Optionally configure top-level `LLM_MODEL` and `LLM_BASE_URL` in the hosting
   environment/secrets. Do not set an owner key for a public unrestricted demo.
6. Launch, inspect build logs, open the actual assigned URL and upload the synthetic
   `data/sales.csv`.

These steps follow the official
[deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
and [dependency guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

## Verify before claiming deployment

- Load the assigned URL from a separate browser session.
- Run a real question using a permitted provider key.
- Verify the Q2→Q3 totals against `data/sales.csv`.
- Confirm evidence, chart download and report download work.
- Confirm an invalid-column question produces correction or a useful limitation.
- Confirm oversized uploads fail and a second session cannot see the first's history.
- Record host URL, source revision, provider/model, date and test outcomes.

**Public deployment has not been completed or verified.**
No deployment URL is claimed. Hosting access and a user-owned repository are still needed.

## Operating constraints

Use provider-side spending limits. The app bounds steps, tool calls, output tokens,
conversation size and per-session turns; these are not a global spending guarantee.
Restarting a session resets session limits. API keys are not written to reports or logs.
Temporary files disappear on session cleanup/process exit and are not durable storage.
For an open, high-traffic service, add host-level concurrency/rate controls and isolated
workers before relying on application limits alone.
