"""Minimal Streamlit interface sharing the CLI's agent and tools."""
import hashlib
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import streamlit as st

from agent.llm_client import APIClient
from agent.loop import Agent
from agent.reporting import markdown
from agent.state import Session
from tools.analysis import Analysis
from tools.dataset import load_tables, load_schema

st.set_page_config(page_title="Agentic Data Analyst", page_icon="📊", layout="wide")
st.title("Agentic Data Analyst")
st.caption("Ask a question. Follow the analysis. Inspect the evidence.")
st.info("Selected data samples and computed tool results are sent to your chosen LLM provider. Use data you are permitted to share.")
with st.sidebar:
    st.header("Connection")
    key = st.text_input("Your API key", type="password")
    model = st.text_input("Tool-calling model", value=os.getenv("LLM_MODEL", ""))
    api_mode_label = st.selectbox("API mode", ["Auto", "Responses", "Chat Completions"])
    api_mode = {"Auto": "auto", "Responses": "responses",
                "Chat Completions": "chat_completions"}[api_mode_label]
    endpoint = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    st.caption(f"Provider endpoint: {endpoint}")
    st.caption("Bring your own key. It is kept only in this browser session. Each question is limited to 12 model steps and 32 tool calls.")
    reset = st.button("Start a new session")
    sample = st.selectbox("Sample dataset", ["Upload my own CSV", "Sales", "Experiments", "Operations"])

uploads = st.file_uploader("CSV datasets · up to 10 files, 20 MiB total", type=["csv"], accept_multiple_files=True)
schema_file = st.file_uploader("Optional schema (JSON)", type=["json"])
st.caption("CSV filenames become table names. A schema can describe column types, primary keys, and many-to-one relationships.")
schema_raw = schema_file.getvalue() if schema_file else None
files = [(upload.name, upload.getvalue()) for upload in uploads]
if not files and sample != "Upload my own CSV":
    files = [(f"{sample.lower()}.csv", (Path(__file__).parent / "data" / f"{sample.lower()}.csv").read_bytes())]
if files:
    digest = hashlib.sha256()
    for name, raw in files:
        digest.update(name.encode())
        digest.update(hashlib.sha256(raw).digest())
    digest.update(schema_raw or b"")
    fingerprint = digest.hexdigest()
    if reset or st.session_state.get("fingerprint") != fingerprint:
        try:
            schema = load_schema(schema_raw)
            tables = load_tables(files, schema)
            old_directory = st.session_state.get("directory")
            if old_directory:
                old_directory.cleanup()
            directory = tempfile.TemporaryDirectory(prefix="data-analyst-")
            st.session_state.update(fingerprint=fingerprint, tables=tables, schema=schema,
                                    session=Session(), directory=directory, last_run=None)
        except Exception as error:
            st.error(f"Cannot load datasets/schema: {error}")
            st.stop()
    for name, frame in st.session_state.tables.items():
        st.caption(f"{name} · {len(frame):,} rows · {len(frame.columns)} columns")
        st.dataframe(frame.head(10), hide_index=True)
    with st.expander("Schema and relationships"):
        st.json(st.session_state.schema)
    question = st.text_area("What would you like to investigate?", max_chars=4000,
                            placeholder="Which region has the lowest total revenue?")
    if st.button("Analyze", type="primary", disabled=not (key and model and question.strip())):
        try:
            client = APIClient(api_key=key, model=model, base_url=endpoint, api_mode=api_mode)
            analysis = Analysis(st.session_state.tables, st.session_state.directory.name, st.session_state.schema)
            with st.status("Analyzing your dataset…", expanded=True) as status:
                def show_event(event):
                    st.write(f"{event['evidence_id']} · {event['tool']} · {'done' if event['ok'] else 'needs correction'}")
                run = Agent(client, analysis.registry()).run(question, st.session_state.session, show_event)
                status.update(label=f"Analysis: {run.status}", state="complete" if run.status in ("answered", "clarification", "unanswerable") else "error")
            st.session_state.last_run = run
        except Exception as error:
            st.error(str(error))
    run = st.session_state.get("last_run")
    if run:
        st.markdown(run.answer)
        for event in run.events:
            path = event.get("result", {}).get("chart_path")
            if path and Path(path).is_file():
                st.image(path)
                st.download_button("Download chart", Path(path).read_bytes(),
                                   file_name=Path(path).name, mime="image/png", key=path)
        st.download_button("Download report", markdown(run), file_name="analysis.md", mime="text/markdown")
        with st.expander("Execution history and metrics"):
            st.json(asdict(run))
