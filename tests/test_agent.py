import json

import httpx
import pandas as pd

from agent.llm_client import APIClient
from agent.loop import Agent
from agent.state import Session
from tools.analysis import Analysis


def call(name, args, id="call"):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


class ScriptedClient:
    """Test double only. Never used in the product or labeled as live evaluation."""
    def __init__(self, messages):
        self.messages = iter(messages)

    def chat(self, messages, tools):
        return next(self.messages), {"prompt_tokens": 10, "completion_tokens": 5}


def registry(tmp_path):
    return Analysis(pd.DataFrame({"sales": [10, 20]}), tmp_path).registry()


def test_recovery_evidence_and_usage(tmp_path):
    client = ScriptedClient([
        call("describe_data", {"columns": ["revenue"]}),
        call("run_sql", {"query": "SELECT SUM(sales) AS total FROM data"}),
        call("describe_data", {"columns": ["sales"]}),
        call("finish", {"report": "Total sales: 30 [E2].", "evidence_ids": ["E2"], "outcome": "answered"})])
    session = Session()
    run = Agent(client, registry(tmp_path), input_price=1, output_price=2).run("Total?", session)
    assert run.status == "answered" and run.errors == 1 and run.recovered_tools == 1
    assert run.input_tokens == 40 and run.output_tokens == 20
    assert run.estimated_cost_usd == 0.00008
    assert [m["role"] for m in session.messages].count("tool") == 4


def test_unknown_evidence_rejected(tmp_path):
    client = ScriptedClient([call("finish", {"report": "100 [E99]", "evidence_ids": ["E99"], "outcome": "answered"})] * 3)
    run = Agent(client, registry(tmp_path)).run("Total?")
    assert run.status == "error_limit"


def test_total_and_citation_format_recovery(tmp_path):
    client = ScriptedClient([
        call("aggregate_data", {"group_by": [], "column": "sales", "operation": "sum"}),
        call("finish", {"report": "Total 30. Evidence: E1", "evidence_ids": ["E1"], "outcome": "answered"}),
        call("finish", {"report": "Total 30 [E1].", "evidence_ids": ["E1"], "outcome": "answered"})])
    run = Agent(client, registry(tmp_path)).run("Total sales?")
    assert run.status == "answered" and run.errors == 1
    assert "[E1]" in run.events[1]["error"]
    assert "Retry finish" in run.events[1]["error"]


def test_step_limit_no_fabricated_summary(tmp_path):
    client = ScriptedClient([call("inspect_dataset", {})] * 2)
    run = Agent(client, registry(tmp_path), max_steps=2).run("Analyze")
    assert run.status == "step_limit" and "incomplete" in run.answer


def test_multiple_calls_finish_stops_execution(tmp_path):
    message = call("finish", {"report": "Which measure?", "evidence_ids": [], "outcome": "clarification"}, "a")
    message["tool_calls"] += call("inspect_dataset", {}, "b")["tool_calls"]
    session = Session()
    run = Agent(ScriptedClient([message]), registry(tmp_path)).run("Best?", session)
    assert run.status == "clarification" and run.tool_calls == 1
    assert not run.events[-1]["ok"]
    assert len([m for m in session.messages if m["role"] == "tool"]) == 2


def test_http_client_preserves_wire_format():
    def handler(request):
        payload = json.loads(request.content)
        assert payload["model"] == "test-model" and "tools" in payload
        return httpx.Response(200, json={"choices": [{"message": call("inspect_dataset", {})}],
                                        "usage": {"prompt_tokens": 2, "completion_tokens": 3}})
    client = APIClient("not-real", "test-model", transport=httpx.MockTransport(handler))
    message, usage = client.chat([{"role": "user", "content": "Analyze"}], [])
    assert message["tool_calls"][0]["function"]["name"] == "inspect_dataset"
    assert usage["completion_tokens"] == 3


def test_responses_api_tool_call_and_history_translation():
    seen = []
    def handler(request):
        payload = json.loads(request.content)
        seen.append((request.url.path, payload))
        return httpx.Response(200, json={
            "output": [{"type": "function_call", "call_id": "fc_1",
                        "name": "inspect_dataset", "arguments": "{}"}],
            "usage": {"input_tokens": 7, "output_tokens": 3}})
    client = APIClient("not-real", "gpt-5.6-sol", transport=httpx.MockTransport(handler))
    message, usage = client.chat([
        {"role": "system", "content": "Analyze."},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "old", "type": "function",
            "function": {"name": "inspect_dataset", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "old", "content": "{\"ok\":true}"}],
        [{"type": "function", "function": {"name": "inspect_dataset",
                                               "description": "Inspect", "parameters": {"type": "object"}}}])
    assert seen[0][0].endswith("/responses")
    assert seen[0][1]["tools"][0]["name"] == "inspect_dataset"
    assert any(item.get("type") == "function_call_output" for item in seen[0][1]["input"])
    assert message["tool_calls"][0]["function"]["name"] == "inspect_dataset"
    assert usage == {"prompt_tokens": 7, "completion_tokens": 3}


def test_provider_error_message_is_preserved_without_key():
    def handler(request):
        return httpx.Response(400, json={"error": {"message": "Use max_output_tokens",
                                                    "code": "unsupported_parameter"}})
    client = APIClient("secret-key", "gpt-4.1", transport=httpx.MockTransport(handler))
    try:
        client.chat([], [])
        assert False, "Expected an API error"
    except RuntimeError as error:
        assert "unsupported_parameter" in str(error)
        assert "Use max_output_tokens" in str(error)
        assert "secret-key" not in str(error)


def test_followup_keeps_prior_evidence(tmp_path):
    client = ScriptedClient([
        call("run_sql", {"query": "SELECT SUM(sales) AS total FROM data"}),
        call("finish", {"report": "30 [E1]", "evidence_ids": ["E1"], "outcome": "answered"}),
        call("finish", {"report": "The same total is 30 [E1].", "evidence_ids": ["E1"], "outcome": "answered"})])
    agent = Agent(client, registry(tmp_path))
    session = Session()
    agent.run("Total?", session)
    run = agent.run("Repeat the total.", session)
    assert run.status == "answered"
    assert run.evidence_log[0]["evidence_id"] == "E1"
    from agent.reporting import markdown
    assert "Evidence from earlier questions" in markdown(run)


def test_tool_call_budget_closes_protocol_batch(tmp_path):
    message = call("inspect_dataset", {}, "a")
    message["tool_calls"] += call("inspect_dataset", {}, "b")["tool_calls"]
    session = Session()
    run = Agent(ScriptedClient([message]), registry(tmp_path), max_calls=1).run("Analyze", session)
    assert run.status == "tool_limit" and run.tool_calls == 1
    assert len([m for m in session.messages if m["role"] == "tool"]) == 2
