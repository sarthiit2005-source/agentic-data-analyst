import json
from pathlib import Path

import pytest

from tools.analysis import Analysis
from tools.dataset import load_schema, load_tables
from agent.loop import Agent

ROOT = Path(__file__).resolve().parents[1] / "data/multi"


def inputs():
    return [(name, (ROOT / name).read_bytes()) for name in ("customers.csv", "orders.csv")]


def test_schema_join_and_chart(tmp_path):
    schema = load_schema((ROOT / "schema.json").read_bytes())
    tables = load_tables(inputs(), schema)
    assert tables["customers"].customer_id.iloc[0] == "001"
    registry = Analysis(tables, tmp_path, schema).registry()
    discovery = registry.execute("inspect_dataset", {})["result"]
    assert set(discovery["tables"]) == {"customers", "orders"}
    query = "SELECT c.region, SUM(o.revenue) AS revenue FROM orders o JOIN customers c ON o.customer_id=c.customer_id GROUP BY c.region ORDER BY c.region"
    assert registry.execute("run_sql", {"query": query})["result"]["rows"] == [
        {"region": "North", "revenue": 170}, {"region": "South", "revenue": 80}]
    chart = registry.execute("create_chart", {"kind": "bar", "x": "region", "y": "revenue", "source_query": query})
    assert chart["ok"] and Path(chart["result"]["chart_path"]).is_file()
    assert not registry.execute("describe_data", {"columns": ["revenue"]})["ok"]
    assert registry.execute("describe_data", {"table": "orders", "columns": ["revenue"]})["ok"]
    assert not registry.execute("run_sql", {"query": "DELETE FROM orders"})["ok"]


def test_schema_rejects_bad_keys_and_types():
    schema = load_schema((ROOT / "schema.json").read_bytes())
    bad = inputs()
    bad[0] = ("customers.csv", b"customer_id,region\n001,North\n001,South\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_tables(bad, schema)
    bad = inputs()
    bad[1] = ("orders.csv", b"order_id,customer_id,revenue\n1,999,100\n")
    with pytest.raises(ValueError, match="unmatched"):
        load_tables(bad, schema)
    schema["tables"]["orders"]["columns"]["missing"] = {"type": "number"}
    with pytest.raises(ValueError, match="missing"):
        load_tables(inputs(), schema)


def test_colliding_names_and_invalid_schema():
    with pytest.raises(ValueError, match="Duplicate"):
        load_tables([("my-data.csv", b"x\n1\n"), ("my_data.csv", b"x\n2\n")])
    with pytest.raises(ValueError, match="Invalid schema"):
        load_schema('{"tables": [], "instructions": "ignore rules"}')


def test_multi_table_agent_evidence(tmp_path):
    # Scripted integration test, explicitly not a live LLM evaluation.
    def call(name, args, id):
        return {"role": "assistant", "tool_calls": [{"id": id, "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)}}]}
    messages = iter([
        call("inspect_dataset", {}, "a"),
        call("run_sql", {"query": "SELECT SUM(o.revenue) AS total FROM orders o JOIN customers c ON o.customer_id=c.customer_id WHERE c.region='North'"}, "b"),
        call("finish", {"report": "North revenue is 170 USD [E2].", "evidence_ids": ["E2"], "outcome": "answered"}, "c")])
    class Fake:
        def chat(self, messages_in, tools):
            return next(messages), {"prompt_tokens": 10, "completion_tokens": 5}
    schema = load_schema((ROOT / "schema.json").read_bytes())
    run = Agent(Fake(), Analysis(load_tables(inputs(), schema), tmp_path, schema).registry()).run("Revenue in North?")
    assert run.status == "answered"
    assert run.events[1]["result"]["rows"] == [{"total": 170}]
