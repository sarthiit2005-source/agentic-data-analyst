"""Offline oracle checks and live agent/single-shot SQL baseline evaluations."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import time

from agent.llm_client import APIClient
from agent.loop import Agent
from agent.state import Run
from cli import price
from tools.analysis import Analysis
from tools.dataset import load_csv

ROOT = Path(__file__).resolve().parents[1]


def lookup(value, path):
    for key in path:
        value = value[key]
    return value


def close(a, b):
    return isinstance(a, (int, float)) and math.isclose(a, b, rel_tol=0.005, abs_tol=0.00005)


def numbers(text):
    # Exclude citation IDs before screening report numbers.
    text = re.sub(r"\[E\d+\]", "", text).replace(",", "")
    return [float(x) for x in re.findall(r"(?<![A-Za-z])[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)]


def screen(task, run):
    """A screening check, NOT a substitute for semantic grading."""
    if task["category"] in ("clarification", "unanswerable"):
        return run.status == task["expected"]
    if run.status != "answered":
        return False
    if task["category"] == "chart":
        return any(e["ok"] and e["tool"] == "create_chart" and
                   json.loads(e["arguments"]) == task["oracle"]["arguments"]
                   for e in run.events)
    found = any(close(n, task["expected"]) for n in numbers(run.answer))
    if task.get("inject_error"):
        found = found and run.errors > 0 and run.recovered_tools > 0
    return found


def single_shot(client, frame, registry, question):
    """One LLM request, one SQL call, no repair or observation feedback."""
    start = time.perf_counter()
    run = Run()
    schemas = [s for s in registry.schemas() if s["function"]["name"] == "run_sql"]
    try:
        message, usage = client.chat([
            {"role": "system", "content": "Answer by calling run_sql exactly once on SQLite table data. Only read-only SQL is permitted."},
            {"role": "user", "content": f"Schema: {[(c, str(frame[c].dtype)) for c in frame]}\nQuestion: {question}"}], schemas)
        run.steps = 1
        run.input_tokens = usage.get("prompt_tokens", 0)
        run.output_tokens = usage.get("completion_tokens", 0)
        run.usage_complete = "prompt_tokens" in usage and "completion_tokens" in usage
        calls = message.get("tool_calls") or []
        if len(calls) != 1 or calls[0]["function"]["name"] != "run_sql":
            run.status = "invalid_baseline_response"
        else:
            arguments = calls[0]["function"]["arguments"]
            result = registry.execute("run_sql", arguments)
            run.tool_calls = 1
            run.events = [{"evidence_id": "E1", "tool": "run_sql", "arguments": arguments, **result}]
            run.errors = int(not result["ok"])
            run.status = "answered" if result["ok"] else "tool_error"
            run.answer = json.dumps(result.get("result", {}))
    except Exception:
        run.status = "api_error"
    run.latency_seconds = time.perf_counter() - start
    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["offline", "live"], default="offline")
    parser.add_argument("--system", choices=["agent", "single-shot-sql"], default="agent")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", default="evaluation/results/offline.json")
    args = parser.parse_args()
    if args.mode == "live" and args.output == "evaluation/results/offline.json":
        parser.error("Live runs require --output to avoid overwriting offline results.")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")
    if args.mode == "live":
        from dotenv import load_dotenv
        load_dotenv()
        client = APIClient()
    tasks = json.loads((ROOT / "evaluation/tasks.json").read_text(encoding="utf-8"))
    if args.limit:
        tasks = tasks[:args.limit]
    results = []
    for task in tasks:
        frame = load_csv(ROOT / task["dataset"])
        registry = Analysis(frame, ROOT / "outputs" / "benchmark" / task["id"]).registry()
        row = {"id": task["id"], "category": task["category"], "expected": task["expected"]}
        if args.mode == "offline":
            if "oracle" not in task:
                row.update(status="not_applicable", passed=None)
            else:
                oracle = task["oracle"]
                result = registry.execute(oracle["tool"], oracle["arguments"])
                value = lookup(result["result"], oracle["path"]) if result["ok"] else None
                passed = (Path(value).is_file() if task["expected"] == "file_exists" and value else
                          close(value, task["expected"]) if isinstance(task["expected"], (float, int)) else False)
                row.update(status="tool_oracle_check", passed=passed, observed=value)
        else:
            if task.get("inject_error"):
                original = registry.execute
                injected = [False]
                def execute(name, arguments):
                    if name != "finish" and not injected[0]:
                        injected[0] = True
                        return {"ok": False, "error": "Injected transient tool failure. Retry the same operation."}
                    return original(name, arguments)
                registry.execute = execute
            if args.system == "agent":
                run = Agent(client, registry, input_price=price("LLM_INPUT_PRICE"),
                            output_price=price("LLM_OUTPUT_PRICE")).run(task["question"])
            else:
                run = single_shot(client, frame, registry, task["question"])
                ip, op = price("LLM_INPUT_PRICE"), price("LLM_OUTPUT_PRICE")
                if run.usage_complete and ip is not None and op is not None:
                    run.estimated_cost_usd = (run.input_tokens * ip + run.output_tokens * op) / 1e6
            row.update(screen_passed=screen(task, run), manual_task_success=None,
                       failure_category=None if screen(task, run) else run.status if run.status != "answered" else "screen_mismatch",
                       run=asdict(run))
        results.append(row)
        print(task["id"], row.get("passed", row.get("screen_passed")))
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "mode": args.mode,
              "system": args.system, "model": client.model if args.mode == "live" else None,
              "task_count": len(tasks), "results": results}
    if args.mode == "offline":
        checked = [r for r in results if r["passed"] is not None]
        report["summary"] = {"checked": len(checked), "passed": sum(r["passed"] for r in checked),
                             "live_agent_success_rate": None}
    else:
        report["summary"] = {"screen_pass_rate": sum(r["screen_passed"] for r in results) / len(results),
                             "manually_verified_task_success_rate": None,
                             "failure_categories": dict(Counter(r["failure_category"] for r in results if r["failure_category"])),
                             "mean_latency_seconds": sum(r["run"]["latency_seconds"] for r in results) / len(results),
                             "tool_errors": sum(r["run"]["errors"] for r in results),
                             "same_tool_recoveries": sum(r["run"]["recovered_tools"] for r in results)}
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 1 if args.mode == "offline" and any(r.get("passed") is False for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
