import json
import re
import time

from agent.prompts import SYSTEM
from agent.state import Run, Session
from tools.registry import object_schema


class Agent:
    def __init__(self, client, registry, max_steps=12, max_calls=32,
                 max_tokens=50_000, input_price=None, output_price=None):
        self.client, self.registry = client, registry
        self.max_steps, self.max_calls, self.max_tokens = max_steps, max_calls, max_tokens
        self.input_price, self.output_price = input_price, output_price

    def run(self, question, session=None, on_event=None):
        if not question.strip() or len(question) > 4000:
            raise ValueError("Question must contain 1–4,000 characters.")
        session = session if session is not None else Session()
        if session.turns >= 10:
            raise ValueError("Session turn limit reached. Start a new session.")
        run = Run()
        started = time.perf_counter()
        pending_errors = set()
        consecutive_errors = 0
        evidence = {e["evidence_id"] for e in session.history if e.get("ok") and e["tool"] != "finish"}

        def finish(report, evidence_ids, outcome):
            cited = set(re.findall(r"\[(E\d+)\]", report))
            if not set(evidence_ids).issubset(evidence) or not cited.issubset(evidence):
                raise ValueError("Report references unknown or unsuccessful evidence.")
            if outcome == "answered" and (not cited or not cited.issubset(set(evidence_ids))):
                example = next(iter(sorted(evidence)), "E1")
                raise ValueError(
                    f"Put bracketed citations inside report, e.g. 'The computed value is ... [{example}].', "
                    f"and include '{example}' in evidence_ids. Bare 'Evidence: {example}' is not valid. "
                    "Retry finish with corrected citations; do not repeat successful calculations."
                )
            return {"report": report, "outcome": outcome, "evidence_ids": evidence_ids}

        # One registry per session; replace the finish closure on follow-up turns.
        self.registry.tools.pop("finish", None)
        self.registry.add("finish", "Submit final report or request clarification.", object_schema({
            "report": {"type": "string", "minLength": 1, "maxLength": 12000,
                       "description": "Markdown report. Cite computed findings using bracketed IDs, e.g. 'Total: 1160 [E3].' Bare E3 is not a citation."},
            "evidence_ids": {"type": "array", "items": {"type": "string", "pattern": "^E[0-9]+$"},
                             "uniqueItems": True},
            "outcome": {"enum": ["answered", "clarification", "unanswerable"]}},
            ["report", "evidence_ids", "outcome"]), finish)
        if not session.messages:
            session.messages.append({"role": "system", "content": SYSTEM})
        session.messages.append({"role": "user", "content": question})
        session.turns += 1

        for step in range(self.max_steps):
            # Stop without trimming tool messages out of their protocol context.
            if len(json.dumps(session.messages)) > 160_000:
                run.status = "context_limit"
                break
            if run.input_tokens + run.output_tokens >= self.max_tokens:
                run.status = "token_limit"
                break
            run.steps = step + 1
            try:
                message, usage = self.client.chat(session.messages, self.registry.schemas())
                if not isinstance(message, dict):
                    raise ValueError("Invalid LLM response")
                calls = message.get("tool_calls") or []
                if not isinstance(calls, list) or any(
                    not isinstance(c, dict) or not isinstance(c.get("id"), str)
                    or not isinstance(c.get("function"), dict)
                    or not isinstance(c["function"].get("name"), str) for c in calls
                ) or len({c["id"] for c in calls}) != len(calls):
                    raise ValueError("Malformed tool calls from provider")
            except Exception as error:
                run.status = "api_error"
                detail = str(error)[:600] or type(error).__name__
                run.answer = f"The LLM request failed: {detail}"
                break
            run.usage_complete &= "prompt_tokens" in usage and "completion_tokens" in usage
            run.input_tokens += usage.get("prompt_tokens", 0)
            run.output_tokens += usage.get("completion_tokens", 0)
            session.messages.append(message)
            if not calls:
                session.messages.append({"role": "user", "content": "Use tools as needed, then call finish. Do not invent findings."})
                continue
            for call in calls:
                name = call["function"]["name"]
                arguments = call["function"].get("arguments", "{}")
                if run.status != "running" or run.tool_calls >= self.max_calls:
                    result = {"ok": False, "error": "Run stopped; this call was not executed."}
                    if run.status == "running":
                        run.status = "tool_limit"
                else:
                    run.tool_calls += 1
                    result = self.registry.execute(name, arguments)
                    if result["ok"]:
                        if name in pending_errors:
                            run.recovered_tools += 1
                            pending_errors.remove(name)
                        consecutive_errors = 0
                    else:
                        run.errors += 1
                        consecutive_errors += 1
                        pending_errors.add(name)
                event = {"evidence_id": f"E{len(session.history) + 1}", "tool": name,
                         "arguments": arguments, **result}
                session.history.append(event)
                run.events.append(event)
                if result["ok"] and name != "finish":
                    evidence.add(event["evidence_id"])
                # Bound observations explicitly instead of silently truncating JSON.
                observation = json.dumps(event)
                if len(observation) > 24_000:
                    observation = json.dumps({"evidence_id": event["evidence_id"], "ok": False,
                                              "error": "Result too large for model context. Request fewer rows/columns or aggregate."})
                session.messages.append({"role": "tool", "tool_call_id": call["id"], "content": observation})
                if on_event:
                    on_event(event)
                if name == "finish" and result["ok"]:
                    run.status = result["result"]["outcome"]
                    run.answer = result["result"]["report"]
                elif consecutive_errors >= 3 and run.status == "running":
                    run.status = "error_limit"
            if run.status != "running":
                break
        if run.status == "running":
            run.status = "step_limit"
        if not run.answer:
            run.answer = f"Analysis incomplete ({run.status}). Review the execution log for computed partial results."
        run.latency_seconds = round(time.perf_counter() - started, 4)
        run.evidence_log = [e for e in session.history if e.get("ok") and e["tool"] != "finish"]
        if run.usage_complete and self.input_price is not None and self.output_price is not None:
            run.estimated_cost_usd = (run.input_tokens * self.input_price +
                                      run.output_tokens * self.output_price) / 1_000_000
        return run
