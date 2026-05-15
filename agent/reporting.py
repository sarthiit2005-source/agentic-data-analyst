import json
from dataclasses import asdict
from pathlib import Path


def markdown(run):
    parts = [f"# Analysis report\n\nStatus: **{run.status}**\n\n{run.answer}",
             "\n## Execution metrics\n\n```json\n" +
             json.dumps({k: v for k, v in asdict(run).items() if k not in ("answer", "events", "evidence_log")}, indent=2) + "\n```",
             "\n## Execution log"]
    for event in run.events:
        parts.append(f"\n### {event['evidence_id']}: {event['tool']}\n\n```json\n{json.dumps(event, indent=2)}\n```")
    current_ids = {event["evidence_id"] for event in run.events}
    prior = [event for event in run.evidence_log if event["evidence_id"] not in current_ids]
    if prior:
        parts.append("\n## Evidence from earlier questions")
        for event in prior:
            parts.append(f"\n### {event['evidence_id']}: {event['tool']}\n\n```json\n{json.dumps(event, indent=2)}\n```")
    return "\n".join(parts)


def save(run, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.md").write_text(markdown(run), encoding="utf-8")
    (directory / "trace.json").write_text(json.dumps(asdict(run), indent=2), encoding="utf-8")
