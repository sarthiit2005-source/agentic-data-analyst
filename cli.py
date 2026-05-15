"""Run with: python cli.py --file data/sales.csv --question 'Total revenue?'"""
import argparse
import os
from pathlib import Path
from uuid import uuid4

from agent.llm_client import APIClient
from agent.loop import Agent
from agent.reporting import save
from tools.analysis import Analysis
from tools.dataset import load_tables, load_schema, MAX_BYTES


def price(name):
    value = os.getenv(name)
    return float(value) if value else None


def main():
    parser = argparse.ArgumentParser(description="Agentic Data Analyst")
    parser.add_argument("--file", required=True, nargs="+", help="One or more CSV paths.")
    parser.add_argument("--schema", help="Optional JSON schema path.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-steps", type=int, choices=range(1, 21), default=12)
    args = parser.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    output = Path(args.output or f"outputs/{uuid4().hex}")
    try:
        schema = load_schema(Path(args.schema).read_bytes() if args.schema else None)
        files = []
        for filename in args.file:
            with Path(filename).open("rb") as handle:
                files.append((Path(filename).name, handle.read(MAX_BYTES + 1)))
        tables = load_tables(files, schema)
        client = APIClient()
        agent = Agent(client, Analysis(tables, output, schema).registry(), max_steps=args.max_steps,
                      input_price=price("LLM_INPUT_PRICE"), output_price=price("LLM_OUTPUT_PRICE"))
        run = agent.run(args.question, on_event=lambda e: print(f"{e['evidence_id']} {e['tool']}: {'ok' if e['ok'] else e.get('error')}"))
        save(run, output)
        print(f"\n{run.answer}\n\nReport: {output.resolve() / 'report.md'}")
        return 0 if run.status in ("answered", "clarification", "unanswerable") else 2
    except (ValueError, OSError) as error:
        print(f"Cannot start analysis: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
