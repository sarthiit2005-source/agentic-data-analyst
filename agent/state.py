from dataclasses import dataclass, field
from typing import Any


@dataclass
class Session:
    """Conversation state, separate from the immutable analysis dataset."""

    messages: list[dict] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    turns: int = 0


@dataclass
class Run:
    status: str = "running"
    answer: str = ""
    steps: int = 0
    tool_calls: int = 0
    errors: int = 0
    recovered_tools: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usage_complete: bool = True
    latency_seconds: float = 0
    estimated_cost_usd: float | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    evidence_log: list[dict[str, Any]] = field(default_factory=list)
