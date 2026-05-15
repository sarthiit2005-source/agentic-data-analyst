from agent.state import Run
from evaluation.benchmark import screen


def test_numeric_screen_ignores_citation_numbers():
    task = {"category": "numeric", "expected": 12}
    assert not screen(task, Run(status="answered", answer="See [E12]."))
    assert screen(task, Run(status="answered", answer="Total: 12 [E1]."))
    assert not screen(task, Run(status="step_limit", answer="12"))
