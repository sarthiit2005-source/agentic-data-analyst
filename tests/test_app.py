from streamlit.testing.v1 import AppTest
from pathlib import Path


def test_app_starts_without_api_key():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=20)
    assert not app.exception
    assert app.title[0].value == "Agentic Data Analyst"
    assert app.selectbox[0].label == "API mode"
    app.selectbox[1].select("Sales").run()
    assert not app.exception
    assert len(app.dataframe[0].value) == 10
    assert next(button for button in app.button if button.label == "Analyze").disabled
