"""Small provider boundary using the OpenAI-compatible HTTP wire format."""
import os
import time
from typing import Protocol

import httpx


class Client(Protocol):
    def chat(self, messages: list[dict], tools: list[dict]) -> tuple[dict, dict]: ...


class APIClient:
    def __init__(self, api_key=None, model=None, base_url=None, transport=None):
        self.key = api_key or os.getenv("LLM_API_KEY")
        self.model = model or os.getenv("LLM_MODEL")
        self.url = (base_url or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        if not self.key or not self.model:
            raise ValueError("Set LLM_API_KEY and LLM_MODEL to run live analysis.")
        if not self.url.startswith("https://"):
            raise ValueError("The LLM API endpoint must use HTTPS.")
        self.transport = transport

    def chat(self, messages, tools):
        payload = dict(model=self.model, messages=messages, tools=tools,
                       tool_choice="auto", max_tokens=1800)
        # Retry only transient service errors; preserve the complete evidence.
        with httpx.Client(timeout=45, transport=self.transport, follow_redirects=False) as client:
            for attempt in range(3):
                try:
                    response = client.post(self.url + "/chat/completions", json=payload,
                                           headers={"Authorization": f"Bearer {self.key}"})
                    if response.status_code == 429 or response.status_code >= 500:
                        if attempt < 2:
                            time.sleep(0.5 * 2 ** attempt)
                            continue
                    if response.is_error:
                        raise RuntimeError(f"LLM API returned HTTP {response.status_code}. Check model, credentials, and quota.")
                    data = response.json()
                    message = data["choices"][0]["message"]
                    if message.get("role") != "assistant":
                        raise ValueError("Invalid assistant response")
                    return {k: message[k] for k in ("role", "content", "tool_calls") if k in message}, data.get("usage", {})
                except httpx.TransportError:
                    if attempt == 2:
                        raise RuntimeError("LLM API connection failed after three attempts.") from None
                    time.sleep(0.5 * 2 ** attempt)
        raise RuntimeError("LLM API request failed.")
