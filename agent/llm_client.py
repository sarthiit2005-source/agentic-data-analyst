"""Small provider boundary using the OpenAI-compatible HTTP wire format."""
import os
import time
from typing import Protocol

import httpx


class Client(Protocol):
    def chat(self, messages: list[dict], tools: list[dict]) -> tuple[dict, dict]: ...


class APIClient:
    def __init__(self, api_key=None, model=None, base_url=None, transport=None,
                 api_mode="auto"):
        self.key = api_key or os.getenv("LLM_API_KEY")
        self.model = model or os.getenv("LLM_MODEL")
        self.url = (base_url or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        if not self.key or not self.model:
            raise ValueError("Set LLM_API_KEY and LLM_MODEL to run live analysis.")
        if not self.url.startswith("https://"):
            raise ValueError("The LLM API endpoint must use HTTPS.")
        self.transport = transport
        if api_mode not in ("auto", "responses", "chat_completions"):
            raise ValueError("api_mode must be auto, responses, or chat_completions.")
        self.api_mode = api_mode

    def _mode(self):
        if self.api_mode != "auto":
            return self.api_mode
        model = self.model.lower()
        return "responses" if (model.startswith(("gpt-5", "o1", "o3", "o4"))
                               or model.endswith(("-sol", "-astra", "-luna"))) else "chat_completions"

    def _safe_error(self, response):
        try:
            error = response.json().get("error", {})
            message = str(error.get("message") or "Request rejected")
            code = error.get("code") or error.get("type")
        except (ValueError, AttributeError):
            message, code = "Request rejected", None
        message = message.replace(self.key, "[REDACTED]")[:500]
        suffix = f" ({code})" if code else ""
        return f"LLM API HTTP {response.status_code}{suffix}: {message}"

    @staticmethod
    def _responses_tools(tools):
        return [{"type": "function", **tool["function"]} for tool in tools]

    @staticmethod
    def _responses_input(messages):
        items = []
        for message in messages:
            role = message["role"]
            if role == "tool":
                items.append({"type": "function_call_output",
                              "call_id": message["tool_call_id"],
                              "output": message.get("content", "")})
            elif role == "assistant" and message.get("tool_calls"):
                if message.get("content"):
                    items.append({"role": "assistant", "content": message["content"]})
                for call in message["tool_calls"]:
                    items.append({"type": "function_call", "call_id": call["id"],
                                  "name": call["function"]["name"],
                                  "arguments": call["function"].get("arguments", "{}")})
            else:
                items.append({"role": role, "content": message.get("content", "")})
        return items

    @staticmethod
    def _parse_response(data):
        calls, text = [], []
        for item in data.get("output", []):
            if item.get("type") == "function_call":
                calls.append({"id": item["call_id"], "type": "function",
                              "function": {"name": item["name"],
                                           "arguments": item.get("arguments", "{}")}})
            elif item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") in ("output_text", "text"):
                        text.append(content.get("text", ""))
        message = {"role": "assistant", "content": "\n".join(text) or None}
        if calls:
            message["tool_calls"] = calls
        usage = data.get("usage", {})
        normalized = {"prompt_tokens": usage.get("input_tokens", 0),
                      "completion_tokens": usage.get("output_tokens", 0)}
        return message, normalized

    def chat(self, messages, tools):
        mode = self._mode()
        if mode == "responses":
            payload = dict(model=self.model, input=self._responses_input(messages),
                           tools=self._responses_tools(tools), tool_choice="auto",
                           max_output_tokens=1800)
            path = "/responses"
        else:
            payload = dict(model=self.model, messages=messages, tools=tools,
                           tool_choice="auto", max_tokens=1800)
            path = "/chat/completions"
        # Retry only transient service errors; preserve the complete evidence.
        with httpx.Client(timeout=45, transport=self.transport, follow_redirects=False) as client:
            for attempt in range(3):
                try:
                    response = client.post(self.url + path, json=payload,
                                           headers={"Authorization": f"Bearer {self.key}"})
                    if response.status_code == 429 or response.status_code >= 500:
                        if attempt < 2:
                            time.sleep(0.5 * 2 ** attempt)
                            continue
                    if response.is_error:
                        raise RuntimeError(self._safe_error(response))
                    data = response.json()
                    if mode == "responses":
                        return self._parse_response(data)
                    message = data["choices"][0]["message"]
                    if message.get("role") != "assistant":
                        raise ValueError("Invalid assistant response")
                    return {k: message[k] for k in ("role", "content", "tool_calls") if k in message}, data.get("usage", {})
                except httpx.TransportError:
                    if attempt == 2:
                        raise RuntimeError("LLM API connection failed after three attempts.") from None
                    time.sleep(0.5 * 2 ** attempt)
        raise RuntimeError("LLM API request failed.")
