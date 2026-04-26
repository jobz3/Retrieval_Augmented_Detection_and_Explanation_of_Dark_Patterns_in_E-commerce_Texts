"""
Thin wrapper around the Ollama Python client.
Handles JSON-mode requests and retries on malformed output.

Project-specific behavior:
- qwen3:* uses client.chat() with think=False via the HTTP API (avoids CLI timeout)
- qwen3.5:* runs correctly on GPU on this HPC setup
- but the Python client path can return empty output
- for qwen3.5:* we therefore use generate(), and if that returns empty text,
  we fall back to the Ollama CLI against the same local server
"""

from __future__ import annotations

import json
import os
import time

import ollama


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_BIN = os.getenv("OLLAMA_BIN", "ollama")
DEFAULT_MODEL = "qwen3:8b"


def _extract_json_object(raw: str) -> dict:
    """
    Extract the last valid top-level JSON object from a noisy model reply.

    This is robust to:
    - long reasoning text before the answer
    - markdown fences
    - example JSON blocks earlier in the reply
    - a final valid JSON object at the end
    """
    text = raw.strip()
    if not text:
        raise ValueError(
            "Could not find a JSON object in the model output.\n"
            f"Raw output:\n{raw}"
        )

    # Remove simple fenced wrappers if the whole response is fenced.
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()

    candidates: list[str] = []
    start = None
    depth = 0
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
            continue

        if ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start : i + 1])
                    start = None
            continue

    # Prefer the last valid JSON object, since the model often "thinks" first
    # and emits the final answer later.
    for candidate in reversed(candidates):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    raise ValueError(
        "Could not find a valid JSON object in the model output.\n"
        f"Raw output:\n{raw}"
    )



def chat_json(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system: str | None = None,
    temperature: float = 0.0,
    timeout: float = 300.0,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> dict:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    options = {"temperature": temperature}

    for attempt in range(1, max_retries + 1):
        try:
            client = ollama.Client(host=OLLAMA_BASE_URL, timeout=timeout)

            # qwen3:* supports think=False via the HTTP API — use that instead of
            # the CLI subprocess path, which times out on long prompts.
            if model.lower().startswith("qwen3:"):
                options_with_think = {**options, "think": False}
                response = client.chat(
                    model=model,
                    messages=messages,
                    format="json",
                    options=options_with_think,
                )
            else:
                response = client.chat(
                    model=model,
                    messages=messages,
                    format="json",
                    options=options,
                )

            if isinstance(response, dict):
                resp_dict = response
            else:
                try:
                    resp_dict = response.model_dump()
                except Exception:
                    resp_dict = dict(response)

            message = resp_dict.get("message", {}) or {}
            raw = message.get("content", "")

            if not raw or not raw.strip():
                thinking = message.get("thinking", "") or ""
                raise ValueError(
                    "Ollama returned empty message content.\n"
                    f"Thinking snippet:\n{thinking[:1000]}\n\n"
                    f"Full response object:\n{resp_dict}"
                )

            return json.loads(raw)

        except Exception as exc:
            msg = str(exc).lower()
            if "timed out" in msg or "timeout" in msg:
                raise ValueError(f"Ollama request timed out after {timeout}s: {exc}") from exc
            if attempt < max_retries:
                time.sleep(retry_delay)
                continue
            raise ValueError(f"Ollama request failed after {max_retries} attempts: {exc}") from exc

    return {}

def list_models() -> list[str]:
    """Return available model tags from the local Ollama instance."""
    client = ollama.Client(host=OLLAMA_BASE_URL)
    result = client.list()
    if hasattr(result, "models"):
        return [m.model for m in result.models]
    return [m["name"] for m in result.get("models", [])]
