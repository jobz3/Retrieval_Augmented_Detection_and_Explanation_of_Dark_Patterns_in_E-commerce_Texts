"""
Thin wrapper around the Ollama Python client.
Handles JSON-mode requests and retries on malformed output.
"""

from __future__ import annotations

import json
import os
import time

import ollama


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = "qwen3:8b"

# Models that support extended thinking mode (Qwen3 family).
# We disable thinking for structured JSON output — it pollutes the response.
_THINKING_MODELS = {"qwen3"}


def _is_thinking_model(model: str) -> bool:
    return any(tag in model.lower() for tag in _THINKING_MODELS)


def chat_json(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system: str | None = None,
    temperature: float = 0.0,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    timeout: float = 120.0,
) -> dict:
    """
    Send a chat request to Ollama and return a parsed JSON dict.
    Retries up to `max_retries` times on JSON decode errors.

    For Qwen3 models, thinking mode is automatically disabled via the
    `think: false` option so the response is clean JSON without a
    <think>...</think> preamble.

    Args:
        prompt:       User message.
        model:        Ollama model tag (e.g. "qwen3:8b", "mistral-nemo:12b").
        system:       Optional system message.
        temperature:  Sampling temperature (0.0 = greedy).
        max_retries:  Number of retries on malformed JSON.

    Returns:
        Parsed dict from the model's response.

    Raises:
        ValueError if JSON cannot be parsed after all retries.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    client = ollama.Client(host=OLLAMA_BASE_URL, timeout=timeout)

    options: dict = {"temperature": temperature}
    if _is_thinking_model(model):
        options["think"] = False

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat(
                model=model,
                messages=messages,
                format="json",
                options=options,
            )
        except Exception as exc:
            # Covers ReadTimeout, ConnectError, and any other transport errors.
            # Convert to ValueError so pipeline ParseError handlers catch it.
            if attempt < max_retries:
                time.sleep(retry_delay)
                continue
            raise ValueError(f"Ollama request failed after {max_retries} attempts: {exc}") from exc

        raw = response["message"]["content"]
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                raise ValueError(
                    f"Ollama returned invalid JSON after {max_retries} attempts.\n"
                    f"Raw output:\n{raw}\nError: {e}"
                )
    # unreachable
    return {}


def list_models() -> list[str]:
    """Return available model tags from the local Ollama instance."""
    client = ollama.Client(host=OLLAMA_BASE_URL)
    result = client.list()
    # Ollama SDK ≥0.4 returns ListResponse with .models (list of Model objects)
    if hasattr(result, "models"):
        return [m.model for m in result.models]
    # Older SDK returned a plain dict
    return [m["name"] for m in result.get("models", [])]
