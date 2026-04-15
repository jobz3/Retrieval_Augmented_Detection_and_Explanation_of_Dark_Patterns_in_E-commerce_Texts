"""
Thin wrapper around the Ollama Python client.
Handles JSON-mode requests and retries on malformed output.

Project-specific behavior:
- true qwen3:* can use chat JSON mode with think disabled
- qwen3.5:* runs correctly on GPU on this HPC setup
- but the Python client path can return empty output
- for qwen3.5:* we therefore use generate(), and if that returns empty text,
  we fall back to the Ollama CLI against the same local server
"""

from __future__ import annotations

import json
import os
import subprocess
import time

import ollama


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_BIN = os.getenv("OLLAMA_BIN", "ollama")
DEFAULT_MODEL = "qwen3:8b"

_THINKING_MODELS = set()
_GENERATE_JSON_MODELS = {"qwen3.5:"}


def _is_thinking_model(model: str) -> bool:
    model_l = model.lower()
    return any(model_l.startswith(tag) for tag in _THINKING_MODELS)


def _use_generate_json_mode(model: str) -> bool:
    model_l = model.lower()
    return any(model_l.startswith(tag) for tag in _GENERATE_JSON_MODELS)


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


def _flatten_prompt(prompt: str, system: str | None) -> str:
    pieces: list[str] = []
    if system:
        pieces.append("SYSTEM INSTRUCTION:\n" + system.strip())
    pieces.append(
        "USER REQUEST:\n"
        + prompt.strip()
        + "\n\nReturn exactly one JSON object and nothing else."
    )
    return "\n\n".join(pieces).strip()


def _cli_generate_text(prompt: str, model: str, timeout: float) -> str:
    """
    Fall back to `ollama run` because the Python client path for qwen3.5:* can
    return empty content on this HPC setup even while GPU inference works.
    """
    env = os.environ.copy()

    if "OLLAMA_HOST" not in env and OLLAMA_BASE_URL.startswith("http://"):
        env["OLLAMA_HOST"] = OLLAMA_BASE_URL[len("http://") :]
    elif "OLLAMA_HOST" not in env and OLLAMA_BASE_URL.startswith("https://"):
        env["OLLAMA_HOST"] = OLLAMA_BASE_URL[len("https://") :]

    proc = subprocess.run(
        [OLLAMA_BIN, "run", model, prompt],
        capture_output=True,
        text=True,
        timeout=int(timeout) + 30,
        env=env,
    )
    if proc.returncode != 0:
        raise ValueError(
            f"Ollama CLI fallback failed with exit code {proc.returncode}.\n"
            f"STDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
        )
    return proc.stdout


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

    flattened_prompt = (
        prompt
        if not system
        else f"SYSTEM INSTRUCTION:\n{system}\n\nUSER REQUEST:\n{prompt}"
    )

    options = {"temperature": temperature}

    for attempt in range(1, max_retries + 1):
        try:
            # Qwen3 on this Ollama build puts its reasoning into message.thinking
            # and may leave message.content empty when using chat(format="json").
            # Force the CLI no-think path for qwen3 models.
            if model.lower().startswith("qwen3:"):
                raw = _cli_generate_text("/no_think\n" + flattened_prompt, model, timeout)
                return _extract_json_object(raw)

            client = Client(host=OLLAMA_BASE_URL, timeout=timeout)

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
