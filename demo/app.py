"""
Dark Pattern Detection Demo — FastAPI backend.

Endpoints:
  POST /analyze/url   { "url": "https://..." }  → SSE stream of findings
  POST /analyze/text  { "text": "Only 2 left!" } → JSON finding
  GET  /proxy/page?url=...   → proxied HTML for sandboxed iframe preview

Start with:
  cd <project-root>
  PYTHONPATH=. uvicorn demo.app:app --port 8765
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipelines.output_parser import parse_prediction, ParseError
from src.pipelines.prompts import SYSTEM_PROMPT
from src.pipelines.rag_few_shot import predict
from src.pipelines.span_grounding import check_grounding
from src.retrieval.retrieve import Retriever
from src.utils.ollama_client import chat_json, DEFAULT_MODEL

# ---------------------------------------------------------------------------
# App and shared retriever (loaded once at startup)
# ---------------------------------------------------------------------------

app = FastAPI(title="Dark Pattern Detector", version="1.0")

_retriever: Optional[Retriever] = None

_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
_FETCH_HEADERS = {
    "User-Agent": _BROWSER_UA,
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever(encoder="sbert", strategy="knn", k=5)
    return _retriever


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class UrlRequest(BaseModel):
    url: str

class TextRequest(BaseModel):
    text: str


# ---------------------------------------------------------------------------
# Text extraction — DOM pass then structured JSON fallback
# ---------------------------------------------------------------------------

# Tags where product/marketing copy lives; skip layout-only tags
_SCRAPE_TAGS = ["p", "span", "h1", "h2", "h3", "h4", "button", "label", "li", "a"]
_MIN_LEN = 30
_MAX_LEN = 500

# Boilerplate nav/legal text to discard
_SKIP_RE = re.compile(
    r"^(skip\s|home$|menu$|search$|cart$|wishlist$|log\s*(in|out)|sign\s*(in|up)|"
    r"my account|privacy|cookie|terms|contact|about us|©|all rights|copyright|"
    r"back to top|newsletter|follow us|share$|download|select\s+size|"
    r"size\s+guide|add to (cart|bag|wishlist)|sold out|out of stock|"
    r"\d{4}\s*[-–]\s*\d{4}|[a-z]{2,3}\s*\d{2,3}$)",  # size codes like "EU 38"
    re.IGNORECASE,
)

# Very short all-caps fragments that are just UI labels
_LABEL_ONLY_RE = re.compile(r"^[A-Z\s]{2,20}$")

# Keys in JSON blobs whose string values are worth analysing
_JSON_KEYS = {
    "name", "description", "title", "headline", "text",
    "catchphrase", "slogan", "tagline", "badge", "tag",
    "disambiguatingDescription", "alternateName",
    "availability", "itemCondition", "priceValidUntil",
}
_JSON_SKIP_RE = re.compile(
    r"^(https?://|schema\.org|@|[0-9]{4}-[0-9]{2}-[0-9]{2}|\s*$)",
    re.IGNORECASE,
)


def _collect_json_strings(obj, depth: int = 0) -> list[str]:
    if depth > 8:
        return []
    results = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and k.lower() in _JSON_KEYS:
                v = v.strip()
                if 30 <= len(v) <= 500 and not _JSON_SKIP_RE.match(v):
                    results.append(v)
            else:
                results.extend(_collect_json_strings(v, depth + 1))
    elif isinstance(obj, list):
        for item in obj:
            results.extend(_collect_json_strings(item, depth + 1))
    return results


def _extract_from_json_scripts(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    results: list[str] = []

    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
            results.extend(_collect_json_strings(data))
        except (json.JSONDecodeError, TypeError):
            pass

    for pattern in [
        r'window\.__[A-Z_]{2,50}__\s*=\s*(\{[\s\S]{20,20000}?\});?\s*(?:\n|</script>)',
        r'(?:__NEXT_DATA__|__NUXT__|__INITIAL_STATE__|__APP_STATE__)\s*=\s*(\{[\s\S]{20,50000}?\})\s*</script>',
    ]:
        for m in re.finditer(pattern, html, re.IGNORECASE):
            try:
                data = json.loads(m.group(1))
                results.extend(_collect_json_strings(data))
            except (json.JSONDecodeError, ValueError):
                pass

    return results


def _extract_snippets(html: str) -> list[tuple[str, str]]:
    """
    Two-pass extraction, deduped.  Returns at most 25 best candidates as
    (text, source) tuples where source is "DOM" or "JSON".

    Pass 1: visible DOM text from product-copy tags (server-rendered pages).
    Pass 2: strings from embedded JSON blobs (JS-rendered SPAs like aboutyou).
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "meta", "head", "nav", "footer"]):
        tag.decompose()

    seen: set[str] = set()
    snippets: list[tuple[str, str]] = []

    for el in soup.find_all(_SCRAPE_TAGS):
        text = re.sub(r"\s+", " ", el.get_text(separator=" ", strip=True)).strip()
        if (
            _MIN_LEN <= len(text) <= _MAX_LEN
            and text not in seen
            and not _SKIP_RE.match(text)
            and not _LABEL_ONLY_RE.match(text)
        ):
            seen.add(text)
            snippets.append((text, "DOM"))

    # JSON fallback — add anything not already found
    for text in _extract_from_json_scripts(html):
        text = re.sub(r"\s+", " ", text).strip()
        if (
            text not in seen
            and not _SKIP_RE.match(text)
            and not _LABEL_ONLY_RE.match(text)
        ):
            seen.add(text)
            snippets.append((text, "JSON"))

    _SIGNAL_RE = re.compile(
        r'(\d|€|\$|£|%|save|deal|offer|left|only|hurry|limited|free|discount|'
        r'last|sold|stock|buy|order|now|today|expires|valid|off\b|sale\b)',
        re.IGNORECASE,
    )

    def _score(item: tuple[str, str]) -> int:
        s = item[0]
        score = 0
        if _SIGNAL_RE.search(s):
            score += 10
        if len(s) >= 80:
            score += 12
        elif len(s) >= 50:
            score += 8
        elif len(s) >= 35:
            score += 4
        if s[0].isupper() and s[-1] in '.!':
            score += 3
        return score

    snippets.sort(key=_score, reverse=True)

    _NUM_RE = re.compile(r'\d[\d.,\-/]*')
    unique: list[tuple[str, str]] = []
    skeletons: set[str] = set()
    for item in snippets:
        skeleton = _NUM_RE.sub('#', item[0].lower()).strip()
        if skeleton not in skeletons:
            skeletons.add(skeleton)
            unique.append(item)
    return unique[:25]


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

_LABEL_COLORS = {
    "Scarcity":         "#e74c3c",
    "Urgency":          "#e67e22",
    "Social Proof":     "#9b59b6",
    "Misdirection":     "#2980b9",
    "Obstruction":      "#16a085",
    "Forced Action":    "#c0392b",
    "Sneaking":         "#7f8c8d",
    "Not Dark Pattern": "#27ae60",
    "Uncertain":        "#95a5a6",
}


def _analyse_snippet(text: str) -> dict:
    """Single-snippet analysis (used for /analyze/text)."""
    retriever = get_retriever()
    result, grounding, retrieved = predict(text, retriever=retriever)
    d = result.to_dict()
    d["input_text"]       = text
    d["is_dark_pattern"]  = result.is_dark_pattern()
    d["span_exact"]       = grounding["exact"]
    d["color"]            = _LABEL_COLORS.get(d["label"], "#95a5a6")
    d["retrieved_labels"] = [r["category"] for r in retrieved]
    d.pop("reasoning_steps", None)
    return d


# Batch system prompt — same as single but instructs array output
_BATCH_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    "Respond with ONLY a valid JSON object following this exact schema",
    "Respond with ONLY a valid JSON ARRAY where each element is a JSON object following this exact schema",
).replace(
    "(no markdown fences, no extra keys):",
    "(no markdown fences, no extra keys). The array must have exactly as many elements as there are input texts, in the same order:",
)


def _analyse_batch(texts: list[str]) -> list[dict]:
    """
    Send all snippets in a single LLM call and return one dict per snippet.
    Falls back to individual calls for any snippet whose batch result is malformed.
    ~10-20x faster than serial calls for 25 snippets.
    """
    retriever = get_retriever()

    # Build few-shot context using retrieved examples for the first snippet
    # (a representative anchor — gives the model format grounding)
    anchor_retrieved = retriever.retrieve(texts[0])
    examples_block = "\n".join(
        f'Example {i+1}: Input: """{r["text"]}"""\nLabel: {r["category"]}'
        for i, r in enumerate(anchor_retrieved[:3])
    )

    numbered = "\n".join(
        f'{i+1}. """{t}"""' for i, t in enumerate(texts)
    )

    prompt = (
        f"{examples_block}\n\n"
        "---\n\n"
        "Now analyse each of the following product texts. "
        f"Return a JSON array with exactly {len(texts)} objects, one per text, in order.\n\n"
        f"Texts to analyse:\n{numbered}\n\n"
        "Output JSON array:"
    )

    try:
        raw = chat_json(prompt, system=_BATCH_SYSTEM_PROMPT, timeout=120.0)
        # The model may return {"results": [...]} or directly [...]
        if isinstance(raw, list):
            items = raw
        elif isinstance(raw, dict):
            # Find the first list value
            items = next((v for v in raw.values() if isinstance(v, list)), None)
            if items is None:
                raise ValueError("no list in response")
        else:
            raise ValueError(f"unexpected type {type(raw)}")

        if len(items) != len(texts):
            raise ValueError(f"expected {len(texts)} items, got {len(items)}")

        results = []
        for text, item in zip(texts, items):
            try:
                result = parse_prediction(item, text)
                grounding = check_grounding(result, text)
                d = result.to_dict()
                d["input_text"]       = text
                d["is_dark_pattern"]  = result.is_dark_pattern()
                d["span_exact"]       = grounding["exact"]
                d["color"]            = _LABEL_COLORS.get(d["label"], "#95a5a6")
                d["retrieved_labels"] = [r["category"] for r in anchor_retrieved]
                d.pop("reasoning_steps", None)
                results.append(d)
            except (ParseError, Exception):
                # Fall back to individual call for this snippet
                results.append(_analyse_snippet(text))
        return results

    except Exception:
        # Whole batch failed — fall back to serial
        return [_analyse_snippet(t) for t in texts]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/proxy/page")
def proxy_page(url: str):
    if not url.startswith(("http://", "https://")):
        raise HTTPException(400, "Invalid URL")
    try:
        resp = requests.get(url, timeout=15, headers=_FETCH_HEADERS)
        resp.raise_for_status()
        html = resp.text
        base_tag = f'<base href="{url}">'
        m = re.search(r"<head[^>]*>", html, re.IGNORECASE)
        html = html[:m.end()] + base_tag + html[m.end():] if m else base_tag + html
        return HTMLResponse(content=html)
    except requests.RequestException as exc:
        raise HTTPException(502, f"Could not fetch URL: {exc}") from exc


@app.post("/analyze/text")
def analyze_text(req: TextRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "text must not be empty")
    try:
        finding = _analyse_snippet(text)
    except Exception as exc:
        raise HTTPException(500, f"Pipeline error: {exc}") from exc
    return JSONResponse({"findings": [finding]})


@app.post("/analyze/url")
def analyze_url(req: UrlRequest):
    """
    Streams findings as Server-Sent Events (SSE).
    Each event is one JSON finding, sent as soon as the LLM returns.

    Event types:
      data: {"type":"meta", "total": N, "url": "..."}
      data: {"type":"finding", ...finding fields...}
      data: {"type":"done"}
      data: {"type":"error", "message": "..."}
    """
    url = req.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(
            400,
            "Please enter a valid URL starting with http:// or https://  "
            "(it looks like you may have pasted text into the URL field)."
        )

    try:
        resp = requests.get(url, timeout=15, headers=_FETCH_HEADERS)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(502, f"Could not fetch URL: {exc}") from exc

    snippet_pairs = _extract_snippets(resp.text)
    if not snippet_pairs:
        raise HTTPException(
            422,
            "No analysable text found on the page. "
            "The page may require JavaScript to render its content."
        )

    texts = [t for t, _ in snippet_pairs]

    def _stream():
        yield f"data: {json.dumps({'type': 'meta', 'total': len(texts), 'url': url})}\n\n"
        try:
            findings = _analyse_batch(texts)
            for finding, (_, source) in zip(findings, snippet_pairs):
                finding["type"]   = "finding"
                finding["source"] = source
                yield f"data: {json.dumps(finding, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
