"""
Dark Pattern Detector — Streamlit frontend.

Run with:
  cd <project-root>
  PYTHONPATH=. streamlit run demo/streamlit_app.py --server.port 8501
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

import requests
import streamlit as st
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipelines.rag_few_shot import predict
from src.pipelines.span_grounding import check_grounding
from src.retrieval.retrieve import Retriever

# ---------------------------------------------------------------------------
# Shared retriever (loaded once at startup)
# ---------------------------------------------------------------------------

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
# Text extraction — DOM pass then structured JSON fallback
# ---------------------------------------------------------------------------

_SCRAPE_TAGS = ["p", "span", "h1", "h2", "h3", "h4", "button", "label", "li", "a"]
_MIN_LEN = 30
_MAX_LEN = 500

_SKIP_RE = re.compile(
    r"^(skip\s|home$|menu$|search$|cart$|wishlist$|log\s*(in|out)|sign\s*(in|up)|"
    r"my account|privacy|cookie|terms|contact|about us|©|all rights|copyright|"
    r"back to top|newsletter|follow us|share$|download|select\s+size|"
    r"size\s+guide|add to (cart|bag|wishlist)|sold out|out of stock|"
    r"\d{4}\s*[-–]\s*\d{4}|[a-z]{2,3}\s*\d{2,3}$)",
    re.IGNORECASE,
)

_LABEL_ONLY_RE = re.compile(r"^[A-Z\s]{2,20}$")

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
    Two-pass extraction, deduped. Returns at most 25 best candidates as
    (text, source) tuples where source is "DOM" or "JSON".

    Pass 1: visible DOM text from product-copy tags (server-rendered pages).
    Pass 2: strings from embedded JSON blobs (JS-rendered SPAs).
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


# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dark Pattern Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stHeader"]           { background: transparent; }
[data-testid="stMainBlockContainer"] { padding-top: 1rem; }

/* collapse default streamlit top padding on columns */
[data-testid="column"] > div:first-child { padding-top: 0 !important; }

.app-header { border-bottom:1px solid #2e334d; padding-bottom:14px; margin-bottom:18px; }
.app-header h1 { color:#c5cae9; font-size:1.5rem; margin:0; }
.app-header p  { color:#8b90ab; font-size:0.82rem; margin:3px 0 8px; }
.badge {
    display:inline-block; background:#22263a; border:1px solid #2e334d;
    border-radius:20px; padding:2px 9px; font-size:0.68rem; font-weight:600;
    color:#8b90ab; margin-right:4px;
}
</style>
""", unsafe_allow_html=True)

# ── header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <h1>Retrieval-Augmented Dark Pattern Detector</h1>
  <p>RAG-augmented detection and explanation of manipulative e-commerce texts</p>
</div>
""", unsafe_allow_html=True)

# ── helpers ───────────────────────────────────────────────────────────────────

PANEL_HEIGHT = 860   # px — shared by preview iframe and findings scroll panel

_PANEL_CSS = """
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0f1117;
    color: #e8eaf6;
}
#scroll-panel {
    height: PANEL_HEIGHTpx;
    overflow-y: auto;
    padding: 4px 6px 12px;
    scrollbar-width: thin;
    scrollbar-color: #2e334d #0f1117;
}
#scroll-panel::-webkit-scrollbar      { width: 6px; }
#scroll-panel::-webkit-scrollbar-track{ background: #0f1117; }
#scroll-panel::-webkit-scrollbar-thumb{ background: #2e334d; border-radius: 3px; }

.summary-bar {
    display: flex; align-items: center; gap: 8px;
    padding: 8px 2px 10px;
    font-size: 0.82rem; font-weight: 700;
    position: sticky; top: 0; background: #0f1117; z-index: 10;
    border-bottom: 1px solid #2e334d; margin-bottom: 10px;
}
.pill-dark  { background:rgba(231,76,60,.18); color:#e74c3c; border:1px solid rgba(231,76,60,.3);  border-radius:20px; padding:3px 12px; }
.pill-clean { background:rgba(39,174,96,.15); color:#27ae60; border:1px solid rgba(39,174,96,.3);  border-radius:20px; padding:3px 12px; }
.pill-pend  { background:rgba(139,144,171,.1);color:#8b90ab; border:1px solid rgba(139,144,171,.2);border-radius:20px; padding:3px 12px; }

.dp-card {
    background: #1a1d27; border: 1px solid #2e334d;
    border-radius: 10px; padding: 14px 16px; margin-bottom: 10px;
}
.dp-card.dark-card  { border-left: 3px solid var(--card-color, #e74c3c); }
.dp-card.clean-card { border-left: 3px solid #27ae60; }
.dp-card.pend-card  { opacity: 0.65; }

.skel {
    background: linear-gradient(90deg,#22263a 25%,#2e334d 50%,#22263a 75%);
    background-size: 200% 100%;
    animation: shimmer 1.4s infinite;
    border-radius: 4px; height: 11px; margin: 5px 0;
}
.skel.w80 { width:80%; } .skel.w55 { width:55%; } .skel.w35 { width:35%; }
@keyframes shimmer { 0%{background-position:200% 0} 100%{background-position:-200% 0} }

.lbl {
    display:inline-block; padding:3px 11px; border-radius:20px;
    font-size:0.75rem; font-weight:700; color:#fff; margin-bottom:8px;
}

.snip { font-size:0.88rem; color:#e8eaf6; line-height:1.55; margin-bottom:10px; }
.hi {
    background:rgba(255,200,40,0.22); border-bottom:2px solid rgba(255,200,40,0.7);
    border-radius:3px; padding:0 2px; font-weight:600; color:#ffe082;
}

.xpl-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 10px;
}
.xpl-box {
    background: #22263a; border: 1px solid #2e334d; border-radius: 7px;
    padding: 8px 10px;
}
.xpl-box.full { grid-column: 1 / -1; }
.xpl-label {
    font-size: 0.65rem; font-weight: 800; text-transform: uppercase;
    letter-spacing: 0.07em; color: #8b90ab; margin-bottom: 4px;
}
.xpl-value { font-size: 0.82rem; color: #e8eaf6; line-height: 1.4; }

.conf-row { display:flex; align-items:center; gap:8px; }
.conf-bar-bg {
    flex:1; background:#0f1117; border-radius:3px; height:6px; overflow:hidden;
}
.conf-bar-fill { height:100%; border-radius:3px; }
.conf-pct { font-size:0.8rem; font-weight:700; min-width:32px; text-align:right; }

.rewrite {
    background:rgba(39,174,96,0.08); border:1px solid rgba(39,174,96,0.28);
    border-radius:7px; padding:9px 11px; font-size:0.84rem;
    color:#a5d6a7; line-height:1.5;
}
.rewrite-lbl {
    font-size:0.65rem; font-weight:800; text-transform:uppercase;
    letter-spacing:0.07em; color:#52b788; margin-bottom:5px;
}

.grounded {
    display:inline-block; background:rgba(39,174,96,0.15);
    border:1px solid rgba(39,174,96,0.3); border-radius:20px;
    padding:2px 9px; font-size:0.7rem; color:#52b788; margin-top:6px;
}

.src-dom  { display:inline-block; background:rgba(92,107,192,.15); border:1px solid rgba(92,107,192,.3); border-radius:4px; padding:1px 6px; font-size:0.65rem; color:#9fa8da; margin-left:6px; vertical-align:middle; }
.src-json { display:inline-block; background:rgba(249,168,38,.12);  border:1px solid rgba(249,168,38,.3);  border-radius:4px; padding:1px 6px; font-size:0.65rem; color:#ffd54f; margin-left:6px; vertical-align:middle; }

.span-label {
    font-size:0.65rem; font-weight:800; text-transform:uppercase;
    letter-spacing:0.07em; color:#b0963a; margin-bottom:3px;
}
.span-box {
    background:rgba(255,200,40,0.08); border:1px solid rgba(255,200,40,0.25);
    border-radius:6px; padding:7px 10px; margin-bottom:10px;
    font-size:0.86rem; line-height:1.5;
}
</style>
""".replace("PANEL_HEIGHT", str(PANEL_HEIGHT))


def _esc(s: str) -> str:
    return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")


def _highlight(text: str, span: str) -> str:
    if not span:
        return _esc(text)
    idx = text.find(span)
    if idx < 0:
        return _esc(text)
    return _esc(text[:idx]) + f'<span class="hi">{_esc(span)}</span>' + _esc(text[idx+len(span):])


def _skeleton_card(snippet: str, source: str = "DOM") -> str:
    src_cls  = "src-dom" if source == "DOM" else "src-json"
    src_tip  = "Extracted from visible HTML" if source == "DOM" else "Extracted from embedded JSON (page script)"
    return f"""
    <div class="dp-card pend-card">
      <div style="font-size:.72rem;color:#8b90ab;margin-bottom:6px">
        ⏳ Analysing…
        <span class="{src_cls}" title="{src_tip}">{source}</span>
      </div>
      <div class="snip">{_esc(snippet)}</div>
      <div class="skel w80"></div>
      <div class="skel w55"></div>
      <div class="skel w35"></div>
    </div>"""


def _result_card(f: dict) -> str:
    color    = f.get("color", "#95a5a6")
    label    = f.get("label", "Uncertain")
    conf     = f.get("confidence", 0.0)
    conf_pct = int(conf * 100)
    text     = f.get("input_text", "")
    span     = f.get("evidence_span", "")
    rationale= f.get("rationale", "")
    rewrite  = f.get("rewrite", "")
    mechanism= f.get("psychological_mechanism", "")
    harm     = f.get("harm_dimension", "")
    is_dark  = f.get("is_dark_pattern", False)
    grounded = f.get("span_exact", False)
    source   = f.get("source", "DOM")

    bar_color  = color if is_dark else "#27ae60"
    card_class = "dp-card dark-card" if is_dark else "dp-card clean-card"
    src_cls    = "src-dom" if source == "DOM" else "src-json"
    src_tip    = "Extracted from visible HTML" if source == "DOM" else "Extracted from embedded page script (JSON-LD / SPA data)"

    highlighted = _highlight(text, span)

    conf_row = f"""
    <div class="conf-row">
      <div class="conf-bar-bg">
        <div class="conf-bar-fill" style="width:{conf_pct}%;background:{bar_color}"></div>
      </div>
      <div class="conf-pct" style="color:{bar_color}">{conf_pct}%</div>
    </div>"""

    if span and is_dark:
        span_block = f"""
        <div class="span-label">🔍 Key Phrase (Evidence Span)</div>
        <div class="span-box">
          <span class="hi">{_esc(span)}</span>
          <span style="font-size:0.72rem;color:#8b90ab;margin-left:8px">— the specific wording that triggers the dark pattern</span>
        </div>"""
    else:
        span_block = ""

    if is_dark:
        xpl = f"""
        {span_block}
        <div class="xpl-grid">
          <div class="xpl-box">
            <div class="xpl-label">🧠 Psychological Mechanism</div>
            <div class="xpl-value">{_esc(mechanism)}</div>
          </div>
          <div class="xpl-box">
            <div class="xpl-label">⚠️ Consumer Harm</div>
            <div class="xpl-value">{_esc(harm)}</div>
          </div>
          <div class="xpl-box full">
            <div class="xpl-label">💬 Why This Is a Dark Pattern</div>
            <div class="xpl-value">{_esc(rationale)}</div>
          </div>
        </div>
        <div class="rewrite-lbl">✏️ Suggested Neutral Rewrite</div>
        <div class="rewrite">{_esc(rewrite)}</div>
        {('<div class="grounded">✓ Key phrase verified in source text</div>' if grounded else '')}"""
    else:
        xpl = f"""
        <div class="xpl-box" style="margin-bottom:8px">
          <div class="xpl-label">💬 Why This Is Not a Dark Pattern</div>
          <div class="xpl-value" style="color:#8b90ab">{_esc(rationale)}</div>
        </div>"""

    return f"""
    <div class="{card_class}" style="--card-color:{color}">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
        <div>
          <span class="lbl" style="background:{color}">{_esc(label)}</span>
          <span class="{src_cls}" title="{src_tip}">{source}</span>
        </div>
        <span style="font-size:0.72rem;color:#8b90ab">Confidence</span>
      </div>
      {conf_row}
      <div class="snip" style="margin-top:10px">{highlighted}</div>
      {xpl}
    </div>"""


def _full_panel_html(cards: list[str], dark: int, clean: int, pending: int) -> str:
    summary = f"""
    <div class="summary-bar">
      <span class="pill-dark">{dark} dark</span>
      <span class="pill-clean">{clean} clean</span>
      {'<span class="pill-pend">'+str(pending)+' pending</span>' if pending else ''}
    </div>"""
    return _PANEL_CSS + f'<div id="scroll-panel">{summary}{"".join(cards)}</div>'


# ── warm retriever once ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading FAISS index…")
def load_retriever():
    return get_retriever()

load_retriever()

# ── input bar ─────────────────────────────────────────────────────────────────
mode = st.radio(
    "mode", ["🌐  Analyse URL", "📝  Analyse Text"],
    horizontal=True, label_visibility="collapsed",
)

col_input, col_btn = st.columns([6, 1])
with col_input:
    if "URL" in mode:
        user_input = st.text_input(
            "URL", placeholder="https://www.example-shop.com/product/123",
            label_visibility="collapsed",
        )
    else:
        user_input = st.text_area(
            "Text", placeholder="Enter the text to analyse for dark patterns…",
            height=72, label_visibility="collapsed",
        )
with col_btn:
    st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)
    run = st.button("Analyse", type="primary", use_container_width=True)

st.divider()

# ── run ───────────────────────────────────────────────────────────────────────
if run and user_input.strip():
    value = user_input.strip()

    # ── URL mode ──────────────────────────────────────────────────────────────
    if "URL" in mode:
        if not value.startswith(("http://", "https://")):
            st.error("Please enter a valid URL starting with http:// or https://")
            st.stop()

        col_prev, col_find = st.columns([3, 2])

        with col_prev:
            st.markdown("**Page Preview**")
            page_html = None
            with st.spinner("Fetching page…"):
                try:
                    page_resp = requests.get(value, timeout=15, headers=_FETCH_HEADERS)
                    page_resp.raise_for_status()
                    page_html = page_resp.text
                    base_tag  = f'<base href="{value}" target="_blank">'
                    m = re.search(r"<head[^>]*>", page_html, re.IGNORECASE)
                    page_html = (
                        page_html[:m.end()] + base_tag + page_html[m.end():]
                        if m else base_tag + page_html
                    )
                    st.components.v1.html(page_html, height=PANEL_HEIGHT, scrolling=True)
                except Exception as e:
                    st.warning(f"Could not load preview: {e}")

        with col_find:
            st.markdown("**Analysis**")

            with st.spinner("Extracting snippets…"):
                raw_html = page_html or ""
                if not raw_html:
                    try:
                        raw_html = requests.get(value, timeout=15, headers=_FETCH_HEADERS).text
                    except Exception as e:
                        st.error(f"Could not fetch page: {e}")
                        st.stop()
                snippets = _extract_snippets(raw_html)

            if not snippets:
                st.warning("No analysable text found. The page may require JavaScript to render.")
                st.stop()

            panel_ph = st.empty()
            cards    = [_skeleton_card(s, src) for s, src in snippets]
            with panel_ph:
                st.components.v1.html(
                    _full_panel_html(cards, 0, 0, len(snippets)),
                    height=PANEL_HEIGHT + 20,
                    scrolling=False,
                )

            dark_count = clean_count = 0

            for i, (snippet, source) in enumerate(snippets):
                try:
                    finding = _analyse_snippet(snippet)
                    finding["source"] = source
                    cards[i] = _result_card(finding)
                    if finding.get("is_dark_pattern"):
                        dark_count += 1
                    else:
                        clean_count += 1
                except Exception as e:
                    cards[i] = f'<div class="dp-card" style="color:#ef9a9a;border-left:3px solid #e74c3c">⚠ {_esc(str(e))}</div>'

                pending = len(snippets) - i - 1
                with panel_ph:
                    st.components.v1.html(
                        _full_panel_html(cards, dark_count, clean_count, pending),
                        height=PANEL_HEIGHT + 20,
                        scrolling=False,
                    )

    # ── Text mode ─────────────────────────────────────────────────────────────
    else:
        with st.spinner("Analysing…"):
            try:
                finding = _analyse_snippet(value)
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.stop()

        st.components.v1.html(
            _PANEL_CSS + _result_card(finding),
            height=500,
            scrolling=True,
        )

elif run:
    st.warning("Please enter a URL or text.")
