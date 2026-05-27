"""
Prompt templates for the dark pattern explanation pipeline.

Three prompting modes share the same system prompt and output schema:
  - zero_shot   : system prompt + single user message with the target text
  - random_shot : system prompt + k random labelled examples + target text
  - rag_shot    : system prompt + k retrieved labelled examples + target text

The schema is embedded once in the system prompt.
Few-shot examples show Input→Label pairs so the model learns the mapping,
while the full JSON schema in the system prompt drives the output format.
"""

from __future__ import annotations

from src.pipelines.schema import OUTPUT_SCHEMA_STR

# ---------------------------------------------------------------------------
# System prompt (shared by all three pipelines)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""You are an expert in consumer psychology and e-commerce manipulation.
Your task: analyse a product page text and identify any dark pattern — a deceptive copy or UI practice that manipulates consumers.
The text may be in any language (English, German, Italian, or other). Analyse it in the original language — do not translate.

Respond with ONLY a valid JSON object following this exact schema (no markdown fences, no extra keys):
{OUTPUT_SCHEMA_STR}

Dark pattern categories and their key signals:
  Scarcity       — false or exaggerated claims of limited stock or availability
  Urgency        — artificial time pressure (countdown timers, "today only", "expires soon")
  Social Proof   — manipulated social validation ("1,000 people viewing", fake or inflated reviews)
  Misdirection   — attention diverted from important options (pre-ticked boxes, buried opt-outs, confirmshaming)
  Obstruction    — deliberately hard to cancel, unsubscribe, or opt out; excessive notice periods
  Forced Action  — users must take unwanted steps (accept marketing emails, create an account)
  Sneaking       — hidden charges, auto-added items, undisclosed auto-renewal subscriptions
  Not Dark Pattern — transparent, honest product text with no manipulation; clearly states guest checkout is available, no lock-in, easy cancellation

Rules you must follow:
1. reasoning_steps MUST be populated FIRST — think step by step before committing to a label.
2. evidence_span MUST be an exact verbatim substring copied from the input text.
3. confidence is your certainty [0.0–1.0] for the chosen label.
4. For "Not Dark Pattern" set evidence_span to the most representative neutral phrase.
5. Use "Uncertain" only when you genuinely cannot determine the label.
6. rewrite must rewrite the product text to remove the dark pattern while preserving the core offer.
   - If the text is a dark pattern: produce a complete, natural sentence even if the input is short (e.g. "LAST 1 LEFT" → "This item is available — add it to your cart.").
   - If the label is "Not Dark Pattern": copy the input text unchanged."""


# ---------------------------------------------------------------------------
# Optional language-specific signal-phrase appendix
# ---------------------------------------------------------------------------
# Appended to SYSTEM_PROMPT when the caller knows the input is in a specific
# language. Lists per-category surface cues that frequently appear in that
# language but have no direct English equivalent in the EC-DarkPattern training
# corpus. Helps the LLM anchor Italian regulatory/retail vocabulary
# (PEC, raccomandata A/R, RAEE, contributo ambientale, …) to the correct label
# when retrieval alone cannot bridge the gap.

ITALIAN_SIGNAL_APPENDIX = """

Italian-specific surface cues (use these to disambiguate when the input is in Italian):
  Scarcity       — "solo X rimasti", "ultimi X pezzi", "ultime camere", "in esaurimento", "edizione limitata"
  Urgency        — "solo per oggi", "ultima chance", "subito", "scade tra", "termina tra", "affrettati", "ultime ore", "follia del giorno"
  Social Proof   — "prenotato X volte", "scelto da X utenti", "il più venduto", "hanno acquistato", "salvato in X liste dei desideri"
  Misdirection   — "prezzo di pubblicazione", "prezzo consigliato", "prezzo imbattibile", "risparmi rispetto a", strikethrough discount math
  Obstruction    — "raccomandata A/R", "PEC", "Posta Elettronica Certificata", "numero verde (orari limitati)", "area clienti", "modulo PDF", "X giorni di preavviso", "contattare il servizio clienti via telefono", "non è possibile cancellare online"
  Forced Action  — "devi accettare i cookie di profilazione", "devi verificare il numero", "obbligatorio inserire", "iscriviti alla newsletter per", "registrati per continuare la lettura"
  Sneaking       — "applicato in fase di checkout", "costo di servizio", "spese di gestione obbligatorie", "contributo ambientale RAEE", "costi doganali a carico del destinatario", "non include", "rinnovo automatico", auto-added paid subscription
  Not Dark Pattern — "consegna standard 3-5 giorni", "garanzia soddisfatti o rimborsati", "recensioni verificate", "politica di reso 30 giorni", neutral product/category labels, last-access timestamps"""


_LANG_APPENDICES: dict[str, str] = {
    "it": ITALIAN_SIGNAL_APPENDIX,
}


def system_prompt_for(lang: str | None = None) -> str:
    """Return the system prompt, optionally augmented with per-language signal cues.

    `lang` is a language code recognised in `_LANG_APPENDICES` (currently only
    "it"). Any other value — including None — returns the base language-agnostic
    prompt unchanged, so existing English and German callers are unaffected.
    """
    appendix = _LANG_APPENDICES.get((lang or "").lower(), "")
    return SYSTEM_PROMPT + appendix


# ---------------------------------------------------------------------------
# Prompt formatters
# ---------------------------------------------------------------------------

def format_zero_shot_prompt(text: str) -> str:
    """User message for zero-shot inference."""
    return (
        "Analyse the following product text and return a JSON object following the schema.\n\n"
        f'Product text: """{text}"""\n\n'
        "Output JSON:"
    )


def _format_example_block(examples: list[dict]) -> str:
    """
    Format a list of few-shot examples into a prompt block.

    Each example dict must have keys:
        "text"     : product page text
        "category" : gold label string (e.g. "Scarcity")

    Optionally may also have:
        "evidence_span" : key phrase from the text (used if present)
    """
    lines = ["Here are labelled examples to guide your analysis:\n"]
    for i, ex in enumerate(examples, start=1):
        lines.append(f"--- Example {i} ---")
        lines.append(f'Product text: """{ex["text"]}"""')
        lines.append(f'Label: {ex["category"]}')
        if ex.get("evidence_span"):
            lines.append(f'Key phrase: "{ex["evidence_span"]}"')
        lines.append("")
    return "\n".join(lines)


def format_few_shot_prompt(text: str, examples: list[dict]) -> str:
    """
    User message for few-shot inference (random or RAG).

    Args:
        text:     Target product text to classify.
        examples: List of dicts with "text" and "category" keys.
    """
    example_block = _format_example_block(examples)
    return (
        f"{example_block}"
        "---\n\n"
        "Now analyse the following product text and return a JSON object following the schema.\n\n"
        f'Product text: """{text}"""\n\n'
        "Output JSON:"
    )
