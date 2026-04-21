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
The text may be in any language (English, German, or other). Analyse it in the original language — do not translate.

Respond with ONLY a valid JSON object following this exact schema (no markdown fences, no extra keys):
{OUTPUT_SCHEMA_STR}

Dark pattern categories and their key signals:
  Scarcity       — false or exaggerated claims of limited stock or availability
                   (German signals: "Nur noch X Stück", "weniger als X verfügbar")
  Urgency        — artificial time pressure (countdown timers, "today only", "expires soon")
                   (German signals: "nur heute", "nur noch X Stunden", "zeitlich begrenzt")
  Social Proof   — manipulated social validation ("1,000 people viewing", fake or inflated reviews)
                   (German signals: "X Personen sehen sich dieses Produkt an", "X-mal verkauft")
  Misdirection   — attention diverted from important options (pre-ticked boxes, buried opt-outs, confirmshaming)
                   (German signals: "Nein, danke", "Nein Danke ich möchte nicht", dismissive opt-out buttons)
  Obstruction    — deliberately hard to cancel, unsubscribe, or opt out; excessive notice periods
                   (German signals: "Kündigung nur per Einschreiben", "Kündigung X Tage vor Verlängerung", long cancellation deadlines)
  Forced Action  — users must take unwanted steps (accept marketing emails, create an account)
                   (German signals: "müssen Sie ein Kundenkonto anlegen", "stimmen Sie unserem Newsletter zu")
  Sneaking       — hidden charges, auto-added items, undisclosed auto-renewal subscriptions
                   (German signals: "automatische Verlängerung", "verlängert sich automatisch")
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
