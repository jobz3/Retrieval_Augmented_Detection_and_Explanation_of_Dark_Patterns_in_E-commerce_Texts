"""
Shared prompt layer for structured dark-pattern explanation pipelines.
"""

from __future__ import annotations

from src.pipelines.schema import OUTPUT_SCHEMA_STR, PatternType


KNOWN_LABELS = tuple(pattern.value for pattern in PatternType)


SHARED_SYSTEM_PROMPT = f"""You are a careful annotation assistant for e-commerce dark patterns.

Return exactly one JSON object and nothing else.
Do not use markdown, bullet points, or commentary outside the JSON object.

You must follow these rules:
1. The label must be exactly one of: {", ".join(KNOWN_LABELS)}.
2. Include every required schema field.
3. evidence_span must be copied verbatim from the input text.
4. If you cannot support a confident dark-pattern label from the text alone, use "Uncertain".
5. Keep the rationale concise, grounded, and specific to the text.
6. The rewrite should remove manipulative wording while preserving the offer intent where possible.

Required JSON schema:
{OUTPUT_SCHEMA_STR}
"""


def build_user_prompt(input_text: str, examples_block: str | None = None) -> str:
    """Build the shared user prompt for a single product text."""
    blocks = [
        "Analyze the product text below using only the text itself as evidence.",
        (
            "Choose one allowed label, provide a grounded evidence_span, keep the rationale to 1-2 "
            "sentences, and rewrite the full text without the dark pattern."
        ),
    ]

    if examples_block:
        blocks.append(f"Reference examples:\n{examples_block}")

    blocks.append(f'Input text:\n"""\n{input_text}\n"""')
    return "\n\n".join(blocks)
