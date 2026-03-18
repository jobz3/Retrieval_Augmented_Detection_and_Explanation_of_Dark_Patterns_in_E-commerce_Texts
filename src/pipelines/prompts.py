"""
Shared prompt layer for structured dark-pattern explanation pipelines.
"""

from __future__ import annotations

from typing import Iterable, Mapping

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


def build_label_only_examples_block(examples: Iterable[Mapping[str, str]]) -> str:
    """
    Format random few-shot references from the locked training split.

    The locked split does not contain gold explanations, so we include only
    text plus the gold label as an honest, auditable reference format.
    """
    rendered_examples: list[str] = [
        "Reference examples from the locked training split (label-only, no retrieval):"
    ]

    for index, example in enumerate(examples, start=1):
        rendered_examples.append(
            "\n".join(
                [
                    f"Example {index}",
                    "Input text:",
                    '"""',
                    example["text"],
                    '"""',
                    f"Gold label: {example['category']}",
                ]
            )
        )

    return "\n\n".join(rendered_examples)


def build_retrieved_examples_block(examples: Iterable[Mapping[str, str | float | int]]) -> str:
    """
    Format retrieved training references for the RAG prompt.

    Retrieved items are label-anchored training references only. They are not
    gold explanation exemplars because the locked training split does not
    contain explanation fields.
    """
    rendered_examples: list[str] = [
        "Retrieved reference examples from the locked training split (label-only references):"
    ]

    for index, example in enumerate(examples, start=1):
        score = example.get("score")
        score_line = f"Retrieval score: {float(score):.4f}" if score is not None else "Retrieval score: n/a"
        rendered_examples.append(
            "\n".join(
                [
                    f"Retrieved Example {index}",
                    score_line,
                    "Input text:",
                    '"""',
                    str(example["text"]),
                    '"""',
                    f"Gold label: {example['category']}",
                ]
            )
        )

    return "\n\n".join(rendered_examples)
