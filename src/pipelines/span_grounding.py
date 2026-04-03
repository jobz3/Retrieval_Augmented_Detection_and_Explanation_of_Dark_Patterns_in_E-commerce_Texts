"""
Span grounding verification.

Checks whether the evidence_span returned by the LLM is a verbatim substring
of the original input text.  Reports both exact and case-insensitive matches.

Two levels of grounding:
  - exact:            span appears as-is in input (required for faithfulness)
  - case_insensitive: span appears when both are lowercased (acceptable fallback)

Usage:
    from src.pipelines.span_grounding import check_grounding, grounding_rate
    info = check_grounding(result, input_text)
    rate = grounding_rate(results, texts)
"""

from __future__ import annotations

from src.pipelines.schema import PredictionResult


def check_grounding(result: PredictionResult, input_text: str) -> dict:
    """
    Verify that result.evidence_span is grounded in input_text.

    Returns a dict with:
        exact            (bool)  — verbatim substring match
        case_insensitive (bool)  — lowercase match
        span_length      (int)   — character length of the span
        span             (str)   — the evidence span value
    """
    span = result.evidence_span.strip()
    exact = span in input_text
    case_insensitive = span.lower() in input_text.lower()

    return {
        "exact":            exact,
        "case_insensitive": case_insensitive,
        "span_length":      len(span),
        "span":             span,
    }


def grounding_rate(
    results: list[PredictionResult],
    texts: list[str],
    mode: str = "exact",
) -> float:
    """
    Fraction of predictions whose evidence_span is grounded.

    Args:
        results: List of PredictionResult objects.
        texts:   Corresponding input texts (same order).
        mode:    "exact" or "case_insensitive".

    Returns:
        Float in [0, 1].
    """
    if not results:
        return 0.0
    hits = sum(
        check_grounding(r, t)[mode]
        for r, t in zip(results, texts)
    )
    return hits / len(results)


def annotate_grounding(
    records: list[dict],
    mode: str = "exact",
) -> list[dict]:
    """
    Add grounding fields to a list of result dicts (as returned by pipeline predict functions).

    Expects each dict to have "evidence_span" and "input_text" keys.
    Adds: "span_exact", "span_ci" (case-insensitive), "span_length".
    """
    for rec in records:
        span = rec.get("evidence_span", "").strip()
        text = rec.get("input_text", "")
        rec["span_exact"]  = span in text
        rec["span_ci"]     = span.lower() in text.lower()
        rec["span_length"] = len(span)
    return records
