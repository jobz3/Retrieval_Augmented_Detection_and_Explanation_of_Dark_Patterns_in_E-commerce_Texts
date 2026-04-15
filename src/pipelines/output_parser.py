"""
Output parser for the dark pattern explanation pipeline.

Converts a raw dict (from Ollama JSON mode) into a validated PredictionResult.
Handles:
  - Fuzzy label matching (case-insensitive, partial)
  - Confidence clamping to [0, 1]
  - Missing field fallbacks
  - Raises ParseError for unrecoverable failures

Usage:
    from src.pipelines.output_parser import parse_prediction, ParseError
    result = parse_prediction(raw_dict, input_text)
"""

from __future__ import annotations

from src.pipelines.schema import PatternType, PredictionResult


class ParseError(Exception):
    """Raised when a raw LLM dict cannot be converted to a PredictionResult."""


# Alias map for common alternative label phrasings LLMs produce
_LABEL_ALIASES: dict[str, PatternType] = {
    "not a dark pattern": PatternType.NONE,
    "no dark pattern":    PatternType.NONE,
    "none":               PatternType.NONE,
    "n/a":                PatternType.NONE,
    "forced":             PatternType.FORCED_ACTION,
    "sneak":              PatternType.SNEAKING,
    "obstruct":           PatternType.OBSTRUCTION,
    "misdirect":          PatternType.MISDIRECTION,
    "social":             PatternType.SOCIAL_PROOF,
    "proof":              PatternType.SOCIAL_PROOF,
    "scarc":              PatternType.SCARCITY,
    "urgen":              PatternType.URGENCY,
}


def _resolve_label(value: str) -> PatternType:
    """
    Map a raw string to a PatternType.

    Priority:
    1. Exact match (case-insensitive) against PatternType values
    2. Substring match: raw value contains the enum value or vice versa
    3. Alias map for common alternative phrasings
    4. Falls back to PatternType.UNCERTAIN
    """
    normalized = value.strip().lower()

    # 1. Exact match
    for pt in PatternType:
        if pt.value.lower() == normalized:
            return pt

    # 2. Substring / partial match
    for pt in PatternType:
        pt_lower = pt.value.lower()
        if pt_lower in normalized or normalized in pt_lower:
            return pt

    # 3. Alias map
    for alias, pt in _LABEL_ALIASES.items():
        if alias in normalized:
            return pt

    return PatternType.UNCERTAIN


def parse_prediction(raw: dict, input_text: str) -> PredictionResult:
    """
    Convert a raw Ollama response dict into a PredictionResult.

    Args:
        raw:        Parsed JSON dict from the LLM.
        input_text: Original product text (used as rewrite fallback).

    Returns:
        Validated PredictionResult.

    Raises:
        ParseError: If `raw` is not a dict or has a critically malformed structure.
    """
    if not isinstance(raw, dict):
        raise ParseError(f"Expected dict, got {type(raw).__name__}: {raw!r}")

    # --- label ---
    label_str = str(raw.get("label", "Uncertain"))
    label = _resolve_label(label_str)

    # --- confidence ---
    try:
        confidence = float(raw.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    # --- psychological_mechanism ---
    psych = str(raw.get("psychological_mechanism", "")).strip()
    if not psych:
        psych = "unknown"

    # --- evidence_span ---
    span = str(raw.get("evidence_span", "")).strip()
    # If the model returns the full text as span, truncate to first 120 chars as a fallback
    if len(span) > len(input_text) * 0.9 and len(span) > 120:
        span = span[:120].rsplit(" ", 1)[0]

    # --- harm_dimension ---
    harm = str(raw.get("harm_dimension", "")).strip()
    if not harm:
        harm = "unknown"

    # --- rationale ---
    rationale = str(raw.get("rationale", "")).strip()
    if not rationale:
        rationale = "No rationale provided."

    # --- rewrite ---
    rewrite = str(raw.get("rewrite", "")).strip()
    if not rewrite:
        rewrite = input_text  # fall back to original if model omitted rewrite

    try:
        return PredictionResult(
            label=label,
            confidence=confidence,
            psychological_mechanism=psych,
            evidence_span=span,
            harm_dimension=harm,
            rationale=rationale,
            rewrite=rewrite,
        )
    except Exception as exc:
        raise ParseError(f"PredictionResult validation failed: {exc}\nRaw: {raw}") from exc


def parse_batch(
    raws: list[dict],
    input_texts: list[str],
) -> tuple[list[PredictionResult], list[int]]:
    """
    Parse a list of raw dicts.

    Returns:
        (results, failed_indices) where failed_indices lists positions where
        parsing raised ParseError (those entries are replaced with a fallback
        UNCERTAIN result).
    """
    results: list[PredictionResult] = []
    failed: list[int] = []

    for i, (raw, text) in enumerate(zip(raws, input_texts)):
        try:
            results.append(parse_prediction(raw, text))
        except ParseError:
            failed.append(i)
            results.append(
                PredictionResult(
                    label=PatternType.UNCERTAIN,
                    confidence=0.0,
                    psychological_mechanism="parse error",
                    evidence_span="",
                    harm_dimension="unknown",
                    rationale="Output could not be parsed.",
                    rewrite=text,
                )
            )

    return results, failed
