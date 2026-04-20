"""
Span grounding verification — four-tier cascade.

Tier 1 — exact:            verbatim substring (strictest)
Tier 2 — fuzzy:            rapidfuzz partial_ratio ≥ 90 (handles minor whitespace/punctuation edits)
Tier 3 — semantic:         MiniLM cosine similarity ≥ 0.75 (catches paraphrase)
Tier 4 — (reserved)        NLI entailment via MiniCheck — not implemented; kept as placeholder

A span passes grounding at the *highest* tier it satisfies (1 being best).
For aggregate reporting, exact_rate / fuzzy_rate / semantic_rate are all computed
so the paper can discuss the cascade separately.

Usage:
    from src.pipelines.span_grounding import check_grounding, grounding_rate, annotate_grounding
    info = check_grounding(result, input_text)
    rate = grounding_rate(results, texts)
"""

from __future__ import annotations

from src.pipelines.schema import PredictionResult

# Thresholds
_FUZZY_THRESHOLD    = 90    # rapidfuzz partial_ratio [0–100]
_SEMANTIC_THRESHOLD = 0.75  # MiniLM cosine similarity [0–1]
_MINILM_MODEL       = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Lazy singletons — only loaded if Tier 2/3 are actually used
# ---------------------------------------------------------------------------

_fuzz     = None
_minilm   = None


def _get_fuzz():
    global _fuzz
    if _fuzz is None:
        try:
            from rapidfuzz import fuzz as _rf
            _fuzz = _rf
        except ImportError as e:
            raise ImportError("rapidfuzz is required for Tier 2 grounding: pip install rapidfuzz") from e
    return _fuzz


def _get_minilm():
    global _minilm
    if _minilm is None:
        try:
            from sentence_transformers import SentenceTransformer
            _minilm = SentenceTransformer(_MINILM_MODEL)
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for Tier 3 grounding: pip install sentence-transformers"
            ) from e
    return _minilm


# ---------------------------------------------------------------------------
# Core check
# ---------------------------------------------------------------------------

def check_grounding(
    result: PredictionResult,
    input_text: str,
    tiers: int = 1,
) -> dict:
    """
    Verify that result.evidence_span is grounded in input_text.

    Args:
        result:     PredictionResult object.
        input_text: Original product text.
        tiers:      How many tiers to evaluate (1, 2, or 3).
                    Higher tiers require additional packages (rapidfuzz, sentence-transformers).

    Returns a dict with:
        exact            (bool)  — Tier 1: verbatim substring
        case_insensitive (bool)  — Tier 1b: case-folded substring
        fuzzy            (bool)  — Tier 2: rapidfuzz ≥ 90 (only if tiers ≥ 2)
        fuzzy_score      (int)   — raw rapidfuzz score [0–100]
        semantic         (bool)  — Tier 3: MiniLM cosine ≥ 0.75 (only if tiers ≥ 3)
        semantic_score   (float) — raw cosine similarity
        best_tier        (int)   — lowest (best) tier satisfied; 0 if none
        span_length      (int)   — character length of the span
        span             (str)   — the evidence span value
    """
    span = result.evidence_span.strip()
    exact = span in input_text
    ci    = span.lower() in input_text.lower()

    out: dict = {
        "exact":            exact,
        "case_insensitive": ci,
        "fuzzy":            False,
        "fuzzy_score":      0,
        "semantic":         False,
        "semantic_score":   0.0,
        "best_tier":        0,
        "span_length":      len(span),
        "span":             span,
    }

    if exact:
        out["best_tier"] = 1
        # Exact implies fuzzy and semantic also pass — mark them so cascade rates are cumulative
        out["fuzzy"]  = True
        out["semantic"] = True
        if tiers == 1:
            return out

    if tiers >= 2 and span and not exact:
        fuzz = _get_fuzz()
        score = fuzz.partial_ratio(span.lower(), input_text.lower())
        out["fuzzy_score"] = score
        out["fuzzy"] = score >= _FUZZY_THRESHOLD
        if out["fuzzy"] and out["best_tier"] == 0:
            out["best_tier"] = 2

    if tiers >= 3 and span and not exact:
        import numpy as np
        model = _get_minilm()
        embs = model.encode([span, input_text], convert_to_numpy=True, normalize_embeddings=True)
        cos_sim = float(np.dot(embs[0], embs[1]))
        out["semantic_score"] = round(cos_sim, 4)
        out["semantic"] = cos_sim >= _SEMANTIC_THRESHOLD
        if out["semantic"] and out["best_tier"] == 0:
            out["best_tier"] = 3

    return out


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------

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
    tiers: int = 2,
) -> list[dict]:
    """
    Add grounding fields to a list of result dicts (as returned by pipeline predict functions).

    Expects each dict to have "evidence_span" and "input_text" keys.

    Adds (Tier 1):  "span_exact", "span_ci", "span_length"
    Adds (Tier 2):  "span_fuzzy", "span_fuzzy_score"
    Adds (Tier 3):  "span_semantic", "span_semantic_score"
    Adds always:    "span_best_tier"

    Args:
        records: List of prediction dicts (mutated in-place).
        tiers:   How many tiers to compute (1, 2, or 3).
    """
    for rec in records:
        span = rec.get("evidence_span", "").strip()
        text = rec.get("input_text", "")

        # Build a minimal PredictionResult-like proxy so check_grounding can be reused
        class _Proxy:
            evidence_span = span

        info = check_grounding(_Proxy(), text, tiers=tiers)  # type: ignore[arg-type]

        rec["span_exact"]          = info["exact"]
        rec["span_ci"]             = info["case_insensitive"]
        rec["span_length"]         = info["span_length"]
        rec["span_fuzzy"]          = info["fuzzy"]
        rec["span_fuzzy_score"]    = info["fuzzy_score"]
        rec["span_semantic"]       = info["semantic"]
        rec["span_semantic_score"] = info["semantic_score"]
        rec["span_best_tier"]      = info["best_tier"]

    return records


def cascade_summary(records: list[dict]) -> dict:
    """
    Compute cumulative tier grounding rates from annotated records.

    Rates are cumulative: fuzzy_rate includes exact-passing records,
    semantic_rate includes exact- and fuzzy-passing records.

    Expects records already processed by annotate_grounding(tiers≥2).

    Returns:
        dict with exact_rate, fuzzy_rate, semantic_rate,
        any_grounded_rate, avg_fuzzy_score, avg_semantic_score
    """
    n = len(records)
    if n == 0:
        return {}

    exact    = sum(r.get("span_exact",    False) for r in records)
    fuzzy    = sum(r.get("span_fuzzy",    False) for r in records)   # cumulative: includes exact
    semantic = sum(r.get("span_semantic", False) for r in records)   # cumulative: includes exact+fuzzy
    any_gr   = sum(r.get("span_best_tier", 0) > 0 for r in records)

    avg_fuzz = sum(r.get("span_fuzzy_score",    0.0) for r in records) / n
    avg_sem  = sum(r.get("span_semantic_score", 0.0) for r in records) / n

    return {
        "exact_rate":         round(exact    / n, 4),
        "fuzzy_rate":         round(fuzzy    / n, 4),
        "semantic_rate":      round(semantic / n, 4),
        "any_grounded_rate":  round(any_gr   / n, 4),
        "avg_fuzzy_score":    round(avg_fuzz,     2),
        "avg_semantic_score": round(avg_sem,      4),
    }
