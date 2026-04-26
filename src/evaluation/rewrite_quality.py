"""
Layer 4 — Rewrite quality evaluation (J-score).

Three-component score for the detoxified rewrite field:
  STA — Semantic Textual Alignment: BERTScore F1 between rewrite and input
         (measures content preservation — should be high)
  SIM — Similarity ceiling: MPNet cosine similarity
         (cross-checks BERTScore; should be high but < 1 to confirm manipulation was removed)
  FL  — Fluency: avg log-likelihood approximated via sentence perplexity
         using a small language model (gpt2 or distilgpt2)

J-score = mean(STA, SIM, FL_norm)

Soft NER gate: if the rewrite drops a proper noun that is in the input but not
a dark-pattern trigger, the record is flagged (not excluded) as "entity_leak = True".

Usage:
    python -m src.evaluation.rewrite_quality --file results/pipelines/rag_sbert_diversity_k5_cot.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

ROOT        = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "evaluation"

# ---------------------------------------------------------------------------
# Lazy imports — heavy models loaded only when evaluate() is called
# ---------------------------------------------------------------------------

def _load_bertscore():
    from bert_score import score as bs_score
    return bs_score


def _load_mpnet():
    from sentence_transformers import SentenceTransformer
    import numpy as np
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    return model, np


def _load_perplexity_model():
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    model.eval()
    return tokenizer, model, torch


def _load_spacy():
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return nlp


# ---------------------------------------------------------------------------
# Individual scorers
# ---------------------------------------------------------------------------

def _sta_scores(inputs: list[str], rewrites: list[str]) -> list[float]:
    """BERTScore F1 — semantic alignment between input and rewrite."""
    bs_score = _load_bertscore()
    _, _, F = bs_score(rewrites, inputs, lang="en", verbose=False)
    return [float(f) for f in F]


def _sim_scores(inputs: list[str], rewrites: list[str]) -> list[float]:
    """MPNet cosine similarity."""
    model, np = _load_mpnet()
    in_embs = model.encode(inputs,   convert_to_numpy=True, normalize_embeddings=True)
    rw_embs = model.encode(rewrites, convert_to_numpy=True, normalize_embeddings=True)
    return [float(np.dot(a, b)) for a, b in zip(in_embs, rw_embs)]


def _perplexity(text: str, tokenizer: Any, model: Any, torch: Any) -> float:
    """Token-level perplexity via distilgpt2 (lower = more fluent)."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        loss = model(**enc, labels=enc["input_ids"]).loss
    return math.exp(float(loss))


def _fl_scores_normalised(rewrites: list[str]) -> list[float]:
    """
    Fluency scores normalised to [0, 1].
    Heuristic: perplexity ≤ 20 → 1.0; ≥ 200 → 0.0; log-linear in between.
    """
    tokenizer, model, torch = _load_perplexity_model()
    scores = []
    for text in rewrites:
        if not text.strip():
            scores.append(0.0)
            continue
        ppl = _perplexity(text, tokenizer, model, torch)
        # log-linear normalisation: ln(20)=3.0, ln(200)=5.3
        lo, hi = math.log(20), math.log(200)
        normed = 1.0 - max(0.0, min(1.0, (math.log(max(ppl, 1)) - lo) / (hi - lo)))
        scores.append(normed)
    return scores


def _entity_leak_flags(inputs: list[str], rewrites: list[str]) -> list[bool]:
    """
    True if rewrite drops a proper noun present in the input.
    Proper nouns that are also manipulation triggers (numbers, "only", etc.) are excluded.
    """
    nlp = _load_spacy()
    flags = []
    for inp, rw in zip(inputs, rewrites):
        in_ents  = {e.text.lower() for e in nlp(inp).ents  if e.label_ in ("PERSON","ORG","PRODUCT","GPE")}
        rw_ents  = {e.text.lower() for e in nlp(rw).ents   if e.label_ in ("PERSON","ORG","PRODUCT","GPE")}
        flags.append(bool(in_ents - rw_ents))
    return flags


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(
    records: list[dict],
    skip_perplexity: bool = False,
) -> dict:
    """
    Compute J-score and component scores for a list of prediction records.

    Each record must have "input_text" and "rewrite" keys.

    Args:
        records:         List of prediction dicts.
        skip_perplexity: If True, FL component is set to None (faster, useful for CI).

    Returns:
        Summary dict with aggregate stats and per-record scores.
    """
    inputs  = [r.get("input_text", "") for r in records]
    rewrites = [r.get("rewrite", "")   for r in records]

    n = len(records)
    print(f"  Computing STA (BERTScore) for {n} records...")
    sta = _sta_scores(inputs, rewrites)

    print(f"  Computing SIM (MPNet cosine) for {n} records...")
    sim = _sim_scores(inputs, rewrites)

    if skip_perplexity:
        fl: list[float | None] = [None] * n
        j_scores = [statistics.mean([a, b]) for a, b in zip(sta, sim)]
    else:
        print(f"  Computing FL (distilgpt2 perplexity) for {n} records...")
        fl_vals = _fl_scores_normalised(rewrites)
        fl = fl_vals  # type: ignore[assignment]
        j_scores = [statistics.mean([a, b, c]) for a, b, c in zip(sta, sim, fl_vals)]

    print(f"  Computing entity leak flags...")
    entity_leaks = _entity_leak_flags(inputs, rewrites)

    per_record = []
    for i, rec in enumerate(records):
        entry: dict[str, Any] = {
            "id":           rec.get("id", i),
            "label":        rec.get("label", ""),
            "sta":          round(sta[i], 4),
            "sim":          round(sim[i], 4),
            "fl":           round(fl[i], 4) if fl[i] is not None else None,
            "j_score":      round(j_scores[i], 4),
            "entity_leak":  entity_leaks[i],
            "rewrite_len":  len(rewrites[i].split()),
        }
        per_record.append(entry)

    valid_fl = [x for x in fl if x is not None]
    summary: dict[str, Any] = {
        "n":                n,
        "mean_sta":         round(statistics.mean(sta),       4),
        "mean_sim":         round(statistics.mean(sim),       4),
        "mean_fl":          round(statistics.mean(valid_fl),  4) if valid_fl else None,
        "mean_j_score":     round(statistics.mean(j_scores),  4),
        "entity_leak_rate": round(sum(entity_leaks) / n,      4),
        "per_record":       per_record,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite quality (J-score) evaluator")
    parser.add_argument("--file",            required=True, help="Pipeline JSONL output path")
    parser.add_argument("--skip-perplexity", action="store_true", help="Skip distilgpt2 FL computation")
    parser.add_argument("--out",             default=None,  help="Output JSON path")
    args = parser.parse_args()

    in_path = Path(args.file)
    with open(in_path) as f:
        records = [json.loads(l) for l in f if l.strip()]

    print(f"Evaluating rewrite quality for {len(records)} records from {in_path.name}")
    summary = evaluate(records, skip_perplexity=args.skip_perplexity)

    print(f"\nJ-score (mean):  {summary['mean_j_score']:.4f}")
    print(f"  STA (BERTScore F1):        {summary['mean_sta']:.4f}")
    print(f"  SIM (MPNet cosine):        {summary['mean_sim']:.4f}")
    if summary["mean_fl"] is not None:
        print(f"  FL  (fluency, normalised): {summary['mean_fl']:.4f}")
    print(f"Entity leak rate: {summary['entity_leak_rate']:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else RESULTS_DIR / f"rewrite_quality_{in_path.stem}.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
