"""
Cross-lingual evaluation — zero-shot and RAG transfer for non-English test sets.

Supports German (default, `--lang de`) and Italian (`--lang it`). Both languages
share the same evaluation logic; only the source dataset and the output filename
prefix change.

Improvements over baseline:
  #1 Retrieval threshold — if max retrieved score < threshold, fall back to zero-shot
  #2 K tuning — sweep k=1,2,3,5 to find optimal k for low-quality retrieval
  #3 Label diversity re-ranking — penalise retrievals where all k neighbours share one label
  #4 Language-aware system prompt — language-agnostic prompt accepts any input language
  #5 Confidence-based abstention — flag low-confidence predictions for analysis

Usage:
    # German (existing)
    python -m src.evaluation.cross_lingual_eval --mode both --strategy knn
    python -m src.evaluation.cross_lingual_eval --mode ktune
    python -m src.evaluation.cross_lingual_eval --mode both --strategy knn --threshold 0.15 --diversity-alpha 0.3
    python -m src.evaluation.cross_lingual_eval --mode both --encoder multilingual --strategy knn --k 1

    # Italian (CLiC-it 2026 cross-lingual check)
    python -m src.evaluation.cross_lingual_eval --lang it --mode both --encoder multilingual --strategy knn --k 5
    python -m src.evaluation.cross_lingual_eval --lang it --mode ktune --encoder multilingual --strategy knn
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import typer

from src.pipelines.output_parser import parse_prediction, ParseError
from src.pipelines.prompts import SYSTEM_PROMPT, format_zero_shot_prompt, format_few_shot_prompt, system_prompt_for
from src.pipelines.schema import PatternType, PredictionResult
from src.pipelines.span_grounding import check_grounding, annotate_grounding, cascade_summary
from src.utils.io import save_jsonl
from src.utils.ollama_client import chat_json, DEFAULT_MODEL

ROOT        = Path(__file__).resolve().parents[2]
PIPELINES   = ROOT / "results" / "pipelines"
RESULTS_DIR = ROOT / "results" / "evaluation"

# Per-language dataset paths and output filename prefixes. Adding a new language
# is a single entry here: (jsonl path, output prefix).
LANG_DATA: dict[str, tuple[Path, str]] = {
    "de": (ROOT / "data" / "german"  / "german_dark_patterns.jsonl",  "german"),
    "it": (ROOT / "data" / "italian" / "italian_dark_patterns.jsonl", "italian"),
}

# Back-compat alias retained for any external scripts that imported GERMAN_DATA.
GERMAN_DATA = LANG_DATA["de"][0]

ALL_CLASSES = [
    "Forced Action", "Misdirection", "Not Dark Pattern",
    "Obstruction", "Scarcity", "Sneaking", "Social Proof", "Urgency",
]

# Improvement #5: confidence threshold for abstention flagging
ABSTENTION_THRESHOLD = 0.6

app = typer.Typer()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_lang(lang: str) -> list[dict]:
    """Load the per-language test set as a list of records."""
    if lang not in LANG_DATA:
        raise ValueError(f"Unsupported language '{lang}'. Choose: {sorted(LANG_DATA)}")
    path, _ = LANG_DATA[lang]
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_german() -> list[dict]:
    """Back-compat shim — prefer load_lang('de')."""
    return load_lang("de")


def _make_error_record(text: str, gold: str, exc: Exception) -> dict:
    result = PredictionResult(
        label=PatternType.UNCERTAIN,
        confidence=0.0,
        psychological_mechanism="parse error",
        evidence_span="",
        harm_dimension="unknown",
        rationale="Output could not be parsed.",
        rewrite=text,
    )
    rec = result.to_dict()
    rec["input_text"]    = text
    rec["gold_label"]    = gold
    rec["span_exact"]    = False
    rec["span_ci"]       = False
    rec["span_length"]   = 0
    rec["parse_ok"]      = False
    rec["used_fallback"] = False
    rec["abstained"]     = False
    return rec


def _predict_zero_shot(text: str, model: str = DEFAULT_MODEL, lang: str | None = None) -> tuple[PredictionResult, dict]:
    prompt = format_zero_shot_prompt(text)
    raw    = chat_json(prompt, model=model, system=system_prompt_for(lang), temperature=0.0)
    result = parse_prediction(raw, text)
    return result, check_grounding(result, text)


# ---------------------------------------------------------------------------
# Improvement #3: label diversity re-ranking
# ---------------------------------------------------------------------------

def _diversity_rerank(retrieved: list[dict], alpha: float = 0.3) -> list[dict]:
    """
    Re-rank retrieved examples to penalise label homogeneity.

    For each candidate, score = sim_score - alpha * label_redundancy_penalty.
    label_redundancy_penalty = fraction of already-selected examples with same label.

    Args:
        retrieved: list of dicts with "score", "category" keys (sorted by score desc).
        alpha:     penalty weight [0=no reranking, 1=full diversity].

    Returns:
        Re-ranked list (same length).
    """
    if not retrieved or alpha == 0.0:
        return retrieved

    selected: list[dict] = []
    remaining = list(retrieved)

    while remaining:
        selected_labels = Counter(r["category"] for r in selected)
        n_selected = len(selected)

        best_idx, best_score = 0, float("-inf")
        for i, cand in enumerate(remaining):
            redundancy = selected_labels.get(cand["category"], 0) / max(n_selected, 1)
            adjusted   = cand["score"] - alpha * redundancy
            if adjusted > best_score:
                best_score = adjusted
                best_idx   = i

        selected.append(remaining.pop(best_idx))

    return selected


# ---------------------------------------------------------------------------
# RAG predict with improvements #1 + #3
# ---------------------------------------------------------------------------

def _predict_rag_improved(
    text: str,
    retriever,
    model: str = DEFAULT_MODEL,
    threshold: float = 0.0,
    diversity_alpha: float = 0.0,
    lang: str | None = None,
) -> tuple[PredictionResult, dict, list[dict], bool]:
    """
    RAG prediction with optional threshold fallback (#1) and diversity re-ranking (#3).

    Returns:
        (result, grounding, retrieved, used_fallback)
    """
    retrieved = retriever.retrieve(text)

    # Improvement #1: threshold fallback
    max_score = max((r["score"] for r in retrieved), default=0.0)
    if threshold > 0.0 and max_score < threshold:
        result, grounding = _predict_zero_shot(text, model=model, lang=lang)
        return result, grounding, retrieved, True

    # Improvement #3: diversity re-ranking
    if diversity_alpha > 0.0:
        retrieved = _diversity_rerank(retrieved, alpha=diversity_alpha)

    examples  = [{"text": r["text"], "category": r["category"]} for r in retrieved]
    prompt    = format_few_shot_prompt(text, examples)
    raw       = chat_json(prompt, model=model, system=system_prompt_for(lang), temperature=0.0)
    result    = parse_prediction(raw, text)
    grounding = check_grounding(result, text)
    return result, grounding, retrieved, False


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------

def run_zero_shot(records: list[dict], model: str = DEFAULT_MODEL, lang: str = "de") -> list[dict]:
    out_records = []
    n = len(records)
    print(f"Zero-shot {lang.upper()} inference ({n} records) ...")
    for i, rec in enumerate(records):
        print(f"  [{i+1}/{n}]", end="\r", flush=True)
        text, gold = rec["text"], rec["category"]
        try:
            result, grounding = _predict_zero_shot(text, model=model, lang=lang)
            parse_ok = True
        except (ParseError, ValueError) as exc:
            print(f"\n  ⚠ parse error [{i}]: {exc}")
            out_records.append(_make_error_record(text, gold, exc))
            continue

        r = result.to_dict()
        r["id"]           = rec["id"]
        r["input_text"]   = text
        r["gold_label"]   = gold
        r["source"]       = rec.get("source", "")
        r["span_exact"]   = grounding["exact"]
        r["span_ci"]      = grounding["case_insensitive"]
        r["span_length"]  = grounding["span_length"]
        r["parse_ok"]     = parse_ok
        r["used_fallback"] = False
        # Improvement #5: abstention flag
        r["abstained"]    = r["confidence"] < ABSTENTION_THRESHOLD
        out_records.append(r)
    print()
    return out_records


def run_rag(
    records: list[dict],
    strategy: str = "knn",
    k: int = 5,
    encoder: str = "sbert",
    model: str = DEFAULT_MODEL,
    threshold: float = 0.0,
    diversity_alpha: float = 0.0,
    lang: str = "de",
    warmup: int = 0,
) -> list[dict]:
    from src.retrieval.retrieve import Retriever
    retriever = Retriever(encoder=encoder, strategy=strategy, k=k)

    # Optional LLM warm-up: fire `warmup` throwaway RAG predictions before the
    # real loop so a cold-start (--mode rag) run reaches the same warm
    # model/KV-cache/CUDA-kernel state as a --mode both run (which is implicitly
    # warmed by the preceding zero-shot pass). Results are discarded; this only
    # removes the cold-vs-warm confound that made the two modes disagree.
    if warmup > 0 and records:
        n_warm = min(warmup, len(records))
        print(f"  warming up LLM with {n_warm} throwaway RAG call(s) ...")
        for rec in records[:n_warm]:
            try:
                _predict_rag_improved(
                    rec["text"], retriever=retriever, model=model,
                    threshold=threshold, diversity_alpha=diversity_alpha, lang=lang,
                )
            except Exception:
                pass  # warm-up failures are irrelevant; we discard output

    out_records = []
    n = len(records)
    tag = f"k={k}, encoder={encoder}, strategy={strategy}, threshold={threshold}, div_alpha={diversity_alpha}, warmup={warmup}"
    print(f"RAG {lang.upper()} inference ({tag}, {n} records) ...")
    for i, rec in enumerate(records):
        print(f"  [{i+1}/{n}]", end="\r", flush=True)
        text, gold = rec["text"], rec["category"]
        try:
            result, grounding, retrieved, used_fallback = _predict_rag_improved(
                text, retriever=retriever, model=model,
                threshold=threshold, diversity_alpha=diversity_alpha,
                lang=lang,
            )
            parse_ok = True
        except (ParseError, ValueError) as exc:
            print(f"\n  ⚠ parse error [{i}]: {exc}")
            out_records.append(_make_error_record(text, gold, exc))
            continue

        r = result.to_dict()
        r["id"]               = rec["id"]
        r["input_text"]       = text
        r["gold_label"]       = gold
        r["source"]           = rec.get("source", "")
        r["span_exact"]       = grounding["exact"]
        r["span_ci"]          = grounding["case_insensitive"]
        r["span_length"]      = grounding["span_length"]
        r["parse_ok"]         = parse_ok
        r["k"]                = k
        r["encoder"]          = encoder
        r["strategy"]         = strategy
        r["threshold"]        = threshold
        r["diversity_alpha"]  = diversity_alpha
        r["used_fallback"]    = used_fallback
        r["retrieved_labels"] = [rv["category"] for rv in retrieved]
        r["retrieved_scores"] = [round(rv["score"], 4) for rv in retrieved]
        r["max_ret_score"]    = round(max((rv["score"] for rv in retrieved), default=0.0), 4)
        # Improvement #5: abstention flag
        r["abstained"]        = r["confidence"] < ABSTENTION_THRESHOLD
        out_records.append(r)
    print()
    return out_records


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _classification_metrics(records: list[dict], abstain: bool = False) -> dict:
    from sklearn.metrics import f1_score, precision_score, recall_score, cohen_kappa_score, classification_report

    # Improvement #5: optionally exclude abstained records
    eval_recs = [r for r in records if not (abstain and r.get("abstained", False))]
    n_abstained = len(records) - len(eval_recs)

    golds   = [r["gold_label"] for r in eval_recs]
    preds   = [r["label"]      for r in eval_recs]
    present = sorted(set(golds) | set(preds))

    macro_f1 = f1_score(golds, preds, labels=present, average="macro",    zero_division=0)
    macro_p  = precision_score(golds, preds, labels=present, average="macro", zero_division=0)
    macro_r  = recall_score(golds, preds, labels=present, average="macro",   zero_division=0)
    kappa    = cohen_kappa_score(golds, preds, labels=ALL_CLASSES)

    report = classification_report(
        golds, preds,
        labels=ALL_CLASSES, target_names=ALL_CLASSES,
        output_dict=True, zero_division=0,
    )
    per_class = {
        cls: {
            "f1":        round(report[cls]["f1-score"],  4),
            "precision": round(report[cls]["precision"], 4),
            "recall":    round(report[cls]["recall"],    4),
            "support":   report[cls]["support"],
        }
        for cls in ALL_CLASSES
    }
    return {
        "macro_f1":        round(macro_f1, 4),
        "macro_precision": round(macro_p,  4),
        "macro_recall":    round(macro_r,  4),
        "cohen_kappa":     round(kappa,    4),
        "accuracy":        round(sum(g == p for g, p in zip(golds, preds)) / len(golds), 4),
        "n_evaluated":     len(eval_recs),
        "n_abstained":     n_abstained,
        "per_class":       per_class,
    }


def _grounding_metrics(records: list[dict]) -> dict:
    annotate_grounding(records, tiers=2)
    summary = cascade_summary(records)
    n = len(records)
    return {
        "exact_rate":   round(sum(r["span_exact"] for r in records) / n, 4),
        "ci_rate":      round(sum(r["span_ci"]    for r in records) / n, 4),
        "fuzzy_rate":   summary.get("fuzzy_rate", 0.0),
        "avg_span_len": round(sum(r["span_length"] for r in records) / n, 2),
        "empty_spans":  sum(r["span_length"] == 0 for r in records),
    }


def _fallback_stats(records: list[dict]) -> dict:
    n_fallback = sum(r.get("used_fallback", False) for r in records)
    fallback_correct = sum(
        r["gold_label"] == r["label"]
        for r in records if r.get("used_fallback", False)
    )
    rag_correct = sum(
        r["gold_label"] == r["label"]
        for r in records if not r.get("used_fallback", False)
    )
    n_rag = len(records) - n_fallback
    return {
        "n_fallback":         n_fallback,
        "fallback_rate":      round(n_fallback / len(records), 4),
        "fallback_accuracy":  round(fallback_correct / n_fallback, 4) if n_fallback else None,
        "rag_accuracy":       round(rag_correct / n_rag, 4) if n_rag else None,
    }


def print_summary(name: str, clf: dict, grnd: dict, n: int, fallback: dict | None = None) -> None:
    print(f"\n{'='*58}")
    print(f"  {name}  (n={n})")
    print(f"{'='*58}")
    print(f"  Macro F1:        {clf['macro_f1']:.4f}")
    print(f"  Accuracy:        {clf['accuracy']:.4f}")
    print(f"  Cohen's κ:       {clf['cohen_kappa']:.4f}")
    if fallback:
        print(f"  Fallbacks:       {fallback['n_fallback']}/{n} ({fallback['fallback_rate']:.1%})")
        if fallback["fallback_accuracy"] is not None:
            print(f"  Fallback acc:    {fallback['fallback_accuracy']:.4f}")
        if fallback["rag_accuracy"] is not None:
            print(f"  RAG-only acc:    {fallback['rag_accuracy']:.4f}")
    abstained = clf.get("n_abstained", 0)
    if abstained:
        print(f"  Abstained:       {abstained} records excluded")
    print(f"  Span exact:      {grnd['exact_rate']:.4f}")
    print(f"  Span fuzzy≥90:   {grnd['fuzzy_rate']:.4f}")
    print(f"\n  Per-class F1 (support > 0):")
    for cls in ALL_CLASSES:
        pc = clf["per_class"][cls]
        if pc["support"] > 0:
            print(f"    {cls:<20} F1={pc['f1']:.3f}  P={pc['precision']:.3f}  R={pc['recall']:.3f}  n={int(pc['support'])}")


# ---------------------------------------------------------------------------
# Improvement #2: K tuning
# ---------------------------------------------------------------------------

def run_ktune(
    records: list[dict],
    strategy: str = "knn",
    encoder: str = "sbert",
    model: str = DEFAULT_MODEL,
    threshold: float = 0.15,
    diversity_alpha: float = 0.3,
    k_values: list[int] | None = None,
    lang: str = "de",
) -> dict:
    """Sweep k values and report macro F1 for each. Default: [1, 2, 3, 5, 7, 10, 15]."""
    from sklearn.metrics import f1_score

    if k_values is None:
        k_values = [1, 2, 3, 5, 7, 10, 15]

    prefix = LANG_DATA[lang][1]
    results = {}
    print(f"\nK-tuning sweep (lang={lang}, strategy={strategy}, encoder={encoder}, threshold={threshold}, k_values={k_values}) ...")
    for k in k_values:
        recs = run_rag(
            records, strategy=strategy, k=k, encoder=encoder, model=model,
            threshold=threshold, diversity_alpha=diversity_alpha, lang=lang,
        )
        golds = [r["gold_label"] for r in recs]
        preds = [r["label"]      for r in recs]
        present = sorted(set(golds) | set(preds))
        f1 = f1_score(golds, preds, labels=present, average="macro", zero_division=0)
        results[k] = round(f1, 4)
        print(f"  k={k}: macro F1 = {f1:.4f}")

        out = PIPELINES / f"{prefix}_rag_{encoder}_{strategy}_k{k}_improved.jsonl"
        save_jsonl(recs, out)

    best_k = max(results, key=results.__getitem__)
    print(f"\n  Best k = {best_k}  (F1 = {results[best_k]:.4f})")
    return {"k_sweep": results, "best_k": best_k}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    lang:            str   = typer.Option("de",    help="Test-set language: de | it"),
    mode:            str   = typer.Option("both",  help="zero_shot | rag | both | ktune"),
    strategy:        str   = typer.Option("knn",   help="RAG strategy: knn | diversity | prototype"),
    k:               int   = typer.Option(5,       help="RAG k"),
    k_values:        str   = typer.Option("",      help="Comma-separated k values for ktune, e.g. '1,2,3,5,7,10,15'. Empty = default."),
    encoder:         str   = typer.Option("sbert", help="RAG encoder: sbert | bert | multilingual | multilingual_<lang> (e.g. multilingual_de, multilingual_it)"),
    model:           str   = typer.Option(DEFAULT_MODEL, help="Ollama model tag"),
    threshold:       float = typer.Option(0.15,   help="[#1] Min retrieval score; below → zero-shot fallback. 0=disabled"),
    diversity_alpha: float = typer.Option(0.3,    help="[#3] Label diversity penalty weight. 0=disabled"),
    abstain:         bool  = typer.Option(False,   help="[#5] Exclude low-confidence predictions from metrics"),
    warmup:          int   = typer.Option(0,      help="Throwaway RAG calls before the real loop to warm the LLM (removes cold-vs-warm mode confound)"),
) -> None:
    parsed_k_values = [int(x) for x in k_values.split(",") if x.strip()] if k_values.strip() else None
    records = load_lang(lang)
    prefix  = LANG_DATA[lang][1]
    print(f"Loaded {len(records)} {prefix.capitalize()} records")

    PIPELINES.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    if mode == "ktune":
        ktune_result = run_ktune(
            records, strategy=strategy, encoder=encoder, model=model,
            threshold=threshold, diversity_alpha=diversity_alpha,
            k_values=parsed_k_values, lang=lang,
        )
        out = RESULTS_DIR / f"{prefix}_ktune_{encoder}_{strategy}.json"
        with open(out, "w") as f:
            json.dump(ktune_result, f, indent=2)
        print(f"\nK-tune results saved → {out}")
        return

    if mode in ("zero_shot", "both"):
        zs_records = run_zero_shot(records, model=model, lang=lang)
        zs_out = PIPELINES / f"{prefix}_zero_shot.jsonl"
        save_jsonl(zs_records, zs_out)

        clf  = _classification_metrics(zs_records, abstain=abstain)
        grnd = _grounding_metrics(zs_records)
        print_summary(f"Zero-shot ({prefix.capitalize()})", clf, grnd, len(zs_records))
        all_summaries.append({"name": f"zero_shot_{prefix}", "n": len(zs_records), "classification": clf, "grounding": grnd})

    if mode in ("rag", "both"):
        rag_records = run_rag(
            records, strategy=strategy, k=k, encoder=encoder, model=model,
            threshold=threshold, diversity_alpha=diversity_alpha, lang=lang,
            warmup=warmup,
        )
        suffix = f"_t{threshold}_d{diversity_alpha}".replace(".", "p") if (threshold or diversity_alpha) else ""
        rag_out = PIPELINES / f"{prefix}_rag_{encoder}_{strategy}_k{k}{suffix}.jsonl"
        save_jsonl(rag_records, rag_out)

        clf      = _classification_metrics(rag_records, abstain=abstain)
        grnd     = _grounding_metrics(rag_records)
        fallback = _fallback_stats(rag_records)
        print_summary(f"RAG {encoder} {strategy} k={k} ({prefix.capitalize()})", clf, grnd, len(rag_records), fallback=fallback)
        all_summaries.append({
            "name": f"rag_{encoder}_{strategy}_k{k}_{prefix}",
            "n": len(rag_records),
            "classification": clf,
            "grounding": grnd,
            "fallback": fallback,
            "threshold": threshold,
            "diversity_alpha": diversity_alpha,
        })

    # Load English baseline for comparison
    en_best = PIPELINES / "rag_sbert_knn_k5.jsonl"
    if en_best.exists():
        with open(en_best) as f:
            en_records = [json.loads(l) for l in f if l.strip()]
        from src.evaluation.metrics import classification_metrics as en_clf_fn, grounding_metrics as en_grnd_fn
        en_clf  = en_clf_fn(en_records)
        en_grnd = en_grnd_fn(en_records)
        all_summaries.insert(0, {
            "name": "rag_sbert_knn_k5_english",
            "n": len(en_records),
            "classification": en_clf,
            "grounding": en_grnd,
        })
        print(f"\n[English baseline] Macro F1: {en_clf['macro_f1']:.4f}  (n={len(en_records)})")

    out = RESULTS_DIR / f"cross_lingual_comparison_{lang}.json"
    with open(out, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved → {out}")

    # Improvement #5: abstention analysis
    if mode in ("rag", "both") and not abstain:
        rag_records_ref = [r for r in rag_records]
        low_conf = [r for r in rag_records_ref if r.get("abstained", False)]
        if low_conf:
            low_correct = sum(r["gold_label"] == r["label"] for r in low_conf)
            high_conf   = [r for r in rag_records_ref if not r.get("abstained", False)]
            high_correct = sum(r["gold_label"] == r["label"] for r in high_conf)
            print(f"\n[#5 Abstention analysis]")
            print(f"  Low-confidence  (conf < {ABSTENTION_THRESHOLD}): {len(low_conf)} records, accuracy = {low_correct/len(low_conf):.3f}")
            print(f"  High-confidence (conf ≥ {ABSTENTION_THRESHOLD}): {len(high_conf)} records, accuracy = {high_correct/len(high_conf):.3f}")


if __name__ == "__main__":
    app()
