"""
Cross-lingual evaluation — German zero-shot transfer.

Runs the best English-trained pipeline (RAG SBERT diversity k=5) on the
manually annotated German dataset (data/german/german_dark_patterns.jsonl)
with NO German training examples (pure zero-shot transfer).

This answers the paper's RQ3: does the pipeline generalise to German?

Outputs:
  results/pipelines/german_zero_shot.jsonl         — per-record predictions
  results/pipelines/german_rag_sbert_diversity.jsonl
  results/evaluation/cross_lingual_comparison.json — summary table

Usage:
    python -m src.evaluation.cross_lingual_eval
    python -m src.evaluation.cross_lingual_eval --mode zero_shot
    python -m src.evaluation.cross_lingual_eval --mode rag --strategy diversity --k 5
    python -m src.evaluation.cross_lingual_eval --mode both
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from src.pipelines.output_parser import parse_prediction, ParseError
from src.pipelines.prompts import SYSTEM_PROMPT, format_zero_shot_prompt, format_few_shot_prompt
from src.pipelines.schema import PatternType, PredictionResult
from src.pipelines.span_grounding import check_grounding, annotate_grounding, cascade_summary
from src.utils.io import save_jsonl
from src.utils.ollama_client import chat_json, DEFAULT_MODEL

ROOT        = Path(__file__).resolve().parents[2]
GERMAN_DATA = ROOT / "data" / "german" / "german_dark_patterns.jsonl"
PIPELINES   = ROOT / "results" / "pipelines"
RESULTS_DIR = ROOT / "results" / "evaluation"

ALL_CLASSES = [
    "Forced Action", "Misdirection", "Not Dark Pattern",
    "Obstruction", "Scarcity", "Sneaking", "Social Proof", "Urgency",
]

app = typer.Typer()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_german() -> list[dict]:
    with open(GERMAN_DATA, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


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
    rec["input_text"] = text
    rec["gold_label"] = gold
    rec["span_exact"] = False
    rec["span_ci"]    = False
    rec["span_length"] = 0
    rec["parse_ok"]   = False
    return rec


def _predict_zero_shot(
    text: str,
    model: str = DEFAULT_MODEL,
) -> tuple[PredictionResult, dict]:
    prompt = format_zero_shot_prompt(text)
    raw    = chat_json(prompt, model=model, system=SYSTEM_PROMPT, temperature=0.0)
    result = parse_prediction(raw, text)
    grounding = check_grounding(result, text)
    return result, grounding


def _predict_rag(
    text: str,
    retriever,
    model: str = DEFAULT_MODEL,
) -> tuple[PredictionResult, dict, list[dict]]:
    from src.pipelines.rag_few_shot import predict as rag_predict
    return rag_predict(text, retriever=retriever, model=model, temperature=0.0)


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------

def run_zero_shot(records: list[dict], model: str = DEFAULT_MODEL) -> list[dict]:
    out_records = []
    n = len(records)
    print(f"Zero-shot German inference ({n} records, model={model}) ...")
    for i, rec in enumerate(records):
        print(f"  [{i+1}/{n}]", end="\r", flush=True)
        text = rec["text"]
        gold = rec["category"]
        try:
            result, grounding = _predict_zero_shot(text, model=model)
            parse_ok = True
        except (ParseError, ValueError) as exc:
            print(f"\n  ⚠ parse error [{i}]: {exc}")
            out_records.append(_make_error_record(text, gold, exc))
            continue

        r = result.to_dict()
        r["id"]          = rec["id"]
        r["input_text"]  = text
        r["gold_label"]  = gold
        r["source"]      = rec.get("source", "")
        r["span_exact"]  = grounding["exact"]
        r["span_ci"]     = grounding["case_insensitive"]
        r["span_length"] = grounding["span_length"]
        r["parse_ok"]    = parse_ok
        out_records.append(r)
    print()
    return out_records


def run_rag(
    records: list[dict],
    strategy: str = "diversity",
    k: int = 5,
    encoder: str = "sbert",
    model: str = DEFAULT_MODEL,
) -> list[dict]:
    from src.retrieval.retrieve import Retriever
    retriever = Retriever(encoder=encoder, strategy=strategy, k=k)

    out_records = []
    n = len(records)
    print(f"RAG German inference (k={k}, encoder={encoder}, strategy={strategy}, {n} records) ...")
    for i, rec in enumerate(records):
        print(f"  [{i+1}/{n}]", end="\r", flush=True)
        text = rec["text"]
        gold = rec["category"]
        try:
            result, grounding, retrieved = _predict_rag(text, retriever=retriever, model=model)
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
        r["retrieved_labels"] = [rv["category"] for rv in retrieved]
        r["retrieved_scores"] = [round(rv["score"], 4) for rv in retrieved]
        out_records.append(r)
    print()
    return out_records


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _classification_metrics(records: list[dict]) -> dict:
    from sklearn.metrics import f1_score, precision_score, recall_score, cohen_kappa_score, classification_report

    golds = [r["gold_label"] for r in records]
    preds = [r["label"]      for r in records]
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
        "per_class":       per_class,
    }


def _grounding_metrics(records: list[dict]) -> dict:
    annotate_grounding(records, tiers=2)
    summary = cascade_summary(records)
    n = len(records)
    return {
        "exact_rate":     round(sum(r["span_exact"] for r in records) / n, 4),
        "ci_rate":        round(sum(r["span_ci"]    for r in records) / n, 4),
        "fuzzy_rate":     summary.get("fuzzy_rate", 0.0),
        "avg_span_len":   round(sum(r["span_length"] for r in records) / n, 2),
        "empty_spans":    sum(r["span_length"] == 0 for r in records),
    }


def print_summary(name: str, clf: dict, grnd: dict, n: int) -> None:
    print(f"\n{'='*55}")
    print(f"  {name}  (n={n})")
    print(f"{'='*55}")
    print(f"  Macro F1:        {clf['macro_f1']:.4f}")
    print(f"  Accuracy:        {clf['accuracy']:.4f}")
    print(f"  Cohen's κ:       {clf['cohen_kappa']:.4f}")
    print(f"  Span exact:      {grnd['exact_rate']:.4f}")
    print(f"  Span fuzzy≥90:   {grnd['fuzzy_rate']:.4f}")
    print(f"  Avg span len:    {grnd['avg_span_len']:.1f} chars")
    print(f"\n  Per-class F1:")
    for cls in ALL_CLASSES:
        pc = clf["per_class"][cls]
        if pc["support"] > 0:
            print(f"    {cls:<20} F1={pc['f1']:.3f}  sup={pc['support']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    mode:     str  = typer.Option("both",      help="zero_shot | rag | both"),
    strategy: str  = typer.Option("diversity", help="RAG strategy: knn | diversity | prototype"),
    k:        int  = typer.Option(5,           help="RAG k"),
    encoder:  str  = typer.Option("sbert",     help="RAG encoder: sbert | bert"),
    model:    str  = typer.Option(DEFAULT_MODEL, help="Ollama model tag"),
) -> None:
    german = load_german()
    print(f"Loaded {len(german)} German records from {GERMAN_DATA}")

    PIPELINES.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    if mode in ("zero_shot", "both"):
        zs_records = run_zero_shot(german, model=model)
        zs_out = PIPELINES / "german_zero_shot.jsonl"
        save_jsonl(zs_records, zs_out)
        print(f"Saved → {zs_out}")

        clf  = _classification_metrics(zs_records)
        grnd = _grounding_metrics(zs_records)
        print_summary("Zero-shot (German)", clf, grnd, len(zs_records))
        all_summaries.append({"name": "zero_shot_german", "n": len(zs_records), "classification": clf, "grounding": grnd})

    if mode in ("rag", "both"):
        rag_records = run_rag(german, strategy=strategy, k=k, encoder=encoder, model=model)
        rag_out = PIPELINES / f"german_rag_{encoder}_{strategy}_k{k}.jsonl"
        save_jsonl(rag_records, rag_out)
        print(f"Saved → {rag_out}")

        clf  = _classification_metrics(rag_records)
        grnd = _grounding_metrics(rag_records)
        print_summary(f"RAG {encoder} {strategy} k={k} (German)", clf, grnd, len(rag_records))
        all_summaries.append({"name": f"rag_{encoder}_{strategy}_k{k}_german", "n": len(rag_records), "classification": clf, "grounding": grnd})

    # Load best English result for comparison
    en_best = PIPELINES / "rag_sbert_diversity_k5.jsonl"
    if en_best.exists():
        with open(en_best) as f:
            en_records = [json.loads(l) for l in f if l.strip()]
        from src.evaluation.metrics import classification_metrics as en_clf_metrics, grounding_metrics as en_grnd_metrics
        en_clf  = en_clf_metrics(en_records)
        en_grnd = en_grnd_metrics(en_records)
        all_summaries.insert(0, {"name": "rag_sbert_diversity_k5_english", "n": len(en_records), "classification": en_clf, "grounding": en_grnd})
        print(f"\n[English baseline] Macro F1: {en_clf['macro_f1']:.4f}  (n={len(en_records)})")

    out = RESULTS_DIR / "cross_lingual_comparison.json"
    with open(out, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nComparison saved → {out}")


if __name__ == "__main__":
    app()
