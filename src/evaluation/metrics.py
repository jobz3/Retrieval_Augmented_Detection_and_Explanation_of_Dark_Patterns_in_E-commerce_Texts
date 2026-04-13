"""
Phase 3 — Pipeline comparison metrics.

Reads the three completed JSONL prediction files and computes:
  1. Classification: macro F1, per-class F1, precision, recall
  2. Span grounding: exact-match rate, case-insensitive rate, avg span length
  3. Confidence: mean, std, calibration gap (|mean_conf - accuracy|)
  4. Output quality: rewrite coverage, rationale length distribution

Usage:
    python -m src.evaluation.metrics
"""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT        = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "evaluation"
PIPELINES   = ROOT / "results" / "pipelines"

RUNS = [
    {"name": "Zero-shot",           "file": "zero_shot.jsonl"},
    {"name": "Random few-shot k=5", "file": "random_few_shot_k5.jsonl"},
    {"name": "RAG sbert knn k=5",   "file": "rag_sbert_knn_k5.jsonl"},
]

ALL_CLASSES = [
    "Forced Action", "Misdirection", "Not Dark Pattern",
    "Obstruction", "Scarcity", "Sneaking", "Social Proof", "Urgency",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def classification_metrics(records: list[dict]) -> dict:
    golds = [r["gold_label"] for r in records]
    preds = [r["label"]      for r in records]

    present = sorted(set(golds) | set(preds))
    macro_f1  = f1_score(golds, preds, labels=present, average="macro",    zero_division=0)
    macro_p   = precision_score(golds, preds, labels=present, average="macro", zero_division=0)
    macro_r   = recall_score(golds, preds, labels=present, average="macro",   zero_division=0)

    report = classification_report(
        golds, preds,
        labels=ALL_CLASSES, target_names=ALL_CLASSES,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(golds, preds, labels=ALL_CLASSES).tolist()

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
        "macro_f1":       round(macro_f1, 4),
        "macro_precision": round(macro_p,  4),
        "macro_recall":    round(macro_r,  4),
        "per_class":       per_class,
        "confusion_matrix": cm,
    }


def grounding_metrics(records: list[dict]) -> dict:
    n           = len(records)
    exact       = sum(r["span_exact"] for r in records) / n
    ci          = sum(r["span_ci"]    for r in records) / n
    avg_len     = sum(r["span_length"] for r in records) / n
    empty       = sum(r["span_length"] == 0 for r in records)
    return {
        "exact_rate":  round(exact,   4),
        "ci_rate":     round(ci,      4),
        "avg_span_len": round(avg_len, 2),
        "empty_spans": empty,
    }


def confidence_metrics(records: list[dict]) -> dict:
    confs     = [r["confidence"] for r in records]
    golds     = [r["gold_label"] for r in records]
    preds     = [r["label"]      for r in records]
    correct   = [g == p for g, p in zip(golds, preds)]

    mean_conf = sum(confs) / len(confs)
    accuracy  = sum(correct) / len(correct)
    calib_gap = abs(mean_conf - accuracy)

    # Bucket into [0,.2), [.2,.4), ..., [.8,1.0]
    buckets = {f"{i/5:.1f}-{(i+1)/5:.1f}": {"count": 0, "correct": 0}
               for i in range(5)}
    for conf, ok in zip(confs, correct):
        bucket = f"{min(int(conf * 5), 4) / 5:.1f}-{min(int(conf * 5) + 1, 5) / 5:.1f}"
        buckets[bucket]["count"]   += 1
        buckets[bucket]["correct"] += int(ok)

    return {
        "mean_confidence": round(mean_conf, 4),
        "accuracy":        round(accuracy,  4),
        "calibration_gap": round(calib_gap, 4),
        "confidence_buckets": buckets,
    }


def output_quality_metrics(records: list[dict]) -> dict:
    rewrite_lens  = [len(r.get("rewrite",  "").split()) for r in records]
    rationale_lens = [len(r.get("rationale", "").split()) for r in records]
    input_lens    = [len(r.get("input_text", "").split()) for r in records]

    # Rewrite coverage: did the model produce a rewrite (>5 tokens)?
    rewrite_ok = sum(l > 5 for l in rewrite_lens)

    return {
        "rewrite_coverage":      round(rewrite_ok / len(records), 4),
        "avg_rewrite_words":     round(sum(rewrite_lens) / len(records),   2),
        "avg_rationale_words":   round(sum(rationale_lens) / len(records), 2),
        "avg_input_words":       round(sum(input_lens) / len(records),     2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    all_results = []

    for run in RUNS:
        path = PIPELINES / run["file"]
        if not path.exists():
            print(f"  ⚠ Missing: {path} — skipping")
            continue

        records = load(path)
        print(f"\nEvaluating: {run['name']}  (n={len(records)})")

        clf   = classification_metrics(records)
        grnd  = grounding_metrics(records)
        conf  = confidence_metrics(records)
        qual  = output_quality_metrics(records)

        all_results.append({
            "name":            run["name"],
            "n":               len(records),
            "classification":  clf,
            "grounding":       grnd,
            "confidence":      conf,
            "output_quality":  qual,
        })

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("PIPELINE COMPARISON — CLASSIFICATION")
    print("=" * 75)
    header = f"{'Metric':<28}" + "".join(f"{r['name']:>22}" for r in all_results)
    print(header)
    print("-" * 75)

    def row(label, fn):
        return f"{label:<28}" + "".join(f"{fn(r):>22}" for r in all_results)

    print(row("Macro F1",        lambda r: f"{r['classification']['macro_f1']:.4f}"))
    print(row("Macro Precision", lambda r: f"{r['classification']['macro_precision']:.4f}"))
    print(row("Macro Recall",    lambda r: f"{r['classification']['macro_recall']:.4f}"))

    print("\n  Per-class F1:")
    for cls in ALL_CLASSES:
        print(row(f"  {cls}", lambda r, c=cls: f"{r['classification']['per_class'][c]['f1']:.4f}"))

    print("\n" + "=" * 75)
    print("SPAN GROUNDING")
    print("=" * 75)
    print(row("Exact match rate",   lambda r: f"{r['grounding']['exact_rate']:.4f}"))
    print(row("Case-insensitive",   lambda r: f"{r['grounding']['ci_rate']:.4f}"))
    print(row("Avg span length",    lambda r: f"{r['grounding']['avg_span_len']:.1f} tokens"))
    print(row("Empty spans",        lambda r: str(r['grounding']['empty_spans'])))

    print("\n" + "=" * 75)
    print("CONFIDENCE & CALIBRATION")
    print("=" * 75)
    print(row("Mean confidence",   lambda r: f"{r['confidence']['mean_confidence']:.4f}"))
    print(row("Accuracy",          lambda r: f"{r['confidence']['accuracy']:.4f}"))
    print(row("Calibration gap",   lambda r: f"{r['confidence']['calibration_gap']:.4f}"))

    print("\n" + "=" * 75)
    print("OUTPUT QUALITY")
    print("=" * 75)
    print(row("Rewrite coverage",      lambda r: f"{r['output_quality']['rewrite_coverage']:.4f}"))
    print(row("Avg rewrite (words)",   lambda r: f"{r['output_quality']['avg_rewrite_words']:.1f}"))
    print(row("Avg rationale (words)", lambda r: f"{r['output_quality']['avg_rationale_words']:.1f}"))

    print()

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "pipeline_comparison.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved → {out}")


if __name__ == "__main__":
    main()
