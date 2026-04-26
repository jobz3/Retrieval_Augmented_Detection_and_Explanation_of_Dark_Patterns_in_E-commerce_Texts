"""
Full 4-layer rubric evaluation runner.

Wires together all evaluation layers for a single pipeline JSONL output file:

  Layer 1 — Structural validation & calibration
            (classification_metrics + confidence_metrics from metrics.py)
  Layer 2 — Span faithfulness cascade
            (annotate_grounding tiers=3 + cascade_summary from span_grounding.py)
  Layer 3 — LLM-as-judge rationale quality
            (judge_file from rationale_judge.py — requires judge models to be pulled)
  Layer 4 — Rewrite quality J-score
            (evaluate from rewrite_quality.py — requires sentence-transformers, bert-score)

Layers 3 and 4 are opt-in via flags because they are slow/require extra models.

Usage:
    # Layers 1+2 only (fast, no extra dependencies)
    python -m src.evaluation.full_eval --file results/pipelines/rag_sbert_diversity_k5_cot.jsonl

    # All layers
    python -m src.evaluation.full_eval \\
        --file results/pipelines/rag_sbert_diversity_k5_cot.jsonl \\
        --layer3 --layer4

    # Skip perplexity in Layer 4 (faster)
    python -m src.evaluation.full_eval \\
        --file results/pipelines/rag_sbert_diversity_k5_cot.jsonl \\
        --layer3 --layer4 --skip-perplexity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.metrics import (
    ALL_CLASSES,
    classification_metrics,
    confidence_metrics,
    grounding_metrics,
    output_quality_metrics,
)
from src.pipelines.span_grounding import annotate_grounding, cascade_summary

ROOT        = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "evaluation"


def run_layer1(records: list[dict]) -> dict:
    print("  Layer 1: classification + calibration...")
    clf  = classification_metrics(records)
    conf = confidence_metrics(records)
    qual = output_quality_metrics(records)
    return {"classification": clf, "confidence": conf, "output_quality": qual}


def run_layer2(records: list[dict]) -> dict:
    print("  Layer 2: span grounding cascade (Tier 1–3)...")
    annotate_grounding(records, tiers=3)
    cascade = cascade_summary(records)
    legacy  = grounding_metrics(records)   # keeps backward-compat keys (exact_rate, ci_rate, avg_span_len, empty_spans)
    return {"cascade": cascade, "legacy": legacy}


def run_layer3(records: list[dict], limit: int | None, use_cpu: bool) -> dict:
    from src.evaluation.rationale_judge import JUDGE_MODEL, judge_file as _judge_file
    print(f"  Layer 3: rationale judge ({JUDGE_MODEL})...")
    import tempfile, json as _json
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        for r in records:
            tmp.write(_json.dumps(r) + "\n")
        tmp_path = Path(tmp.name)
    try:
        summary = _judge_file(tmp_path, limit=limit, use_cpu=use_cpu)
        summary.pop("records", None)
    finally:
        tmp_path.unlink(missing_ok=True)
    return summary


def run_layer4(records: list[dict], skip_perplexity: bool) -> dict:
    print("  Layer 4: rewrite quality J-score...")
    from src.evaluation.rewrite_quality import evaluate as _evaluate
    summary = _evaluate(records, skip_perplexity=skip_perplexity)
    summary.pop("per_record", None)   # strip per-record detail from top-level output
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Full 4-layer rubric evaluation")
    parser.add_argument("--file",            required=True,              help="Pipeline JSONL output path")
    parser.add_argument("--layer3",          action="store_true",    help="Run Layer 3 (Prometheus-2 judge, slow — needs transformers+bitsandbytes)")
    parser.add_argument("--layer4",          action="store_true",    help="Run Layer 4 (rewrite J-score, needs sentence-transformers)")
    parser.add_argument("--judge-limit",     type=int, default=None, help="Score only first N records in Layer 3")
    parser.add_argument("--judge-cpu",       action="store_true",    help="Force CPU inference for Layer 3 (slow but no VRAM needed)")
    parser.add_argument("--skip-perplexity", action="store_true",    help="Skip distilgpt2 FL in Layer 4")
    parser.add_argument("--out",             default=None,               help="Output JSON path")
    args = parser.parse_args()

    in_path = Path(args.file)
    with open(in_path) as f:
        records = [json.loads(l) for l in f if l.strip()]

    print(f"\nFull rubric evaluation: {in_path.name}  (n={len(records)})")
    print("=" * 60)

    result: dict = {"file": in_path.name, "n": len(records)}

    result["layer1"] = run_layer1(records)
    result["layer2"] = run_layer2(records)

    if args.layer3:
        result["layer3"] = run_layer3(records, args.judge_limit, args.judge_cpu)
    else:
        print("  Layer 3: skipped (pass --layer3 to enable)")

    if args.layer4:
        result["layer4"] = run_layer4(records, args.skip_perplexity)
    else:
        print("  Layer 4: skipped (pass --layer4 to enable)")

    # ------------------------------------------------------------------
    # Summary printout
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    clf = result["layer1"]["classification"]
    print(f"[L1] Macro F1:       {clf['macro_f1']:.4f}")
    print(f"[L1] Cohen's κ:      {clf['cohen_kappa']:.4f}")
    conf = result["layer1"]["confidence"]
    print(f"[L1] Accuracy:       {conf['accuracy']:.4f}")
    print(f"[L1] Calibration gap:{conf['calibration_gap']:.4f}")
    print(f"[L1] Brier score:    {conf['brier_score']:.4f}")
    auroc_str = f"{conf['auroc']:.4f}" if conf["auroc"] is not None else "N/A"
    print(f"[L1] AUROC:          {auroc_str}")

    cas = result["layer2"]["cascade"]
    print(f"[L2] Span exact:     {cas['exact_rate']:.4f}")
    print(f"[L2] Span fuzzy≥90:  {cas['fuzzy_rate']:.4f}")
    print(f"[L2] Span semantic:  {cas['semantic_rate']:.4f}")
    print(f"[L2] Any grounded:   {cas['any_grounded_rate']:.4f}")

    if "layer3" in result:
        l3 = result["layer3"]
        print(f"[L3] Composite (mean): {l3['mean_composite']:.3f}/20  (std={l3['std_composite']:.3f})")
        for criterion in ("coherence", "faithfulness", "specificity", "non_circularity"):
            print(f"[L3]   {criterion:<18}: {l3[f'mean_{criterion}']:.3f}/5")

    if "layer4" in result:
        l4 = result["layer4"]
        print(f"[L4] J-score:        {l4['mean_j_score']:.4f}")
        print(f"[L4]   STA:          {l4['mean_sta']:.4f}")
        print(f"[L4]   SIM:          {l4['mean_sim']:.4f}")
        if l4["mean_fl"] is not None:
            print(f"[L4]   FL:           {l4['mean_fl']:.4f}")
        print(f"[L4] Entity leak:    {l4['entity_leak_rate']:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else RESULTS_DIR / f"full_eval_{in_path.stem}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
