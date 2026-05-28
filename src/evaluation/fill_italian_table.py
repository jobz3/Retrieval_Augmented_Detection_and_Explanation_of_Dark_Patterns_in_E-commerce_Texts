"""
After running the Italian cross-lingual pipeline, this helper prints the
three filled-in table rows (zero-shot, RAG multilingual shared-space,
RAG multilingual_it translated) ready to paste into paper/main.tex
(Table tab:italian_crosslingual).

Run after both invocations:
    # zero-shot + RAG with the shared-space multilingual index
    python -m src.evaluation.cross_lingual_eval --lang it --mode both \
        --encoder multilingual --strategy knn --k 5 \
        --threshold 0.15 --diversity-alpha 0.3

    # RAG with the en->it translated index (requires translate_index.py)
    python -m src.evaluation.cross_lingual_eval --lang it --mode rag \
        --encoder multilingual_it --strategy knn --k 5 \
        --threshold 0.15 --diversity-alpha 0.3

Source artifact:
    results/evaluation/cross_lingual_comparison_it.json
    (re-running --lang it overwrites this file; the helper additionally
     reads results/pipelines/italian_rag_*.jsonl directly so all three
     conditions are recovered even if the JSON only has the last run.)

Usage:
    python -m src.evaluation.fill_italian_table
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PIPELINES = ROOT / "results" / "pipelines"
SUMMARY   = ROOT / "results" / "evaluation" / "cross_lingual_comparison_it.json"


def _fmt(v: float | None) -> str:
    return "TBD" if v is None else f"{v:.3f}"


def _metrics_from_records(records: list[dict]) -> dict:
    from sklearn.metrics import f1_score, cohen_kappa_score

    ALL_CLASSES = [
        "Forced Action", "Misdirection", "Not Dark Pattern",
        "Obstruction", "Scarcity", "Sneaking", "Social Proof", "Urgency",
    ]
    golds = [r["gold_label"] for r in records]
    preds = [r["label"]      for r in records]
    # Fixed-taxonomy macro (see metrics.py): avoids the phantom-class deflation
    # from out-of-taxonomy "Uncertain" predictions under the set-union convention.
    return {
        "macro_f1":    round(f1_score(golds, preds, labels=ALL_CLASSES, average="macro", zero_division=0), 4),
        "cohen_kappa": round(cohen_kappa_score(golds, preds, labels=ALL_CLASSES), 4),
        "accuracy":    round(sum(g == p for g, p in zip(golds, preds)) / len(golds), 4),
    }


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _find_run(pattern: str) -> tuple[Path, dict] | None:
    """Return (path, classification metrics) for the most recent jsonl matching pattern, or None."""
    candidates = sorted(PIPELINES.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates:
        recs = _load_jsonl(p)
        if recs:
            return p, _metrics_from_records(recs)
    return None


def main() -> None:
    if not SUMMARY.exists() and not any(PIPELINES.glob("italian_*.jsonl")):
        raise SystemExit(
            "No Italian artifacts found. Run cross_lingual_eval.py --lang it first."
        )

    zs   = _find_run("italian_zero_shot.jsonl")
    rag  = _find_run("italian_rag_multilingual_knn_k5*.jsonl")
    rag_translated = _find_run("italian_rag_multilingual_it_knn_k5*.jsonl")

    def row(label: str, n: int | None, m: dict | None) -> str:
        if m is None:
            return f"{label:<60} & {'?':>4} & {'TBD':>6} & {'TBD':>6} & {'TBD':>6} \\\\"
        return (
            f"{label:<60} & {n:>4} & "
            f"{_fmt(m['macro_f1']):>6} & "
            f"{_fmt(m['cohen_kappa']):>6} & "
            f"{_fmt(m['accuracy']):>6} \\\\"
        )

    def n_of(found: tuple[Path, dict] | None) -> int | None:
        if found is None:
            return None
        return len(_load_jsonl(found[0]))

    print("% Paste these three rows into Table `tab:italian_crosslingual`,")
    print("% replacing the three TBD lines.")
    print()
    print(row("Zero-shot (Italian)",
              n_of(zs), zs[1] if zs else None))
    print(row("RAG Multilingual KNN $k$=5 (shared-space)",
              n_of(rag), rag[1] if rag else None))
    print(row(r"\textbf{RAG Multilingual$_{\text{IT}}$ KNN $k$=5 (translated)}",
              n_of(rag_translated), rag_translated[1] if rag_translated else None))

    # Quick eyeballing summary
    print("\n# Quick summary:")
    for name, found in [("zero_shot", zs), ("multilingual (shared)", rag), ("multilingual_it (translated)", rag_translated)]:
        if found is None:
            print(f"  {name:<32}  MISSING")
            continue
        path, m = found
        n = n_of(found)
        print(f"  {name:<32}  n={n:>4}  F1={m['macro_f1']:.3f}  kappa={m['cohen_kappa']:.3f}  acc={m['accuracy']:.3f}  ({path.name})")


if __name__ == "__main__":
    main()
