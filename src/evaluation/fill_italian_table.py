"""
After running the Italian cross-lingual pipeline, this helper prints the
two filled-in table rows (zero-shot + RAG multilingual KNN k=5) ready to
paste into paper/main.tex (Table tab:italian_crosslingual).

Run after:
    python -m src.evaluation.cross_lingual_eval --lang it --mode both \
        --encoder multilingual --strategy knn --k 5 \
        --threshold 0.15 --diversity-alpha 0.3

Source artifact:
    results/evaluation/cross_lingual_comparison_it.json

Usage:
    python -m src.evaluation.fill_italian_table
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC  = ROOT / "results" / "evaluation" / "cross_lingual_comparison_it.json"


def _fmt(v: float | None) -> str:
    return "TBD" if v is None else f"{v:.3f}"


def main() -> None:
    if not SRC.exists():
        raise SystemExit(
            f"Missing {SRC}. Run cross_lingual_eval.py with --lang it first."
        )

    data = json.loads(SRC.read_text())
    by_name = {entry["name"]: entry for entry in data}

    zs = by_name.get("zero_shot_italian")
    rag = next(
        (entry for name, entry in by_name.items() if name.startswith("rag_") and name.endswith("_italian")),
        None,
    )

    def row(label: str, entry: dict | None) -> str:
        if entry is None:
            return f"{label:<46} & {'?':>4} & {'TBD':>6} & {'TBD':>6} & {'TBD':>6} \\\\"
        clf = entry["classification"]
        return (
            f"{label:<46} & {entry['n']:>4} & "
            f"{_fmt(clf['macro_f1']):>6} & "
            f"{_fmt(clf['cohen_kappa']):>6} & "
            f"{_fmt(clf['accuracy']):>6} \\\\"
        )

    print("% Paste these two rows into Table `tab:italian_crosslingual`,")
    print("% replacing the two TBD lines.")
    print()
    print(row("Zero-shot (Italian)", zs))
    print(row(r"\textbf{RAG Multilingual IT KNN $k$=5}", rag))

    # Also dump a one-line summary for quick eyeballing.
    print()
    print("# Quick summary:")
    for entry in data:
        clf = entry.get("classification", {})
        f1  = clf.get("macro_f1")
        if f1 is None:
            continue
        print(f"  {entry['name']:<40}  n={entry['n']:>4}  F1={f1:.3f}")


if __name__ == "__main__":
    main()
