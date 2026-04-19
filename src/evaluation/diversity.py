"""
N-gram Diversity Analysis
--------------------------
Compares lexical diversity between real and synthetic examples
for the three augmented classes (Forced Action, Sneaking, Obstruction).

Metrics reported per class:
  - TTR        : Type-Token Ratio  (unique tokens / total tokens)
  - Distinct-1 : unique unigrams  / total unigrams
  - Distinct-2 : unique bigrams   / total bigrams
  - Avg length : mean token count per example
  - Vocab size : total unique tokens

If synthetic examples cluster tightly (low Distinct-2, high n-gram overlap
with the training set), that supports the overfitting concern.
Also reports: pairwise BLEU overlap within synthetic vs within real sets.

Usage:
    python -m src.evaluation.diversity
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR   = ROOT / "data" / "processed"
SYNTHETIC_LOG   = ROOT / "results" / "augmentation" / "synthetic_examples.json"
RESULTS_DIR     = ROOT / "results" / "evaluation"

AUGMENTED_CLASSES = ["Forced Action", "Sneaking", "Obstruction"]


# ---------------------------------------------------------------------------
# Tokenisation (simple whitespace + punctuation split, no stopword removal)
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def bigrams(tokens: list[str]) -> list[tuple[str, str]]:
    return list(zip(tokens, tokens[1:]))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def distinct_n(texts: list[str], n: int) -> float:
    """Distinct-n: unique n-grams / total n-grams across all texts."""
    all_ngrams: list[tuple] = []
    for text in texts:
        toks = tokenize(text)
        if n == 1:
            all_ngrams.extend([(t,) for t in toks])
        else:
            all_ngrams.extend(zip(*[toks[i:] for i in range(n)]))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def ttr(texts: list[str]) -> float:
    all_tokens = [t for text in texts for t in tokenize(text)]
    if not all_tokens:
        return 0.0
    return len(set(all_tokens)) / len(all_tokens)


def avg_length(texts: list[str]) -> float:
    if not texts:
        return 0.0
    return sum(len(tokenize(t)) for t in texts) / len(texts)


def vocab_size(texts: list[str]) -> int:
    return len({t for text in texts for t in tokenize(text)})


def pairwise_bigram_overlap(texts: list[str]) -> float:
    """
    Mean pairwise Jaccard similarity of bigram sets across all pairs.
    High value → examples share many bigrams → low diversity.
    """
    if len(texts) < 2:
        return 0.0
    bg_sets = [set(bigrams(tokenize(t))) for t in texts]
    scores = []
    for a, b in combinations(bg_sets, 2):
        if not a or not b:
            continue
        jaccard = len(a & b) / len(a | b)
        scores.append(jaccard)
    return sum(scores) / len(scores) if scores else 0.0


def ngram_overlap_cross(source: list[str], target: list[str], n: int = 2) -> float:
    """
    Fraction of n-grams in `target` that also appear in `source`.
    Measures how much synthetic text reuses real text patterns.
    """
    source_ngrams: set[tuple] = set()
    for text in source:
        toks = tokenize(text)
        source_ngrams.update(zip(*[toks[i:] for i in range(n)]))

    target_ngrams: list[tuple] = []
    for text in target:
        toks = tokenize(text)
        target_ngrams.extend(zip(*[toks[i:] for i in range(n)]))

    if not target_ngrams:
        return 0.0
    hits = sum(1 for ng in target_ngrams if ng in source_ngrams)
    return hits / len(target_ngrams)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_real_examples(category: str) -> list[str]:
    """Load real examples for a category from all original splits."""
    texts = []
    for split in ("train", "val", "test"):
        path = PROCESSED_DIR / f"{split}.csv"
        if not path.exists():
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                if row["category"] == category:
                    texts.append(row["text"])
    return texts


def load_synthetic_examples(category: str) -> list[str]:
    # Primary: JSON augmentation log
    with open(SYNTHETIC_LOG) as f:
        log = json.load(f)
    texts = log.get(category, [])
    if texts:
        return texts
    # Fallback: load from train_v2.csv (synthetic == True rows)
    path = PROCESSED_DIR / "train_v2.csv"
    if not path.exists():
        return []
    result = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["category"] == category and row.get("synthetic", "").lower() in ("true", "1"):
                result.append(row["text"])
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def analyse_category(category: str) -> dict:
    real = load_real_examples(category)
    synth = load_synthetic_examples(category)

    print(f"\n{'='*60}")
    print(f"  {category}  (real={len(real)}, synthetic={len(synth)})")
    print(f"{'='*60}")

    def report(label: str, texts: list[str]) -> dict:
        if not texts:
            print(f"  {label}: no examples")
            return {}
        d = {
            "n":          len(texts),
            "avg_len":    round(avg_length(texts), 1),
            "vocab_size": vocab_size(texts),
            "ttr":        round(ttr(texts), 4),
            "distinct_1": round(distinct_n(texts, 1), 4),
            "distinct_2": round(distinct_n(texts, 2), 4),
            "pairwise_bigram_overlap": round(pairwise_bigram_overlap(texts), 4),
        }
        print(f"\n  [{label}] n={d['n']}")
        print(f"    Avg length (tokens)    : {d['avg_len']}")
        print(f"    Vocabulary size        : {d['vocab_size']}")
        print(f"    Type-Token Ratio (TTR) : {d['ttr']:.4f}")
        print(f"    Distinct-1             : {d['distinct_1']:.4f}")
        print(f"    Distinct-2             : {d['distinct_2']:.4f}")
        print(f"    Pairwise bigram overlap: {d['pairwise_bigram_overlap']:.4f}  "
              f"{'⚠ high (low diversity)' if d['pairwise_bigram_overlap'] > 0.15 else '✓ low'}")
        return d

    real_stats  = report("real", real)
    synth_stats = report("synthetic", synth)

    # Cross-set overlap: how much of synthetic reuses real bigrams
    if real and synth:
        cross = ngram_overlap_cross(real, synth, n=2)
        print(f"\n  [cross] bigram overlap (real→synthetic): {cross:.4f}  "
              f"{'⚠ high leakage' if cross > 0.40 else '✓ acceptable'}")

        # Compare Distinct-2 directly
        d2_real  = distinct_n(real, 2)
        d2_synth = distinct_n(synth, 2)
        ratio = d2_synth / d2_real if d2_real > 0 else float("inf")
        print(f"  [diversity ratio] synth Distinct-2 / real Distinct-2: {ratio:.3f}  "
              f"{'⚠ synthetic less diverse' if ratio < 0.7 else '✓ comparable diversity'}")
    else:
        cross = None

    return {
        "category": category,
        "real":      real_stats,
        "synthetic": synth_stats,
        "cross_bigram_overlap": round(cross, 4) if cross is not None else None,
    }


def main() -> None:
    print("N-gram Diversity Analysis: Real vs Synthetic Examples")
    print("=" * 60)

    results = []
    for cat in AUGMENTED_CLASSES:
        r = analyse_category(cat)
        results.append(r)

    # Summary table
    print("\n\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"\n{'Category':<18} {'':6} {'TTR':>7} {'Dist-1':>8} {'Dist-2':>8} {'PairOvlp':>10}")
    print("-" * 62)
    for r in results:
        cat = r["category"]
        for label, stats in [("real", r["real"]), ("synth", r["synthetic"])]:
            if not stats:
                continue
            print(f"{'  '+cat if label=='real' else '':18} {label:>6} "
                  f"{stats['ttr']:>7.4f} {stats['distinct_1']:>8.4f} "
                  f"{stats['distinct_2']:>8.4f} {stats['pairwise_bigram_overlap']:>10.4f}")
        cross = r.get("cross_bigram_overlap")
        if cross is not None:
            flag = "⚠" if cross > 0.40 else "✓"
            print(f"  {'cross bigram overlap':>28}: {cross:.4f} {flag}")
        print()

    print("\nInterpretation guide:")
    print("  Distinct-2 < 0.5 of real  → synthetic is repetitive (overfitting risk)")
    print("  Pairwise overlap > 0.15   → examples share many bigrams within set")
    print("  Cross overlap > 0.40      → synthetic heavily reuses real phrasing")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "diversity_analysis.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
