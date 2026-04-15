"""
Random few-shot explanation pipeline.

Samples k random examples from the training split and prepends them as
in-context demonstrations before the target text.

Usage:
    python -m src.pipelines.random_few_shot --text "Only 2 left!" --k 5
    python -m src.pipelines.random_few_shot --file data/processed/test_v2.csv --k 5 --out results/pipelines/random_few_shot_k5.jsonl
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import typer

from src.pipelines.output_parser import parse_prediction, ParseError
from src.pipelines.prompts import SYSTEM_PROMPT, format_few_shot_prompt
from src.pipelines.schema import PatternType, PredictionResult
from src.pipelines.span_grounding import check_grounding
from src.utils.io import save_jsonl
from src.utils.ollama_client import chat_json, DEFAULT_MODEL

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

app = typer.Typer()


# ---------------------------------------------------------------------------
# Example pool
# ---------------------------------------------------------------------------

def _load_train_examples(train_stem: str = "train_v2") -> list[dict]:
    """Load all real (non-synthetic) examples from the training split."""
    path = PROCESSED_DIR / f"{train_stem}.csv"
    with open(path) as f:
        rows = list(csv.DictReader(f))
    # Prefer real examples; fall back to all if no 'synthetic' column
    if "synthetic" in rows[0]:
        rows = [r for r in rows if r["synthetic"].lower() in ("false", "0", "")]
    return rows


def _sample_examples(
    k: int,
    train_rows: list[dict],
    exclude_text: str | None = None,
    seed: int | None = None,
) -> list[dict]:
    """
    Sample k examples from train_rows.

    If exclude_text is given, removes it from the pool first (avoids leaking
    the query into its own context when the query is from the training set).
    """
    pool = [r for r in train_rows if r["text"] != exclude_text]
    rng = random.Random(seed)
    n = min(k, len(pool))
    chosen = rng.sample(pool, n)
    return [{"text": r["text"], "category": r["category"]} for r in chosen]


# ---------------------------------------------------------------------------
# Core predict function
# ---------------------------------------------------------------------------

def predict(
    text: str,
    k: int = 5,
    train_rows: list[dict] | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    seed: int | None = None,
) -> tuple[PredictionResult, dict, list[dict]]:
    """
    Run random few-shot prediction on a single product text.

    Args:
        text:       Target product text.
        k:          Number of few-shot examples.
        train_rows: Pre-loaded training rows (loaded once for batch efficiency).
        model:      Ollama model tag.
        temperature: Sampling temperature.
        seed:       Random seed for reproducible example selection.

    Returns:
        (result, grounding_info, examples_used)
    """
    if train_rows is None:
        train_rows = _load_train_examples()

    examples = _sample_examples(k, train_rows, exclude_text=text, seed=seed)
    prompt   = format_few_shot_prompt(text, examples)
    raw      = chat_json(prompt, model=model, system=SYSTEM_PROMPT, temperature=temperature)
    result   = parse_prediction(raw, text)
    grounding = check_grounding(result, text)
    return result, grounding, examples


def predict_batch(
    texts: list[str],
    gold_labels: list[str] | None = None,
    k: int = 5,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    seed: int | None = None,
    train_stem: str = "train_v2",
    verbose: bool = True,
) -> list[dict]:
    """
    Run random few-shot prediction on a list of texts.

    Returns list of dicts with all result fields + grounding + metadata.
    """
    train_rows = _load_train_examples(train_stem)
    records = []
    n = len(texts)

    for i, text in enumerate(texts):
        if verbose:
            print(f"  [{i+1}/{n}] predicting ...", end="\r", flush=True)

        example_seed = None if seed is None else seed + i
        try:
            result, grounding, examples = predict(
                text, k=k, train_rows=train_rows, model=model,
                temperature=temperature, seed=example_seed,
            )
            parse_ok = True
        except (ParseError, ValueError) as exc:
            if verbose:
                print(f"\n  ⚠ parse error at index {i}: {exc}")
            result = PredictionResult(
                label=PatternType.UNCERTAIN,
                confidence=0.0,
                psychological_mechanism="parse error",
                evidence_span="",
                harm_dimension="unknown",
                rationale="Output could not be parsed.",
                rewrite=text,
            )
            grounding = {"exact": False, "case_insensitive": False, "span": "", "span_length": 0}
            examples  = []
            parse_ok  = False

        rec = result.to_dict()
        rec["input_text"]       = text
        rec["span_exact"]       = grounding["exact"]
        rec["span_ci"]          = grounding["case_insensitive"]
        rec["span_length"]      = grounding["span_length"]
        rec["parse_ok"]         = parse_ok
        rec["k"]                = k
        rec["examples_used"]    = [e["category"] for e in examples]
        if gold_labels is not None:
            rec["gold_label"] = gold_labels[i]
        records.append(rec)

    if verbose:
        print()
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    text:  str  = typer.Option(None, help="Single product text to analyse"),
    file:  Path = typer.Option(None, help="CSV file with 'text' column"),
    out:   Path = typer.Option(None, help="Output JSONL path"),
    k:     int  = typer.Option(5,    help="Number of few-shot examples"),
    model: str  = typer.Option(DEFAULT_MODEL, help="Ollama model tag"),
    temperature: float = typer.Option(0.0, help="Sampling temperature"),
    seed:  int  = typer.Option(42,   help="Random seed"),
    train_stem: str = typer.Option("train_v2", help="Training split stem"),
) -> None:
    if text:
        train_rows = _load_train_examples(train_stem)
        result, grounding, examples = predict(
            text, k=k, train_rows=train_rows, model=model,
            temperature=temperature, seed=seed,
        )
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        print(f"\nSpan grounded (exact): {grounding['exact']}")
        print(f"Examples used: {[e['category'] for e in examples]}")
        return

    if file is None:
        file = ROOT / "data" / "processed" / "test_v2.csv"

    with open(file) as f:
        rows = list(csv.DictReader(f))

    texts       = [r["text"] for r in rows]
    gold_labels = [r.get("category", "") for r in rows]

    print(f"Random few-shot (k={k}) on {len(texts)} examples (model={model}) ...")
    records = predict_batch(
        texts, gold_labels=gold_labels, k=k, model=model,
        temperature=temperature, seed=seed, train_stem=train_stem,
    )

    exact_rate = sum(r["span_exact"] for r in records) / len(records)
    parse_rate = sum(r["parse_ok"]   for r in records) / len(records)
    print(f"Parse success rate : {parse_rate:.1%}")
    print(f"Span grounding rate: {exact_rate:.1%} (exact)")

    if out is None:
        out = ROOT / "results" / "pipelines" / f"random_few_shot_k{k}.jsonl"
    save_jsonl(records, out)


if __name__ == "__main__":
    app()
