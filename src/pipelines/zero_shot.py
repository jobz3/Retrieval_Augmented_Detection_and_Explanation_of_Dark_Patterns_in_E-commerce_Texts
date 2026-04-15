"""
Zero-shot explanation pipeline.

Sends a single user message with the product text (no examples) and returns
a structured PredictionResult plus grounding metadata.

Usage:
    python -m src.pipelines.zero_shot --text "Only 2 left in stock!"
    python -m src.pipelines.zero_shot --file data/processed/test_v2.csv --out results/zero_shot.jsonl
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import typer

from src.pipelines.output_parser import parse_prediction, ParseError
from src.pipelines.prompts import SYSTEM_PROMPT, format_zero_shot_prompt
from src.pipelines.schema import PredictionResult
from src.pipelines.span_grounding import check_grounding
from src.utils.io import save_jsonl
from src.utils.ollama_client import chat_json, DEFAULT_MODEL

ROOT = Path(__file__).resolve().parents[2]
app = typer.Typer()


# ---------------------------------------------------------------------------
# Core predict function
# ---------------------------------------------------------------------------

def predict(
    text: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
) -> tuple[PredictionResult, dict]:
    """
    Run zero-shot prediction on a single product text.

    Returns:
        (result, grounding_info)
        where grounding_info has keys: exact, case_insensitive, span, span_length
    """
    prompt = format_zero_shot_prompt(text)
    raw = chat_json(prompt, model=model, system=SYSTEM_PROMPT, temperature=temperature)
    result = parse_prediction(raw, text)
    grounding = check_grounding(result, text)
    return result, grounding


def predict_batch(
    texts: list[str],
    gold_labels: list[str] | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    verbose: bool = True,
) -> list[dict]:
    """
    Run zero-shot prediction on a list of texts.

    Returns:
        List of dicts with all PredictionResult fields plus grounding info,
        input_text, and (if provided) gold_label.
    """
    records = []
    n = len(texts)
    for i, text in enumerate(texts):
        if verbose:
            print(f"  [{i+1}/{n}] predicting ...", end="\r", flush=True)
        try:
            result, grounding = predict(text, model=model, temperature=temperature)
            parse_ok = True
        except (ParseError, ValueError) as exc:
            if verbose:
                print(f"\n  ⚠ parse error at index {i}: {exc}")
            from src.pipelines.schema import PatternType
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
            parse_ok = False

        rec = result.to_dict()
        rec["input_text"]  = text
        rec["span_exact"]  = grounding["exact"]
        rec["span_ci"]     = grounding["case_insensitive"]
        rec["span_length"] = grounding["span_length"]
        rec["parse_ok"]    = parse_ok
        if gold_labels is not None:
            rec["gold_label"] = gold_labels[i]
        records.append(rec)

    if verbose:
        print()  # newline after progress
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    text: str  = typer.Option(None,  help="Single product text to analyse"),
    file: Path = typer.Option(None,  help="CSV file with 'text' column (e.g. test_v2.csv)"),
    out:  Path = typer.Option(None,  help="Output JSONL path (default: results/pipelines/zero_shot.jsonl)"),
    model: str = typer.Option(DEFAULT_MODEL, help="Ollama model tag"),
    temperature: float = typer.Option(0.0, help="Sampling temperature"),
) -> None:
    if text:
        result, grounding = predict(text, model=model, temperature=temperature)
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        print(f"\nSpan grounded (exact): {grounding['exact']}")
        return

    if file is None:
        file = ROOT / "data" / "processed" / "test_v2.csv"

    with open(file) as f:
        rows = list(csv.DictReader(f))

    texts      = [r["text"] for r in rows]
    gold_labels = [r.get("category", "") for r in rows]

    print(f"Zero-shot inference on {len(texts)} examples (model={model}) ...")
    records = predict_batch(texts, gold_labels=gold_labels, model=model, temperature=temperature)

    # Summary stats
    exact_rate = sum(r["span_exact"] for r in records) / len(records)
    parse_rate = sum(r["parse_ok"]   for r in records) / len(records)
    print(f"Parse success rate : {parse_rate:.1%}")
    print(f"Span grounding rate: {exact_rate:.1%} (exact)")

    if out is None:
        out = ROOT / "results" / "pipelines" / "zero_shot.jsonl"
    save_jsonl(records, out)


if __name__ == "__main__":
    app()
