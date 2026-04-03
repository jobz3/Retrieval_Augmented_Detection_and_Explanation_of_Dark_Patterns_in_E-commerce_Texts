"""
RAG few-shot explanation pipeline.

Uses a pre-built FAISS index to retrieve the k most semantically similar
training examples, then uses them as in-context demonstrations.

Three retrieval strategies are supported (passed to the Retriever):
  knn       — top-k by cosine similarity
  prototype — prefer class-prototypical examples
  diversity — MMR: balance similarity and diversity

Usage:
    python -m src.pipelines.rag_few_shot --text "Only 2 left!" --k 5
    python -m src.pipelines.rag_few_shot --file data/processed/test_v2.csv --k 5 --strategy knn
    python -m src.pipelines.rag_few_shot --file data/processed/test_v2.csv --k 5 --strategy diversity --encoder sbert
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import typer

from src.pipelines.output_parser import parse_prediction, ParseError
from src.pipelines.prompts import SYSTEM_PROMPT, format_few_shot_prompt
from src.pipelines.schema import PatternType, PredictionResult
from src.pipelines.span_grounding import check_grounding
from src.retrieval.retrieve import Retriever
from src.utils.io import save_jsonl
from src.utils.ollama_client import chat_json, DEFAULT_MODEL

ROOT = Path(__file__).resolve().parents[2]
app = typer.Typer()


# ---------------------------------------------------------------------------
# Core predict function
# ---------------------------------------------------------------------------

def predict(
    text: str,
    retriever: Retriever,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
) -> tuple[PredictionResult, dict, list[dict]]:
    """
    Run RAG few-shot prediction on a single product text.

    Args:
        text:        Target product text.
        retriever:   Pre-loaded Retriever instance (encoder + strategy + k baked in).
        model:       Ollama model tag.
        temperature: Sampling temperature.

    Returns:
        (result, grounding_info, retrieved_examples)
        where retrieved_examples is the raw list of dicts from Retriever.retrieve()
        (each has "text", "category", "score").
    """
    retrieved = retriever.retrieve(text)

    # Convert retrieval results to the format expected by format_few_shot_prompt
    examples = [{"text": r["text"], "category": r["category"]} for r in retrieved]

    prompt   = format_few_shot_prompt(text, examples)
    raw      = chat_json(prompt, model=model, system=SYSTEM_PROMPT, temperature=temperature)
    result   = parse_prediction(raw, text)
    grounding = check_grounding(result, text)
    return result, grounding, retrieved


def predict_batch(
    texts: list[str],
    retriever: Retriever,
    gold_labels: list[str] | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    verbose: bool = True,
) -> list[dict]:
    """
    Run RAG few-shot prediction on a list of texts.

    Returns list of dicts with all result fields + grounding + retrieval metadata.
    """
    records = []
    n = len(texts)

    for i, text in enumerate(texts):
        if verbose:
            print(f"  [{i+1}/{n}] predicting ...", end="\r", flush=True)

        try:
            result, grounding, retrieved = predict(
                text, retriever=retriever, model=model, temperature=temperature,
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
            retrieved = []
            parse_ok  = False

        rec = result.to_dict()
        rec["input_text"]       = text
        rec["span_exact"]       = grounding["exact"]
        rec["span_ci"]          = grounding["case_insensitive"]
        rec["span_length"]      = grounding["span_length"]
        rec["parse_ok"]         = parse_ok
        rec["k"]                = retriever.k
        rec["encoder"]          = retriever.encoder
        rec["strategy"]         = retriever.strategy
        rec["retrieved_labels"] = [r["category"] for r in retrieved]
        rec["retrieved_scores"] = [round(r["score"], 4) for r in retrieved]
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
    text:     str  = typer.Option(None,  help="Single product text to analyse"),
    file:     Path = typer.Option(None,  help="CSV file with 'text' column"),
    out:      Path = typer.Option(None,  help="Output JSONL path"),
    k:        int  = typer.Option(5,     help="Number of retrieved examples"),
    encoder:  str  = typer.Option("sbert", help="Retrieval encoder: sbert | bert"),
    strategy: str  = typer.Option("knn",   help="Retrieval strategy: knn | prototype | diversity"),
    model:    str  = typer.Option(DEFAULT_MODEL, help="Ollama model tag"),
    temperature: float = typer.Option(0.0, help="Sampling temperature"),
) -> None:
    retriever = Retriever(encoder=encoder, strategy=strategy, k=k)

    if text:
        result, grounding, retrieved = predict(
            text, retriever=retriever, model=model, temperature=temperature,
        )
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        print(f"\nSpan grounded (exact): {grounding['exact']}")
        print(f"Retrieved labels: {[r['category'] for r in retrieved]}")
        return

    if file is None:
        file = ROOT / "data" / "processed" / "test_v2.csv"

    with open(file) as f:
        rows = list(csv.DictReader(f))

    texts       = [r["text"] for r in rows]
    gold_labels = [r.get("category", "") for r in rows]

    print(f"RAG few-shot (k={k}, encoder={encoder}, strategy={strategy}) on {len(texts)} examples ...")
    records = predict_batch(
        texts, retriever=retriever, gold_labels=gold_labels,
        model=model, temperature=temperature,
    )

    exact_rate = sum(r["span_exact"] for r in records) / len(records)
    parse_rate = sum(r["parse_ok"]   for r in records) / len(records)
    print(f"Parse success rate : {parse_rate:.1%}")
    print(f"Span grounding rate: {exact_rate:.1%} (exact)")

    if out is None:
        out = ROOT / "results" / "pipelines" / f"rag_{encoder}_{strategy}_k{k}.jsonl"
    save_jsonl(records, out)


if __name__ == "__main__":
    app()
