"""
Phase 2 Step C: retrieval-augmented explanation pipeline using locked-train indices.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from pydantic import ValidationError

from src.pipelines.prompts import (
    SHARED_SYSTEM_PROMPT,
    build_retrieved_examples_block,
    build_user_prompt,
)
from src.pipelines.schema import PredictionResult
from src.pipelines.zero_shot import (
    DEFAULT_INPUT_FILE,
    label_in_known_set,
    load_rows,
    resolve_path,
    slugify_model_name,
    span_is_valid,
    summary_rates,
    to_repo_relative,
)
from src.retrieval.retrieve import Retriever
from src.utils.config import load_config
from src.utils.ollama_client import chat_json


CONFIG = load_config()
ROOT = Path(__file__).resolve().parents[2]
LOCKED_TRAIN_FILE = CONFIG.paths.locked_processed_data / "train.csv"
INDEX_DIR = CONFIG.paths.indices
PREDICTIONS_DIR = ROOT / "results" / "predictions" / "rag"
ANALYSIS_DIR = ROOT / "results" / "analysis" / "phase2_rag_sanity"

app = typer.Typer()


def load_tuple_set(csv_path: Path) -> set[tuple[str, str]]:
    df = pd.read_csv(csv_path)
    return set(zip(df["text"], df["category"]))


def verify_retrieval_source(encoder: str, index_dir: Path, locked_train_file: Path) -> dict[str, Any]:
    metadata_path = index_dir / f"{encoder}_metadata.json"
    index_path = index_dir / f"{encoder}_index.faiss"
    embeddings_path = index_dir / f"{encoder}_embeddings.npy"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Retrieval metadata not found: {metadata_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"Retrieval index not found: {index_path}")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Retrieval embedding matrix not found: {embeddings_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    metadata_tuples = {(item["text"], item["category"]) for item in metadata}
    train_tuples = load_tuple_set(locked_train_file)
    val_tuples = load_tuple_set(CONFIG.paths.locked_processed_data / "val.csv")
    test_tuples = load_tuple_set(CONFIG.paths.locked_processed_data / "test.csv")

    return {
        "encoder": encoder,
        "index_path": to_repo_relative(index_path),
        "metadata_path": to_repo_relative(metadata_path),
        "embeddings_path": to_repo_relative(embeddings_path),
        "retrieval_source_train_only": metadata_tuples == train_tuples,
        "val_overlap_count": len(metadata_tuples & val_tuples),
        "test_overlap_count": len(metadata_tuples & test_tuples),
        "metadata_size": len(metadata),
    }


@app.command()
def main(
    input_file: Path = typer.Option(DEFAULT_INPUT_FILE, help="CSV split file to analyze"),
    model: str = typer.Option(CONFIG.ollama.default_model, help="Local Ollama model tag"),
    encoder: str = typer.Option(CONFIG.retrieval.encoder, help="Retrieval encoder to use"),
    top_k: int = typer.Option(3, min=1, help="Number of retrieved examples"),
    strategy: str = typer.Option(CONFIG.retrieval.strategy, help="Retrieval strategy"),
    limit: int = typer.Option(5, min=1, help="Number of rows to process for sanity checking"),
    temperature: float = typer.Option(CONFIG.ollama.temperature, help="Sampling temperature"),
) -> None:
    input_path = resolve_path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input split file not found: {input_path}")

    rows = load_rows(input_path, limit)
    if not rows:
        raise ValueError(f"No rows found in input split file: {input_path}")

    retrieval_info = verify_retrieval_source(
        encoder=encoder,
        index_dir=INDEX_DIR,
        locked_train_file=LOCKED_TRAIN_FILE,
    )
    if not retrieval_info["retrieval_source_train_only"]:
        raise ValueError(
            "Retrieval metadata does not match the locked training split exactly. "
            "Refusing to run RAG with a mixed retrieval source."
        )

    retriever = Retriever(
        encoder=encoder,
        strategy=strategy,
        k=top_k,
        diversity_lambda=CONFIG.retrieval.diversity_lambda,
        index_dir=INDEX_DIR,
    )

    model_slug = slugify_model_name(model)
    run_name = (
        f"rag_{input_path.stem}_limit{limit}_{encoder}_{strategy}_k{top_k}_{model_slug}"
    )
    predictions_path = PREDICTIONS_DIR / f"{run_name}.jsonl"
    summary_path = ANALYSIS_DIR / f"{run_name}_summary.json"
    representatives_path = ANALYSIS_DIR / f"{run_name}_representative_outputs.json"
    traces_path = ANALYSIS_DIR / f"{run_name}_retrieval_traces.json"

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    representative_outputs: list[dict[str, Any]] = []
    retrieval_traces: list[dict[str, Any]] = []

    for row in rows:
        input_text = row["text"]
        retrieved_examples = retriever.retrieve(input_text)
        examples_block = build_retrieved_examples_block(retrieved_examples)
        prompt = build_user_prompt(input_text, examples_block=examples_block)

        record: dict[str, Any] = {
            "page_id": row.get("page_id"),
            "gold_label": row.get("category"),
            "input_text": input_text,
            "variant": "rag",
            "model": model,
            "encoder": encoder,
            "strategy": strategy,
            "top_k": top_k,
            "json_valid": False,
            "schema_valid": False,
            "label_in_known_set": False,
            "evidence_span_valid": False,
            "retrieved_examples": retrieved_examples,
        }

        try:
            raw_output = chat_json(
                prompt=prompt,
                model=model,
                system=SHARED_SYSTEM_PROMPT,
                temperature=temperature,
            )
            record["raw_output"] = raw_output
            record["json_valid"] = True
            record["label_in_known_set"] = label_in_known_set(raw_output)

            try:
                prediction = PredictionResult.model_validate(raw_output)
                record["schema_valid"] = True
                record["prediction"] = prediction.to_dict()
                record["evidence_span_valid"] = span_is_valid(prediction, input_text)
            except ValidationError as exc:
                record["validation_error"] = str(exc)
        except Exception as exc:  # noqa: BLE001
            record["error"] = str(exc)

        records.append(record)

        trace = {
            "page_id": row.get("page_id"),
            "gold_label": row.get("category"),
            "input_text": input_text,
            "encoder": encoder,
            "strategy": strategy,
            "top_k": top_k,
            "retrieved_examples": retrieved_examples,
        }
        retrieval_traces.append(trace)

        if len(representative_outputs) < 3:
            representative_outputs.append(
                {
                    "page_id": record["page_id"],
                    "gold_label": record["gold_label"],
                    "prediction": record.get("prediction"),
                    "evidence_span_valid": record["evidence_span_valid"],
                    "retrieved_examples": retrieved_examples,
                }
            )

    with predictions_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    json_valid_count = sum(1 for record in records if record["json_valid"])
    schema_pass_count = sum(1 for record in records if record["schema_valid"])
    label_pass_count = sum(1 for record in records if record["label_in_known_set"])
    span_pass_count = sum(1 for record in records if record["evidence_span_valid"])

    summary = {
        "variant": "rag",
        "model": model,
        "input_file": to_repo_relative(input_path),
        "processed_items": len(records),
        "encoder": encoder,
        "strategy": strategy,
        "top_k": top_k,
        "json_valid_count": json_valid_count,
        "json_invalid_count": len(records) - json_valid_count,
        "json_valid_rate": summary_rates(json_valid_count, len(records)),
        "schema_pass_count": schema_pass_count,
        "schema_fail_count": len(records) - schema_pass_count,
        "schema_pass_rate": summary_rates(schema_pass_count, len(records)),
        "label_in_known_set_pass_count": label_pass_count,
        "label_in_known_set_fail_count": len(records) - label_pass_count,
        "label_in_known_set_pass_rate": summary_rates(label_pass_count, len(records)),
        "evidence_span_pass_count": span_pass_count,
        "evidence_span_fail_count": len(records) - span_pass_count,
        "evidence_span_pass_rate": summary_rates(span_pass_count, len(records)),
        "retrieval_info": retrieval_info,
        "predictions_path": to_repo_relative(predictions_path),
        "representative_outputs_path": to_repo_relative(representatives_path),
        "retrieval_traces_path": to_repo_relative(traces_path),
    }

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with representatives_path.open("w", encoding="utf-8") as handle:
        json.dump(representative_outputs, handle, indent=2, ensure_ascii=False)
    with traces_path.open("w", encoding="utf-8") as handle:
        json.dump(retrieval_traces, handle, indent=2, ensure_ascii=False)

    print(f"RAG sanity run complete for {len(records)} items")
    print(f"JSON valid: {json_valid_count}/{len(records)}")
    print(f"Schema pass: {schema_pass_count}/{len(records)}")
    print(f"Label in known set: {label_pass_count}/{len(records)}")
    print(f"Evidence span grounded: {span_pass_count}/{len(records)}")
    print(f"Retrieval source verified as train-only: {retrieval_info['retrieval_source_train_only']}")
    print(f"Predictions saved to {to_repo_relative(predictions_path)}")
    print(f"Summary saved to {to_repo_relative(summary_path)}")


if __name__ == "__main__":
    app()
