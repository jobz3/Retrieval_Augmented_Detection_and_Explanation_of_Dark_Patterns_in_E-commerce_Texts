"""
Phase 2 Step B: random few-shot explanation pipeline using locked train examples.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import typer
from pydantic import ValidationError

from src.pipelines.prompts import (
    SHARED_SYSTEM_PROMPT,
    build_label_only_examples_block,
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
from src.utils.config import load_config
from src.utils.ollama_client import chat_json


CONFIG = load_config()
DEFAULT_EXAMPLE_FILE = CONFIG.paths.locked_processed_data / "train.csv"
ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_DIR = ROOT / "results" / "predictions" / "random_few_shot"
ANALYSIS_DIR = ROOT / "results" / "analysis" / "phase2_random_few_shot_sanity"

app = typer.Typer()


def sample_examples(
    example_rows: list[dict[str, str]],
    target_row: dict[str, str],
    num_examples: int,
    seed: int,
    row_index: int,
) -> list[dict[str, str]]:
    target_page_id = target_row.get("page_id")
    candidates = [
        row for row in example_rows
        if not target_page_id or row.get("page_id") != target_page_id
    ]

    if len(candidates) < num_examples:
        raise ValueError(
            f"Not enough candidate examples to sample {num_examples} items from the locked train split."
        )

    row_seed = f"{seed}:{target_page_id or row_index}"
    rng = random.Random(row_seed)
    return rng.sample(candidates, num_examples)


@app.command()
def main(
    input_file: Path = typer.Option(DEFAULT_INPUT_FILE, help="CSV split file to analyze"),
    example_file: Path = typer.Option(
        DEFAULT_EXAMPLE_FILE,
        help="CSV file used as the random few-shot example pool",
    ),
    model: str = typer.Option(CONFIG.ollama.default_model, help="Local Ollama model tag"),
    limit: int = typer.Option(5, min=1, help="Number of rows to process for sanity checking"),
    num_examples: int = typer.Option(3, min=1, help="Number of random examples per prompt"),
    seed: int = typer.Option(CONFIG.project.seed, help="Deterministic sampling seed"),
    temperature: float = typer.Option(CONFIG.ollama.temperature, help="Sampling temperature"),
) -> None:
    input_path = resolve_path(input_file)
    example_path = resolve_path(example_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input split file not found: {input_path}")
    if not example_path.exists():
        raise FileNotFoundError(f"Example split file not found: {example_path}")

    rows = load_rows(input_path, limit)
    example_rows = load_rows(example_path, None)
    if not rows:
        raise ValueError(f"No rows found in input split file: {input_path}")
    if not example_rows:
        raise ValueError(f"No rows found in example split file: {example_path}")

    model_slug = slugify_model_name(model)
    run_name = (
        f"random_few_shot_{input_path.stem}_limit{limit}_k{num_examples}_seed{seed}_{model_slug}"
    )
    predictions_path = PREDICTIONS_DIR / f"{run_name}.jsonl"
    summary_path = ANALYSIS_DIR / f"{run_name}_summary.json"
    representatives_path = ANALYSIS_DIR / f"{run_name}_representative_outputs.json"

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    representative_outputs: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows):
        input_text = row["text"]
        sampled_examples = sample_examples(example_rows, row, num_examples, seed, row_index)
        examples_block = build_label_only_examples_block(sampled_examples)
        prompt = build_user_prompt(input_text, examples_block=examples_block)

        record: dict[str, Any] = {
            "page_id": row.get("page_id"),
            "gold_label": row.get("category"),
            "input_text": input_text,
            "variant": "random_few_shot",
            "model": model,
            "example_source_file": to_repo_relative(example_path),
            "num_examples": num_examples,
            "seed": seed,
            "sampled_examples": [
                {
                    "page_id": example.get("page_id"),
                    "category": example.get("category"),
                    "text": example.get("text"),
                }
                for example in sampled_examples
            ],
            "json_valid": False,
            "schema_valid": False,
            "label_in_known_set": False,
            "evidence_span_valid": False,
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

        if record["schema_valid"] and len(representative_outputs) < 3:
            representative_outputs.append(
                {
                    "page_id": record["page_id"],
                    "gold_label": record["gold_label"],
                    "prediction": record["prediction"],
                    "evidence_span_valid": record["evidence_span_valid"],
                    "sampled_examples": record["sampled_examples"],
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
        "variant": "random_few_shot",
        "model": model,
        "input_file": to_repo_relative(input_path),
        "example_source_file": to_repo_relative(example_path),
        "processed_items": len(records),
        "num_examples": num_examples,
        "seed": seed,
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
        "predictions_path": to_repo_relative(predictions_path),
        "representative_outputs_path": to_repo_relative(representatives_path),
    }

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with representatives_path.open("w", encoding="utf-8") as handle:
        json.dump(representative_outputs, handle, indent=2, ensure_ascii=False)

    print(f"Random few-shot sanity run complete for {len(records)} items")
    print(f"JSON valid: {json_valid_count}/{len(records)}")
    print(f"Schema pass: {schema_pass_count}/{len(records)}")
    print(f"Label in known set: {label_pass_count}/{len(records)}")
    print(f"Evidence span grounded: {span_pass_count}/{len(records)}")
    print(f"Predictions saved to {to_repo_relative(predictions_path)}")
    print(f"Summary saved to {to_repo_relative(summary_path)}")


if __name__ == "__main__":
    app()
