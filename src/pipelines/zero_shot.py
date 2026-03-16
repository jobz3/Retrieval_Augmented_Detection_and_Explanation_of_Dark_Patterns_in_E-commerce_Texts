"""
Phase 2 Step A: zero-shot explanation pipeline using the shared prompt layer.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import typer
from pydantic import ValidationError

from src.pipelines.prompts import KNOWN_LABELS, SHARED_SYSTEM_PROMPT, build_user_prompt
from src.pipelines.schema import PredictionResult
from src.utils.config import load_config
from src.utils.ollama_client import chat_json


CONFIG = load_config()
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_FILE = CONFIG.paths.locked_processed_data / "val.csv"
PREDICTIONS_DIR = ROOT / "results" / "predictions" / "zero_shot"
ANALYSIS_DIR = ROOT / "results" / "analysis" / "phase2_zero_shot_sanity"

app = typer.Typer()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def to_repo_relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def slugify_model_name(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def load_rows(input_file: Path, limit: int | None) -> list[dict[str, str]]:
    with input_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return rows if limit is None else rows[:limit]


def label_in_known_set(raw_output: dict[str, Any]) -> bool:
    label = raw_output.get("label")
    return isinstance(label, str) and label in KNOWN_LABELS


def span_is_valid(prediction: PredictionResult, input_text: str) -> bool:
    span = prediction.evidence_span.strip()
    return bool(span) and span in input_text


def summary_rates(pass_count: int, total_count: int) -> float:
    if total_count == 0:
        return 0.0
    return round(pass_count / total_count, 4)


@app.command()
def main(
    input_file: Path = typer.Option(DEFAULT_INPUT_FILE, help="CSV split file to analyze"),
    model: str = typer.Option(CONFIG.ollama.default_model, help="Local Ollama model tag"),
    limit: int = typer.Option(5, min=1, help="Number of rows to process for sanity checking"),
    temperature: float = typer.Option(CONFIG.ollama.temperature, help="Sampling temperature"),
) -> None:
    input_path = resolve_path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input split file not found: {input_path}")

    rows = load_rows(input_path, limit)
    if not rows:
        raise ValueError(f"No rows found in input split file: {input_path}")

    model_slug = slugify_model_name(model)
    run_name = f"zero_shot_{input_path.stem}_limit{limit}_{model_slug}"
    predictions_path = PREDICTIONS_DIR / f"{run_name}.jsonl"
    summary_path = ANALYSIS_DIR / f"{run_name}_summary.json"
    representatives_path = ANALYSIS_DIR / f"{run_name}_representative_outputs.json"

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    representative_outputs: list[dict[str, Any]] = []

    for row in rows:
        input_text = row["text"]
        prompt = build_user_prompt(input_text)
        record: dict[str, Any] = {
            "page_id": row.get("page_id"),
            "gold_label": row.get("category"),
            "input_text": input_text,
            "variant": "zero_shot",
            "model": model,
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
        except Exception as exc:  # noqa: BLE001 - operational failures should be recorded per row
            record["error"] = str(exc)

        records.append(record)

        if record["schema_valid"] and len(representative_outputs) < 3:
            representative_outputs.append(
                {
                    "page_id": record["page_id"],
                    "gold_label": record["gold_label"],
                    "prediction": record["prediction"],
                    "evidence_span_valid": record["evidence_span_valid"],
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
        "variant": "zero_shot",
        "model": model,
        "input_file": to_repo_relative(input_path),
        "processed_items": len(records),
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

    print(f"Zero-shot sanity run complete for {len(records)} items")
    print(f"JSON valid: {json_valid_count}/{len(records)}")
    print(f"Schema pass: {schema_pass_count}/{len(records)}")
    print(f"Label in known set: {label_pass_count}/{len(records)}")
    print(f"Evidence span grounded: {span_pass_count}/{len(records)}")
    print(f"Predictions saved to {to_repo_relative(predictions_path)}")
    print(f"Summary saved to {to_repo_relative(summary_path)}")


if __name__ == "__main__":
    app()
