"""
Generate Phase 1c seed-robustness artifacts for plain vs weighted RoBERTa.

Usage:
    python -m src.evaluation.report_seed_robustness
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import typer
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.baselines.train_classifier import evaluate as evaluate_checkpoint
from src.data.dataset import DarkPatternDataset, load_label_map
from src.utils.config import load_config


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


CONFIG = load_config()
ROOT = Path(__file__).resolve().parents[2]
LOCKED_SPLIT_DIR = CONFIG.paths.locked_processed_data
ROBUSTNESS_MODEL_DIR = CONFIG.paths.seed_robustness_models
ROBUSTNESS_RESULTS_DIR = CONFIG.paths.seed_robustness_results
ROBUSTNESS_ANALYSIS_DIR = CONFIG.paths.phase1c_analysis
WEIGHTED_VARIANT = CONFIG.phase1b.weighted_variant
SEEDS = CONFIG.phase1c.seeds

app = typer.Typer()


@dataclass(frozen=True)
class VariantSpec:
    model_key: str
    variant: str
    seed: int
    checkpoint_name: str
    checkpoint_path: Path


def to_repo_relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def label_names_for_split(processed_dir: Path) -> list[str]:
    label_map = load_label_map(processed_dir)
    id2label = {idx: label for label, idx in label_map.items()}
    return [id2label[idx] for idx in range(len(id2label))]


def load_train_class_counts(processed_dir: Path, label_names: list[str]) -> dict[str, int]:
    train_df = pd.read_csv(processed_dir / "train.csv")
    return {
        label_name: int((train_df["label_id"] == label_id).sum())
        for label_id, label_name in enumerate(label_names)
    }


def variant_specs(model_key: str) -> list[VariantSpec]:
    specs: list[VariantSpec] = []
    for seed in SEEDS:
        specs.append(
            VariantSpec(
                model_key=model_key,
                variant="plain",
                seed=seed,
                checkpoint_name=f"{model_key}_plain_seed{seed}",
                checkpoint_path=ROBUSTNESS_MODEL_DIR / f"{model_key}_plain_seed{seed}",
            )
        )
        specs.append(
            VariantSpec(
                model_key=model_key,
                variant=WEIGHTED_VARIANT,
                seed=seed,
                checkpoint_name=f"{model_key}_{WEIGHTED_VARIANT}_seed{seed}",
                checkpoint_path=ROBUSTNESS_MODEL_DIR / f"{model_key}_{WEIGHTED_VARIANT}_seed{seed}",
            )
        )
    return specs


def evaluate_variant_split(
    spec: VariantSpec,
    split: str,
    label_names: list[str],
    device: str,
    batch_size: int,
) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray]:
    cfg = CONFIG.baselines.by_name(spec.model_key)
    tokenizer = AutoTokenizer.from_pretrained(spec.checkpoint_path)
    model_obj = AutoModelForSequenceClassification.from_pretrained(spec.checkpoint_path).to(device)

    dataset = DarkPatternDataset(
        split,
        tokenizer,
        cfg.max_length,
        processed_dir=LOCKED_SPLIT_DIR,
    )
    loader = DataLoader(dataset, batch_size=batch_size)
    _, preds, labels = evaluate_checkpoint(model_obj, loader, device)

    all_label_ids = list(range(len(label_names)))
    report = classification_report(
        labels,
        preds,
        labels=all_label_ids,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(labels, preds, labels=all_label_ids)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    if device == "cuda":
        torch.cuda.empty_cache()

    return report, per_class_rows(spec, split, report, label_names), cm_norm


def per_class_rows(
    spec: VariantSpec,
    split: str,
    report: dict[str, object],
    label_names: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label_name in label_names:
        metrics = report[label_name]
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "model": spec.model_key,
                "variant": spec.variant,
                "seed": spec.seed,
                "checkpoint_name": spec.checkpoint_name,
                "split": split,
                "class": label_name,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1-score"],
                "support": metrics["support"],
            }
        )
    return rows


def write_normalized_confusion_plot(
    matrix: np.ndarray,
    labels: list[str],
    output_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        vmin=0.0,
        vmax=1.0,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    separator = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for record in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in record) + " |")
    return "\n".join(lines) + "\n"


@app.command()
def main(
    model: str = typer.Option("roberta", help="Model family to evaluate"),
    batch_size: int = typer.Option(32, help="Evaluation batch size"),
) -> None:
    if model != "roberta":
        raise ValueError("Phase 1c seed robustness is currently scoped to RoBERTa only.")

    if not LOCKED_SPLIT_DIR.exists():
        raise FileNotFoundError(f"Locked split directory not found: {LOCKED_SPLIT_DIR}")

    specs = variant_specs(model)
    missing_checkpoints = [spec.checkpoint_path for spec in specs if not spec.checkpoint_path.exists()]
    if missing_checkpoints:
        formatted = ", ".join(to_repo_relative(path) for path in missing_checkpoints)
        raise FileNotFoundError(f"Missing checkpoints required for seed robustness: {formatted}")

    ROBUSTNESS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ROBUSTNESS_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    label_names = label_names_for_split(LOCKED_SPLIT_DIR)
    train_class_counts = load_train_class_counts(LOCKED_SPLIT_DIR, label_names)
    minority_labels = [
        label for label, count in train_class_counts.items()
        if count < CONFIG.phase1b.minority_train_threshold
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary_rows: list[dict[str, object]] = []
    per_class_rows_all: list[dict[str, object]] = []

    for spec in specs:
        print(f"Evaluating {spec.checkpoint_name} on locked split using {device}")

        split_reports: dict[str, dict[str, object]] = {}
        for split in ("val", "test"):
            report, class_rows, cm_norm = evaluate_variant_split(
                spec, split, label_names, device, batch_size
            )
            split_reports[split] = report
            per_class_rows_all.extend(class_rows)

            if split == "test":
                cm_df = pd.DataFrame(cm_norm, index=label_names, columns=label_names)
                cm_csv_path = ROBUSTNESS_ANALYSIS_DIR / f"confusion_matrix_{spec.checkpoint_name}_test_norm.csv"
                cm_png_path = ROBUSTNESS_ANALYSIS_DIR / f"confusion_matrix_{spec.checkpoint_name}_test_norm.png"
                cm_df.to_csv(cm_csv_path)
                write_normalized_confusion_plot(
                    cm_norm,
                    label_names,
                    cm_png_path,
                    f"{spec.checkpoint_name} normalized confusion matrix (test)",
                )

        val_report = split_reports["val"]
        test_report = split_reports["test"]
        summary_row = {
            "model": spec.model_key,
            "variant": spec.variant,
            "seed": spec.seed,
            "checkpoint_name": spec.checkpoint_name,
            "val_accuracy": val_report["accuracy"],
            "val_macro_f1": val_report["macro avg"]["f1-score"],  # type: ignore[index]
            "test_accuracy": test_report["accuracy"],
            "test_macro_f1": test_report["macro avg"]["f1-score"],  # type: ignore[index]
            "test_weighted_f1": test_report["weighted avg"]["f1-score"],  # type: ignore[index]
            "test_minority_avg_f1": float(
                np.mean([test_report[label]["f1-score"] for label in minority_labels])  # type: ignore[index]
            ),
            "Forced Action F1": test_report["Forced Action"]["f1-score"],  # type: ignore[index]
            "Sneaking F1": test_report["Sneaking"]["f1-score"],  # type: ignore[index]
        }
        summary_rows.append(summary_row)

        run_json = {
            "model": spec.model_key,
            "variant": spec.variant,
            "seed": spec.seed,
            "checkpoint_name": spec.checkpoint_name,
            "split_root": to_repo_relative(LOCKED_SPLIT_DIR),
            "train_class_counts": train_class_counts,
            "minority_labels": minority_labels,
            "val_report": val_report,
            "test_report": test_report,
            "test_minority_avg_f1": summary_row["test_minority_avg_f1"],
        }
        run_json_path = ROBUSTNESS_RESULTS_DIR / f"{spec.checkpoint_name}.json"
        with run_json_path.open("w", encoding="utf-8") as handle:
            json.dump(run_json, handle, indent=2)

    summary_df = pd.DataFrame(summary_rows)
    summary_df["variant"] = pd.Categorical(summary_df["variant"], ["plain", WEIGHTED_VARIANT], ordered=True)
    summary_df["seed"] = pd.Categorical(summary_df["seed"], SEEDS, ordered=True)
    summary_df = summary_df.sort_values(["variant", "seed"]).reset_index(drop=True)

    aggregate_df = (
        summary_df.groupby("variant", observed=True)[
            ["test_accuracy", "test_macro_f1", "test_weighted_f1", "test_minority_avg_f1"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregate_df.columns = [
        "variant",
        "test_accuracy_mean",
        "test_accuracy_std",
        "test_macro_f1_mean",
        "test_macro_f1_std",
        "test_weighted_f1_mean",
        "test_weighted_f1_std",
        "test_minority_avg_f1_mean",
        "test_minority_avg_f1_std",
    ]

    per_class_df = pd.DataFrame(per_class_rows_all)
    per_class_df["variant"] = pd.Categorical(per_class_df["variant"], ["plain", WEIGHTED_VARIANT], ordered=True)
    per_class_df["seed"] = pd.Categorical(per_class_df["seed"], SEEDS, ordered=True)
    per_class_df = per_class_df.sort_values(["split", "variant", "seed", "class"]).reset_index(drop=True)

    seed_table_csv = ROBUSTNESS_ANALYSIS_DIR / "seed_by_seed_comparison.csv"
    seed_table_md = ROBUSTNESS_ANALYSIS_DIR / "seed_by_seed_comparison.md"
    aggregate_csv = ROBUSTNESS_ANALYSIS_DIR / "aggregate_summary.csv"
    aggregate_md = ROBUSTNESS_ANALYSIS_DIR / "aggregate_summary.md"
    per_class_val_csv = ROBUSTNESS_ANALYSIS_DIR / "per_class_metrics_val.csv"
    per_class_test_csv = ROBUSTNESS_ANALYSIS_DIR / "per_class_metrics_test.csv"

    summary_df.to_csv(seed_table_csv, index=False)
    seed_table_md.write_text(dataframe_to_markdown(summary_df), encoding="utf-8")
    aggregate_df.to_csv(aggregate_csv, index=False)
    aggregate_md.write_text(dataframe_to_markdown(aggregate_df), encoding="utf-8")
    per_class_df[per_class_df["split"] == "val"].to_csv(per_class_val_csv, index=False)
    per_class_df[per_class_df["split"] == "test"].to_csv(per_class_test_csv, index=False)

    print(f"Phase 1c seed robustness analysis written to {to_repo_relative(ROBUSTNESS_ANALYSIS_DIR)}")


if __name__ == "__main__":
    app()
