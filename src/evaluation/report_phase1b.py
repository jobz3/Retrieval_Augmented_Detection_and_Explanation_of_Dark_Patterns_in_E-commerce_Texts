"""
Generate standardized Phase 1b comparison artifacts from plain and weighted checkpoints.

Usage:
    python -m src.evaluation.report_phase1b
"""

from __future__ import annotations

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
ANALYSIS_DIR = CONFIG.paths.phase1b_analysis
VARIANT_NAME = CONFIG.phase1b.weighted_variant

app = typer.Typer()


@dataclass(frozen=True)
class VariantSpec:
    model_key: str
    variant: str
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


def variant_specs() -> list[VariantSpec]:
    return [
        VariantSpec("bert", "plain", "bert_plain", CONFIG.paths.models / "bert"),
        VariantSpec(
            "bert",
            VARIANT_NAME,
            f"bert_{VARIANT_NAME}",
            CONFIG.paths.weighted_models / f"bert_{VARIANT_NAME}",
        ),
        VariantSpec("roberta", "plain", "roberta_plain", CONFIG.paths.models / "roberta"),
        VariantSpec(
            "roberta",
            VARIANT_NAME,
            f"roberta_{VARIANT_NAME}",
            CONFIG.paths.weighted_models / f"roberta_{VARIANT_NAME}",
        ),
    ]


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


def overall_row(
    spec: VariantSpec,
    split: str,
    report: dict[str, object],
    minority_labels: list[str],
) -> dict[str, object]:
    minority_avg_f1 = float(
        np.mean([report[label]["f1-score"] for label in minority_labels])  # type: ignore[index]
    )
    return {
        "model": spec.model_key,
        "variant": spec.variant,
        "checkpoint_name": spec.checkpoint_name,
        "split": split,
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],  # type: ignore[index]
        "weighted_f1": report["weighted avg"]["f1-score"],  # type: ignore[index]
        "minority_avg_f1": minority_avg_f1,
        "Forced Action F1": report["Forced Action"]["f1-score"],  # type: ignore[index]
        "Sneaking F1": report["Sneaking"]["f1-score"],  # type: ignore[index]
    }


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
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]

    for record in df.itertuples(index=False, name=None):
        values = [str(value) for value in record]
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join(rows) + "\n"


@app.command()
def main(batch_size: int = typer.Option(32, help="Evaluation batch size")) -> None:
    if not LOCKED_SPLIT_DIR.exists():
        raise FileNotFoundError(f"Locked split directory not found: {LOCKED_SPLIT_DIR}")

    specs = variant_specs()
    missing_checkpoints = [spec.checkpoint_path for spec in specs if not spec.checkpoint_path.exists()]
    if missing_checkpoints:
        formatted = ", ".join(to_repo_relative(path) for path in missing_checkpoints)
        raise FileNotFoundError(f"Missing checkpoints required for Phase 1b reporting: {formatted}")

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    label_names = label_names_for_split(LOCKED_SPLIT_DIR)
    train_class_counts = load_train_class_counts(LOCKED_SPLIT_DIR, label_names)
    minority_labels = [
        label for label, count in train_class_counts.items()
        if count < CONFIG.phase1b.minority_train_threshold
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    overall_rows: list[dict[str, object]] = []
    per_class_rows_all: list[dict[str, object]] = []

    for spec in specs:
        print(f"Evaluating {spec.checkpoint_name} on locked split using {device}")
        for split in ("val", "test"):
            report, class_rows, cm_norm = evaluate_variant_split(
                spec, split, label_names, device, batch_size
            )
            overall_rows.append(overall_row(spec, split, report, minority_labels))
            per_class_rows_all.extend(class_rows)

            if split == "test":
                cm_df = pd.DataFrame(cm_norm, index=label_names, columns=label_names)
                cm_csv_path = ANALYSIS_DIR / f"confusion_matrix_{spec.checkpoint_name}_test_norm.csv"
                cm_png_path = ANALYSIS_DIR / f"confusion_matrix_{spec.checkpoint_name}_test_norm.png"
                cm_df.to_csv(cm_csv_path)
                write_normalized_confusion_plot(
                    cm_norm,
                    label_names,
                    cm_png_path,
                    f"{spec.checkpoint_name} normalized confusion matrix (test)",
                )

    overall_df = pd.DataFrame(overall_rows)
    overall_df["model"] = pd.Categorical(overall_df["model"], ["bert", "roberta"], ordered=True)
    overall_df["variant"] = pd.Categorical(
        overall_df["variant"], ["plain", VARIANT_NAME], ordered=True
    )
    overall_df["split"] = pd.Categorical(overall_df["split"], ["val", "test"], ordered=True)
    overall_df = overall_df.sort_values(["model", "variant", "split"]).reset_index(drop=True)

    per_class_df = pd.DataFrame(per_class_rows_all)
    per_class_df["model"] = pd.Categorical(per_class_df["model"], ["bert", "roberta"], ordered=True)
    per_class_df["variant"] = pd.Categorical(
        per_class_df["variant"], ["plain", VARIANT_NAME], ordered=True
    )
    per_class_df["split"] = pd.Categorical(per_class_df["split"], ["val", "test"], ordered=True)
    per_class_df = per_class_df.sort_values(["split", "model", "variant", "class"]).reset_index(
        drop=True
    )

    overall_csv = ANALYSIS_DIR / "overall_comparison.csv"
    overall_md = ANALYSIS_DIR / "overall_comparison.md"
    per_class_val_csv = ANALYSIS_DIR / "per_class_metrics_val.csv"
    per_class_test_csv = ANALYSIS_DIR / "per_class_metrics_test.csv"
    per_class_f1_csv = ANALYSIS_DIR / "per_class_f1_matrix_test.csv"
    per_class_heatmap_png = ANALYSIS_DIR / "per_class_f1_heatmap_test.png"

    overall_df.to_csv(overall_csv, index=False)
    overall_md.write_text(dataframe_to_markdown(overall_df), encoding="utf-8")
    per_class_df[per_class_df["split"] == "val"].to_csv(per_class_val_csv, index=False)
    per_class_df[per_class_df["split"] == "test"].to_csv(per_class_test_csv, index=False)

    heatmap_source = per_class_df[per_class_df["split"] == "test"].copy()
    heatmap_source["model_variant"] = (
        heatmap_source["model"].astype(str) + "_" + heatmap_source["variant"].astype(str)
    )
    f1_matrix = heatmap_source.pivot(index="model_variant", columns="class", values="f1")
    ordered_rows = ["bert_plain", f"bert_{VARIANT_NAME}", "roberta_plain", f"roberta_{VARIANT_NAME}"]
    f1_matrix = f1_matrix.reindex(index=ordered_rows, columns=label_names)
    f1_matrix.to_csv(per_class_f1_csv)

    plt.figure(figsize=(12, 4))
    sns.heatmap(f1_matrix, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    plt.title("Per-class F1 on locked test split")
    plt.xlabel("Class")
    plt.ylabel("Model / variant")
    plt.tight_layout()
    plt.savefig(per_class_heatmap_png, dpi=200)
    plt.close()

    print(f"Phase 1b analysis written to {to_repo_relative(ANALYSIS_DIR)}")


if __name__ == "__main__":
    app()
