"""
Load a saved baseline checkpoint and evaluate on any split.

Usage:
    python -m src.baselines.evaluate --model bert --split test
    python -m src.baselines.evaluate --model roberta --split test
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report, f1_score
import torch

from src.data.dataset import DarkPatternDataset, load_label_map
from src.baselines.train_classifier import evaluate as _evaluate

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "outputs" / "models"

app = typer.Typer()


@app.command()
def main(
    model: str = typer.Option("bert", help="Model key: bert | roberta"),
    split: str = typer.Option("test", help="Split to evaluate: train | val | test"),
    batch_size: int = typer.Option(32),
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = MODEL_DIR / model

    if not save_path.exists():
        raise FileNotFoundError(
            f"No saved model at {save_path}. "
            f"Run `python -m src.baselines.train_classifier --model {model}` first."
        )

    label_map = load_label_map()
    id2label = {v: k for k, v in label_map.items()}
    label_names = [id2label[i] for i in range(len(id2label))]

    tokenizer = AutoTokenizer.from_pretrained(save_path)
    model_obj = AutoModelForSequenceClassification.from_pretrained(save_path).to(device)

    dataset = DarkPatternDataset(split, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size)

    macro_f1, preds, labels = _evaluate(model_obj, loader, device)

    print(f"\n=== {model.upper()} on {split} split ===")
    print(f"Macro F1: {macro_f1:.4f}")
    print(classification_report(labels, preds, target_names=label_names, zero_division=0))


if __name__ == "__main__":
    app()
