"""
Phase 1 — Baseline Classifier Training
----------------------------------------
Fine-tunes BERT-base or RoBERTa-large on the EC-DarkPattern training split
(multi-class classification over 7 dark pattern categories + "Not Dark Pattern").

Usage:
    python -m src.baselines.train_classifier --model bert
    python -m src.baselines.train_classifier --model roberta

Outputs:
    outputs/models/{model_name}/          — saved HuggingFace model + tokenizer
    results/baselines/{model_name}.json   — macro F1, per-class metrics, confusion matrix
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import typer
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm

from src.data.dataset import DarkPatternDataset, load_label_map
from src.utils.config import load_config

CONFIG = load_config()
ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = CONFIG.paths.models
RESULTS_DIR = CONFIG.paths.baseline_results

app = typer.Typer()


def resolve_path_override(path: Path | None, default: Path) -> Path:
    if path is None:
        return default
    return path if path.is_absolute() else ROOT / path


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(
    model: AutoModelForSequenceClassification,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: AutoModelForSequenceClassification,
    loader: DataLoader,
    device: str,
) -> tuple[float, list[int], list[int]]:
    """Returns (macro_f1, all_preds, all_labels)."""
    model.eval()
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = outputs.logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return macro_f1, all_preds, all_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@app.command()
def main(
    model: str = typer.Option("bert", help="Model key: bert | roberta"),
    seed: int = typer.Option(CONFIG.project.seed, help="Random seed"),
    processed_dir: Path | None = typer.Option(None, help="Override processed split directory"),
    save_dir: Path | None = typer.Option(None, help="Override checkpoint save directory"),
    results_path: Path | None = typer.Option(None, help="Override results JSON path"),
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = CONFIG.baselines.by_name(model)
    processed_dir_path = resolve_path_override(processed_dir, CONFIG.paths.processed_data)
    save_path = resolve_path_override(save_dir, MODEL_DIR / model)
    results_output_path = resolve_path_override(results_path, RESULTS_DIR / f"{model}.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    label_map = load_label_map(processed_dir_path)
    num_labels = len(label_map)
    id2label = {v: k for k, v in label_map.items()}
    label_names = [id2label[i] for i in range(num_labels)]

    print(f"\nLoading tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    train_dataset = DarkPatternDataset("train", tokenizer, cfg.max_length, processed_dir_path)
    val_dataset = DarkPatternDataset("val", tokenizer, cfg.max_length, processed_dir_path)
    test_dataset = DarkPatternDataset("test", tokenizer, cfg.max_length, processed_dir_path)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size * 2)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size * 2)

    print(f"Loading model: {cfg.model_name} (num_labels={num_labels})")
    model_obj = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label_map,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model_obj.parameters(), lr=cfg.learning_rate, weight_decay=0.01
    )
    total_steps = len(train_loader) * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Training
    best_val_f1 = 0.0
    best_epoch = 0
    for epoch in range(1, cfg.num_epochs + 1):
        train_loss = train_epoch(model_obj, train_loader, optimizer, scheduler, device)
        val_f1, _, _ = evaluate(model_obj, val_loader, device)
        print(f"Epoch {epoch}/{cfg.num_epochs} - loss: {train_loss:.4f}, val macro F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            save_path.mkdir(parents=True, exist_ok=True)
            model_obj.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"  New best checkpoint saved to {save_path}")

    # Final test evaluation using best checkpoint
    print(f"\nBest epoch: {best_epoch} (val macro F1 = {best_val_f1:.4f})")
    print(f"Loading best checkpoint for test evaluation ...")
    best_model = AutoModelForSequenceClassification.from_pretrained(save_path).to(device)
    test_f1, test_preds, test_labels = evaluate(best_model, test_loader, device)

    all_label_ids = list(range(len(label_names)))
    report = classification_report(
        test_labels, test_preds,
        labels=all_label_ids, target_names=label_names,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(test_labels, test_preds, labels=all_label_ids).tolist()

    print(f"\nTest macro F1: {test_f1:.4f}")
    print(classification_report(
        test_labels, test_preds,
        labels=all_label_ids, target_names=label_names, zero_division=0,
    ))

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "model": cfg.model_name,
        "model_key": model,
        "seed": seed,
        "best_val_macro_f1": best_val_f1,
        "test_macro_f1": test_f1,
        "classification_report": report,
        "confusion_matrix": cm,
        "label_names": label_names,
        "split_root": str(processed_dir_path.relative_to(ROOT)),
        "config": {
            "model_name": cfg.model_name,
            "max_length": cfg.max_length,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "num_epochs": cfg.num_epochs,
            "warmup_ratio": cfg.warmup_ratio,
        },
    }
    results_output_path.parent.mkdir(parents=True, exist_ok=True)
    with results_output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_output_path}")


if __name__ == "__main__":
    app()
