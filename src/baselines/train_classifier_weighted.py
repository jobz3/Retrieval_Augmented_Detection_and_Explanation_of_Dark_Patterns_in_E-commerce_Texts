"""
Phase 1b weighted baseline training on the locked Phase 1 split.

Usage:
    python -m src.baselines.train_classifier_weighted --model bert
    python -m src.baselines.train_classifier_weighted --model roberta
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import typer
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.baselines.train_classifier import evaluate as evaluate_checkpoint
from src.data.dataset import DarkPatternDataset, load_label_map
from src.utils.config import load_config


CONFIG = load_config()
ROOT = Path(__file__).resolve().parents[2]
LOCKED_SPLIT_DIR = CONFIG.paths.locked_processed_data
WEIGHTED_MODEL_DIR = CONFIG.paths.weighted_models
WEIGHTED_RESULTS_DIR = CONFIG.paths.weighted_baseline_results
VARIANT_NAME = CONFIG.phase1b.weighted_variant

app = typer.Typer()


def to_repo_relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_weight_spec(processed_dir: Path) -> tuple[list[str], dict[str, int], list[float]]:
    label_map = load_label_map(processed_dir)
    id2label = {idx: label for label, idx in label_map.items()}
    label_names = [id2label[idx] for idx in range(len(id2label))]

    train_df = pd.read_csv(processed_dir / "train.csv")
    total_examples = len(train_df)
    num_classes = len(label_names)

    class_counts: dict[str, int] = {}
    class_weights: list[float] = []
    for label_id, label_name in enumerate(label_names):
        class_count = int((train_df["label_id"] == label_id).sum())
        if class_count <= 0:
            raise ValueError(f"Locked train split is missing class '{label_name}'.")
        class_counts[label_name] = class_count
        class_weights.append(total_examples / (num_classes * class_count))

    return label_names, class_counts, class_weights


def train_epoch_weighted(
    model: AutoModelForSequenceClassification,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_fn: torch.nn.Module,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@app.command()
def main(
    model: str = typer.Option("bert", help="Model key: bert | roberta"),
    seed: int = typer.Option(CONFIG.project.seed, help="Random seed"),
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = CONFIG.baselines.by_name(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    variant_key = f"{model}_{VARIANT_NAME}"
    save_path = WEIGHTED_MODEL_DIR / variant_key
    results_path = WEIGHTED_RESULTS_DIR / f"{variant_key}.json"

    label_map = load_label_map(LOCKED_SPLIT_DIR)
    num_labels = len(label_map)
    id2label = {idx: label for label, idx in label_map.items()}
    label_names, train_class_counts, class_weights = load_weight_spec(LOCKED_SPLIT_DIR)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    train_dataset = DarkPatternDataset(
        "train", tokenizer, cfg.max_length, processed_dir=LOCKED_SPLIT_DIR
    )
    val_dataset = DarkPatternDataset(
        "val", tokenizer, cfg.max_length, processed_dir=LOCKED_SPLIT_DIR
    )
    test_dataset = DarkPatternDataset(
        "test", tokenizer, cfg.max_length, processed_dir=LOCKED_SPLIT_DIR
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size * 2)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size * 2)

    model_obj = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label_map,
    ).to(device)

    class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weight_tensor)

    optimizer = torch.optim.AdamW(
        model_obj.parameters(), lr=cfg.learning_rate, weight_decay=0.01
    )
    total_steps = len(train_loader) * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_f1 = -1.0
    best_epoch = 0

    print(f"Using device: {device}")
    print(f"Training variant: {variant_key}")
    print(f"Locked split: {to_repo_relative(LOCKED_SPLIT_DIR)}")
    print(f"Class weights: {dict(zip(label_names, class_weights, strict=True))}")

    for epoch in range(1, cfg.num_epochs + 1):
        train_loss = train_epoch_weighted(
            model_obj, train_loader, optimizer, scheduler, loss_fn, device
        )
        val_f1, _, _ = evaluate_checkpoint(model_obj, val_loader, device)
        print(
            f"Epoch {epoch}/{cfg.num_epochs} - loss: {train_loss:.4f}, "
            f"val macro F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            save_path.mkdir(parents=True, exist_ok=True)
            model_obj.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"  New best checkpoint saved to {to_repo_relative(save_path)}")

    best_model = AutoModelForSequenceClassification.from_pretrained(save_path).to(device)
    test_f1, test_preds, test_labels = evaluate_checkpoint(best_model, test_loader, device)

    all_label_ids = list(range(len(label_names)))
    report = classification_report(
        test_labels,
        test_preds,
        labels=all_label_ids,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(test_labels, test_preds, labels=all_label_ids).tolist()

    results = {
        "model": cfg.model_name,
        "model_key": model,
        "variant": VARIANT_NAME,
        "checkpoint_name": variant_key,
        "split_root": to_repo_relative(LOCKED_SPLIT_DIR),
        "best_val_macro_f1": best_val_f1,
        "best_epoch": best_epoch,
        "test_macro_f1": test_f1,
        "test_accuracy": report["accuracy"],
        "classification_report": report,
        "confusion_matrix": cm,
        "label_names": label_names,
        "class_weight_formula": "N_train / (K * n_c)",
        "train_class_counts": train_class_counts,
        "class_weights": {
            label: weight for label, weight in zip(label_names, class_weights, strict=True)
        },
        "config": {
            "model_name": cfg.model_name,
            "max_length": cfg.max_length,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "num_epochs": cfg.num_epochs,
            "warmup_ratio": cfg.warmup_ratio,
        },
    }

    WEIGHTED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"\nBest epoch: {best_epoch} (val macro F1 = {best_val_f1:.4f})")
    print(f"Test macro F1: {test_f1:.4f}")
    print(classification_report(
        test_labels,
        test_preds,
        labels=all_label_ids,
        target_names=label_names,
        zero_division=0,
    ))
    print(f"Results saved to {to_repo_relative(results_path)}")


if __name__ == "__main__":
    app()
