"""
Cross-dataset Generalisation: Mathur et al. (2019) as held-out test
---------------------------------------------------------------------
Evaluates the BERT model trained on Yada et al. v2 data against the
Mathur et al. dark-patterns dataset without any fine-tuning on it.

Why this matters:
  A model that only generalises within Yada may have learned Yada-specific
  surface patterns rather than the underlying dark pattern signal.  Good
  performance on Mathur — a completely independent crawl from different
  websites with different writing styles — is stronger evidence of genuine
  generalisation.

What Mathur contains:
  1,818 UI text snippets scraped from 11K shopping sites, labelled with the
  same 7 dark pattern categories used by Yada (+ no "Not Dark Pattern" class).
  Each row has a "Deceptive?" flag (Yes / No / Depends) indicating whether
  the annotators judged the instance to be genuinely deceptive.

Evaluation strategy:
  - Full set    : all 1,818 rows (as a macro-level coverage test)
  - Deceptive   : Deceptive? == Yes (208 rows — the clean positives)
  - Non-deceptive: Deceptive? == No (1,584 rows — borderline / ambiguous)
  Because Mathur has no "Not Dark Pattern" class the model is tested only on
  the 7 dark pattern categories.  We report macro F1 over those 7 classes.

Usage:
    python -m src.evaluation.cross_dataset
    python -m src.evaluation.cross_dataset --model-path outputs/models/bert_train_v2
    python -m src.evaluation.cross_dataset --deceptive-only
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import torch
import typer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "evaluation"
CACHE_PATH  = ROOT / "data" / "external" / "mathur_dark_patterns.csv"

MATHUR_CSV_URL = (
    "https://raw.githubusercontent.com/aruneshmathur/dark-patterns"
    "/master/data/final-dark-patterns/dark-patterns.csv"
)

app = typer.Typer()


# ---------------------------------------------------------------------------
# Download / cache Mathur CSV
# ---------------------------------------------------------------------------

def fetch_mathur(cache: Path = CACHE_PATH) -> list[dict]:
    """Download Mathur dark-patterns.csv (cached after first run)."""
    if not cache.exists():
        print(f"Downloading Mathur dataset → {cache} ...")
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MATHUR_CSV_URL, cache)
        print("  Done.")
    else:
        print(f"Using cached Mathur data: {cache}")

    import csv
    with open(cache, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows


# ---------------------------------------------------------------------------
# Mapping and filtering
# ---------------------------------------------------------------------------

# Mathur's column names
COL_TEXT      = "Pattern String"
COL_CATEGORY  = "Pattern Category"
COL_DECEPTIVE = "Deceptive?"

# Categories present in both datasets (Yada label_map keys)
VALID_CATEGORIES = {
    "Forced Action", "Misdirection", "Obstruction",
    "Scarcity", "Sneaking", "Social Proof", "Urgency",
}


def load_and_filter(
    rows: list[dict],
    deceptive_only: bool = False,
) -> tuple[list[str], list[str]]:
    """
    Extract (texts, categories) from Mathur rows.

    Filters:
      - Category must be one of the 7 shared labels
      - Text must be non-empty
      - If deceptive_only: keep only Deceptive? == Yes
    """
    texts, cats = [], []
    for row in rows:
        cat  = row.get(COL_CATEGORY, "").strip()
        text = row.get(COL_TEXT, "").strip()
        decp = row.get(COL_DECEPTIVE, "").strip()

        if cat not in VALID_CATEGORIES:
            continue
        if not text:
            continue
        if deceptive_only and decp.lower() != "yes":
            continue

        texts.append(text)
        cats.append(cat)

    return texts, cats


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        label_ids: list[int],
        tokenizer,
        max_length: int = 128,
    ):
        self.texts     = texts
        self.label_ids = label_ids
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.label_ids[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_eval(
    model,
    loader: DataLoader,
    device: str,
) -> tuple[list[int], list[int]]:
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = outputs.logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(batch["labels"].tolist())
    return all_preds, all_labels


def evaluate_split(
    texts: list[str],
    cats: list[str],
    label_map: dict[str, int],
    id2label: dict[int, str],
    model,
    tokenizer,
    device: str,
    split_name: str,
    batch_size: int = 32,
) -> dict:
    """Run evaluation and return metrics dict."""
    label_ids    = [label_map[c] for c in cats]
    # Only the label IDs actually present in this split
    present_ids  = sorted(set(label_ids))
    present_names = [id2label[i] for i in present_ids]

    dataset = TextDataset(texts, label_ids, tokenizer)
    loader  = DataLoader(dataset, batch_size=batch_size)

    preds, labels = run_eval(model, loader, device)

    macro_f1 = f1_score(labels, preds, labels=present_ids, average="macro", zero_division=0)
    report   = classification_report(
        labels, preds,
        labels=present_ids, target_names=present_names,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(labels, preds, labels=present_ids).tolist()

    print(f"\n{'='*65}")
    print(f"  {split_name}  (n={len(texts)})")
    print(f"{'='*65}")
    print(classification_report(
        labels, preds,
        labels=present_ids, target_names=present_names, zero_division=0,
    ))
    print(f"  Macro F1 (7 dark-pattern classes): {macro_f1:.4f}")

    # Show where the model is confused
    print("\n  Most frequent mispredictions:")
    confusion: dict[tuple, int] = {}
    for true, pred in zip(labels, preds):
        if true != pred:
            key = (id2label[true], id2label[pred])
            confusion[key] = confusion.get(key, 0) + 1
    for (true_name, pred_name), count in sorted(confusion.items(), key=lambda x: -x[1])[:10]:
        print(f"    {true_name:20s} → {pred_name:20s}  ({count}×)")

    return {
        "split":             split_name,
        "n":                 len(texts),
        "macro_f1":          round(macro_f1, 4),
        "classification_report": report,
        "confusion_matrix":  cm,
        "label_names":       present_names,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@app.command()
def main(
    model_path:      Path = typer.Option(None,  help="HF model dir (default: outputs/models/bert_train_v2)"),
    deceptive_only:  bool = typer.Option(False, help="Only evaluate on Deceptive?==Yes rows"),
    batch_size:      int  = typer.Option(32,    help="Inference batch size"),
) -> None:
    if model_path is None:
        model_path = ROOT / "outputs" / "models" / "bert_train_v2"

    # Load label map
    with open(ROOT / "data" / "processed" / "label_map.json") as f:
        label_map = json.load(f)
    id2label = {v: k for k, v in label_map.items()}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nModel:  {model_path}")
    print(f"Device: {device}")

    print("\nLoading model and tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

    # Fetch data
    raw_rows = fetch_mathur()
    print(f"\nMathur dataset: {len(raw_rows)} rows total")

    # Distribution overview
    from collections import Counter
    cat_counts = Counter(r.get(COL_CATEGORY, "?") for r in raw_rows)
    dec_counts = Counter(r.get(COL_DECEPTIVE, "?") for r in raw_rows)
    print("\nCategory distribution:")
    for cat, n in cat_counts.most_common():
        mark = "✓" if cat in VALID_CATEGORIES else "✗ (skipped)"
        print(f"  {cat:25s} {n:4d}  {mark}")
    print(f"\nDeceptive? distribution: {dict(dec_counts)}")

    results = []

    if deceptive_only:
        texts, cats = load_and_filter(raw_rows, deceptive_only=True)
        r = evaluate_split(texts, cats, label_map, id2label, model, tokenizer, device,
                           "Deceptive only (Deceptive?=Yes)", batch_size)
        results.append(r)
    else:
        # Full set
        texts_all, cats_all = load_and_filter(raw_rows, deceptive_only=False)
        r_all = evaluate_split(texts_all, cats_all, label_map, id2label, model, tokenizer, device,
                               "All Mathur (7 shared categories)", batch_size)
        results.append(r_all)

        # Deceptive subset
        texts_dec, cats_dec = load_and_filter(raw_rows, deceptive_only=True)
        r_dec = evaluate_split(texts_dec, cats_dec, label_map, id2label, model, tokenizer, device,
                               "Deceptive only (Deceptive?=Yes)", batch_size)
        results.append(r_dec)

    # Summary
    print(f"\n{'='*65}")
    print("CROSS-DATASET SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Split':<40} {'n':>6}  {'Macro F1':>10}")
    print(f"  {'-'*58}")
    for r in results:
        print(f"  {r['split']:<40} {r['n']:>6}  {r['macro_f1']:>10.4f}")

    # Reference: Yada in-distribution test F1
    yada_path = ROOT / "results" / "baselines" / "bert_train_v2.json"
    if yada_path.exists():
        with open(yada_path) as f:
            yada = json.load(f)
        print(f"\n  Yada in-distribution test F1 (reference): {yada['test_macro_f1']:.4f}")
        print(f"  Cross-dataset / in-distribution ratio:     "
              f"{results[0]['macro_f1'] / yada['test_macro_f1']:.2f}")

    print()
    print("Interpretation:")
    print("  Macro F1 ≥ 0.70  → strong cross-dataset generalisation")
    print("  Macro F1 0.50–0.70 → partial — model learned some real signals")
    print("  Macro F1 < 0.50  → mostly domain-specific, limited transfer")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "cross_dataset_mathur.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    app()
