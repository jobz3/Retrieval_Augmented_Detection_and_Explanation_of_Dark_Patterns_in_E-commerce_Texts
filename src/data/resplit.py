"""
Phase 1.5 — Re-split with Synthetic Data
-----------------------------------------
Pools the original examples (train + val + test) with synthetic examples
from augment.py, then re-runs a stratified 70/15/15 split.

This ensures rare classes (Forced Action, Sneaking, Obstruction) appear
in all three splits so validation and test metrics are meaningful.

Synthetic examples are flagged with synthetic=True in the output CSVs.
For human evaluation and the paper, real-only subsets can be extracted
by filtering on this column.

Usage:
    python -m src.data.resplit

Outputs:
    data/processed/train_v2.csv
    data/processed/val_v2.csv
    data/processed/test_v2.csv
    data/processed/label_map.json   (unchanged)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
SYNTHETIC_LOG = ROOT / "results" / "augmentation" / "synthetic_examples.json"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
SEED        = 42


def load_original_pool() -> pd.DataFrame:
    """Combine train + val + test into one pool, marking them as real."""
    frames = []
    for split in ("train", "val", "test"):
        df = pd.read_csv(PROCESSED_DIR / f"{split}.csv")
        df["synthetic"] = False
        frames.append(df)
    pool = pd.concat(frames, ignore_index=True)
    # Drop exact text duplicates (safety)
    pool = pool.drop_duplicates(subset=["text", "category"])
    print(f"Original pool: {len(pool)} examples")
    return pool


def load_synthetic(label_map: dict[str, int]) -> pd.DataFrame:
    """Load synthetic examples from the augmentation log."""
    with open(SYNTHETIC_LOG) as f:
        log: dict[str, list[str]] = json.load(f)

    rows = []
    start_id = 9000
    for cat, texts in log.items():
        for i, text in enumerate(texts):
            rows.append({
                "page_id":      start_id + i,
                "text":         text,
                "binary_label": 1,
                "category":     cat,
                "label_id":     label_map[cat],
                "synthetic":    True,
            })
        start_id += len(texts) + 1

    df = pd.DataFrame(rows)
    if df.empty:
        print("Synthetic pool: 0 examples")
        return pd.DataFrame(columns=["page_id", "text", "binary_label", "category", "label_id", "synthetic"])
    if "category" not in df.columns:
        raise ValueError(f"Synthetic examples loaded but missing 'category' column. Columns: {list(df.columns)}")
    print(f"Synthetic pool: {len(df)} examples  "
          f"({df['category'].value_counts().to_dict()})")
    return df


def stratified_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_size = 1.0 - TRAIN_RATIO
    train_df, temp_df = train_test_split(
        df, test_size=test_size, random_state=SEED, stratify=df["category"]
    )

    val_fraction_of_temp = VAL_RATIO / test_size
    min_class = temp_df["category"].value_counts().min()
    if min_class < 2:
        rare = temp_df["category"].value_counts()[
            temp_df["category"].value_counts() < 2
        ].index.tolist()
        print(f"Warning: {rare} have <2 samples in temp — using non-stratified val/test split.")
        stratify_arg = None
    else:
        stratify_arg = temp_df["category"]

    val_df, test_df = train_test_split(
        temp_df,
        test_size=1.0 - val_fraction_of_temp,
        random_state=SEED,
        stratify=stratify_arg,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def print_distribution(label: str, df: pd.DataFrame) -> None:
    print(f"\n{label} ({len(df)} total):")
    counts = df["category"].value_counts()
    synth_counts = df[df["synthetic"]]["category"].value_counts()
    for cat, n in counts.items():
        s = synth_counts.get(cat, 0)
        tag = f"  ({s} synthetic)" if s else ""
        print(f"  {n:>5}  {cat}{tag}")


def main() -> None:
    label_map: dict[str, int] = json.loads(
        (PROCESSED_DIR / "label_map.json").read_text()
    )

    original = load_original_pool()
    # Ensure label_id is present on original rows
    original["label_id"] = original["category"].map(label_map)

    synthetic = load_synthetic(label_map)

    pool = pd.concat([original, synthetic], ignore_index=True)
    pool = pool.drop_duplicates(subset=["text", "category"])
    print(f"\nCombined pool: {len(pool)} examples")

    train_df, val_df, test_df = stratified_split(pool)

    cols = ["page_id", "text", "binary_label", "category", "label_id", "synthetic"]
    for df in (train_df, val_df, test_df):
        for col in cols:
            if col not in df.columns:
                df[col] = False

    print_distribution("train_v2", train_df)
    print_distribution("val_v2",   val_df)
    print_distribution("test_v2",  test_df)

    train_df[cols].to_csv(PROCESSED_DIR / "train_v2.csv", index=False)
    val_df[cols].to_csv(PROCESSED_DIR / "val_v2.csv",     index=False)
    test_df[cols].to_csv(PROCESSED_DIR / "test_v2.csv",   index=False)

    print(f"\nSaved train_v2 / val_v2 / test_v2 to {PROCESSED_DIR}/")
    print("\nNext: re-train BERT on the new split:")
    print("  python -m src.baselines.train_classifier --model bert --train-file train_v2")


if __name__ == "__main__":
    main()
