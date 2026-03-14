"""
Phase 1 — Data Preprocessing
-----------------------------
Loads the EC-DarkPattern dataset (dataset.tsv), cleans it, prints EDA statistics,
and writes stratified train / val / test splits to data/processed/.

Usage:
    python -m src.data.preprocess

Outputs:
    data/processed/train.csv
    data/processed/val.csv
    data/processed/test.csv
    data/processed/label_map.json   # category -> int id
"""

import json
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW_TSV = ROOT / "data" / "raw" / "dataset.tsv"
PROCESSED_DIR = ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# The 7 dark pattern categories from Yada et al. (2022)
DARK_PATTERN_CATEGORIES = [
    "Scarcity",
    "Urgency",
    "Social Proof",
    "Misdirection",
    "Obstruction",
    "Forced Action",
    "Sneaking",
]
NOT_DARK_PATTERN_LABEL = "Not Dark Pattern"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw(path: Path = RAW_TSV) -> pd.DataFrame:
    """Load the raw TSV into a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Clone the dataset with:\n"
            "  git clone https://github.com/yamanalab/ec-darkpattern.git\n"
            "then copy dataset/dataset.tsv → data/raw/dataset.tsv"
        )
    df = pd.read_csv(path, sep="\t")
    return df


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise column names, drop nulls and duplicates, normalise category labels.

    The raw TSV has columns: page_id, text, label, Pattern Category
    We rename to: page_id, text, binary_label, category
    """
    # Flexible column rename — handle any casing or spacing variations
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Rename to canonical names
    rename_map = {}
    for col in df.columns:
        if "pattern" in col and "category" in col:
            rename_map[col] = "category"
        elif col in ("label", "labels"):
            rename_map[col] = "binary_label"
    df = df.rename(columns=rename_map)

    # Drop rows with missing text or category
    before = len(df)
    df = df.dropna(subset=["text", "category"])
    df = df[df["text"].str.strip() != ""]

    # Drop exact duplicates on (text, category)
    df = df.drop_duplicates(subset=["text", "category"])
    after = len(df)
    print(f"Dropped {before - after} rows (null / empty / duplicate). Remaining: {after}")

    # Normalise category: strip whitespace, title-case
    df["category"] = df["category"].str.strip()

    # Rows with binary_label == 0 are "Not Dark Pattern"
    # Ensure category reflects this
    if "binary_label" in df.columns:
        df.loc[df["binary_label"] == 0, "category"] = NOT_DARK_PATTERN_LABEL

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------

def build_label_map(df: pd.DataFrame) -> dict[str, int]:
    """Return {category_string: int_id}, sorted for reproducibility."""
    categories = sorted(df["category"].unique().tolist())
    return {cat: idx for idx, cat in enumerate(categories)}


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------

def print_eda(df: pd.DataFrame) -> None:
    print("\n=== EDA ===")
    print(f"Total samples: {len(df)}")
    print(f"\nCategory distribution:")
    counts = df["category"].value_counts()
    for cat, n in counts.items():
        pct = 100 * n / len(df)
        print(f"  {cat:<22} {n:>5}  ({pct:.1f}%)")

    print(f"\nText length (chars) — mean: {df['text'].str.len().mean():.0f}, "
          f"median: {df['text'].str.len().median():.0f}, "
          f"max: {df['text'].str.len().max()}")
    print(f"Text length (tokens ≈ words) — mean: {df['text'].str.split().str.len().mean():.1f}")

    if "binary_label" in df.columns:
        dark = (df["binary_label"] == 1).sum()
        not_dark = (df["binary_label"] == 0).sum()
        print(f"\nBinary: dark pattern={dark}, not dark pattern={not_dark}")


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split by category.
    Returns (train_df, val_df, test_df).
    """
    # First split off test
    test_size = 1.0 - train_ratio
    train_df, temp_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["category"]
    )

    # Split temp into val and test (equal halves when val_ratio == test_ratio)
    val_fraction_of_temp = val_ratio / test_size

    # Some rare categories may have only 1 sample in temp; stratify only when safe
    min_class_count = temp_df["category"].value_counts().min()
    if min_class_count < 2:
        rare = temp_df["category"].value_counts()[temp_df["category"].value_counts() < 2].index.tolist()
        print(f"Warning: {rare} have <2 samples in temp split — using non-stratified val/test split.")
        stratify_arg = None
    else:
        stratify_arg = temp_df["category"]

    val_df, test_df = train_test_split(
        temp_df,
        test_size=1.0 - val_fraction_of_temp,
        random_state=seed,
        stratify=stratify_arg,
    )

    print(f"\nSplit sizes — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_map: dict[str, int],
    out_dir: Path = PROCESSED_DIR,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Add integer label column
    for split_df in (train_df, val_df, test_df):
        split_df["label_id"] = split_df["category"].map(label_map)

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    with open(out_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\nSaved splits and label_map to {out_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading raw dataset from {RAW_TSV} ...")
    df = load_raw()
    df = clean(df)
    print_eda(df)
    label_map = build_label_map(df)
    print(f"\nLabel map: {label_map}")
    train_df, val_df, test_df = stratified_split(df)
    save_splits(train_df, val_df, test_df, label_map)


if __name__ == "__main__":
    main()
