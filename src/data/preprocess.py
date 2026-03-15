"""
Phase 1 data preprocessing for EC-DarkPattern.

Usage:
    python -m src.data.preprocess

Outputs:
    data/processed/ec_darkpattern/train.csv
    data/processed/ec_darkpattern/val.csv
    data/processed/ec_darkpattern/test.csv
    data/processed/ec_darkpattern/label_map.json
    data/processed/ec_darkpattern/raw_audit.json
    data/processed/ec_darkpattern/split_manifest.json
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from src.utils.config import load_config, raw_audit_path, split_manifest_path


NOT_DARK_PATTERN_LABEL = "Not Dark Pattern"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw(path: Path) -> pd.DataFrame:
    """Load the canonical raw TSV into a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(
            f"Canonical raw dataset not found at {path}.\n"
            "Run `python -m src.data.bootstrap_raw` first."
        )
    return pd.read_csv(path, sep="\t")


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Standardise column names, drop nulls and duplicates, and normalise labels.

    Returns:
        cleaned DataFrame, plus simple removal statistics for the audit artifact.
    """
    working_df = df.copy()
    working_df.columns = [column.strip().lower().replace(" ", "_") for column in working_df.columns]

    rename_map = {}
    for column in working_df.columns:
        if "pattern" in column and "category" in column:
            rename_map[column] = "category"
        elif column in {"label", "labels"}:
            rename_map[column] = "binary_label"
    working_df = working_df.rename(columns=rename_map)

    required_columns = {"text", "category"}
    missing_columns = required_columns.difference(working_df.columns)
    if missing_columns:
        raise ValueError(f"Raw dataset is missing required columns: {sorted(missing_columns)}")

    removal_stats = {
        "input_rows": int(len(working_df)),
        "dropped_missing_text_or_category": 0,
        "dropped_empty_text": 0,
        "dropped_duplicate_text_category": 0,
    }

    before = len(working_df)
    working_df = working_df.dropna(subset=["text", "category"]).copy()
    removal_stats["dropped_missing_text_or_category"] = before - len(working_df)

    working_df["text"] = working_df["text"].astype(str)
    before = len(working_df)
    working_df = working_df[working_df["text"].str.strip() != ""].copy()
    removal_stats["dropped_empty_text"] = before - len(working_df)

    working_df["category"] = working_df["category"].astype(str).str.strip()

    if "binary_label" in working_df.columns:
        binary_values = pd.to_numeric(working_df["binary_label"], errors="coerce")
        working_df.loc[binary_values == 0, "category"] = NOT_DARK_PATTERN_LABEL

    before = len(working_df)
    working_df = working_df.drop_duplicates(subset=["text", "category"]).reset_index(drop=True)
    removal_stats["dropped_duplicate_text_category"] = before - len(working_df)

    return working_df, removal_stats


def build_label_map(df: pd.DataFrame, configured_categories: list[str]) -> dict[str, int]:
    """Return a reproducible category -> id mapping using the config-defined order."""
    observed_categories = set(df["category"].unique().tolist())
    configured_category_set = set(configured_categories)
    unknown_categories = sorted(observed_categories.difference(configured_category_set))
    if unknown_categories:
        raise ValueError(f"Observed categories not present in config.yaml: {unknown_categories}")

    ordered_categories = [category for category in configured_categories if category in observed_categories]
    return {category: index for index, category in enumerate(ordered_categories)}


def text_length_stats(texts: pd.Series) -> dict[str, float | int]:
    char_lengths = texts.str.len()
    token_lengths = texts.str.split().str.len()
    return {
        "char_min": int(char_lengths.min()),
        "char_mean": round(float(char_lengths.mean()), 2),
        "char_median": float(char_lengths.median()),
        "char_max": int(char_lengths.max()),
        "token_min": int(token_lengths.min()),
        "token_mean": round(float(token_lengths.mean()), 2),
        "token_median": float(token_lengths.median()),
        "token_max": int(token_lengths.max()),
    }


def build_raw_audit(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    raw_path: Path,
    removal_stats: dict[str, int],
) -> dict:
    raw_label_column = None
    for candidate in ["Pattern Category", "pattern_category", "category"]:
        if candidate in raw_df.columns:
            raw_label_column = candidate
            break

    raw_labels = raw_df[raw_label_column].astype(str).str.strip() if raw_label_column else pd.Series(dtype=str)
    cleaned_labels = cleaned_df["category"].astype(str)

    return {
        "source_raw_path": str(raw_path),
        "raw_file_sha256": file_sha256(raw_path),
        "raw_input": {
            "total_row_count": int(len(raw_df)),
            "per_class_counts": dict(Counter(raw_labels.tolist())),
            "unique_labels": sorted(raw_labels.unique().tolist()),
            "text_length_stats": text_length_stats(raw_df["text"].astype(str)),
        },
        "cleaned_dataset": {
            "total_row_count": int(len(cleaned_df)),
            "per_class_counts": dict(Counter(cleaned_labels.tolist())),
            "unique_labels": sorted(cleaned_labels.unique().tolist()),
            "text_length_stats": text_length_stats(cleaned_df["text"].astype(str)),
        },
        "normalization": removal_stats,
    }


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)


def allocate_split_counts(total_count: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """
    Deterministically allocate per-class counts across train/val/test.

    Train must always contain the class when the class exists.
    Val/test must each contain the class when that is mathematically feasible.
    """
    if total_count <= 0:
        return 0, 0, 0

    positive_split_indices = [index for index, ratio in enumerate(ratios) if ratio > 0]
    minimums = [0, 0, 0]

    if 0 in positive_split_indices:
        minimums[0] = 1

    if total_count >= len(positive_split_indices):
        minimums = [1 if ratio > 0 else 0 for ratio in ratios]

    if sum(minimums) > total_count:
        minimums = [1 if index == 0 else 0 for index in range(len(ratios))]

    targets = [total_count * ratio for ratio in ratios]
    counts = minimums[:]
    remaining = total_count - sum(counts)

    for _ in range(remaining):
        best_index = max(
            range(len(ratios)),
            key=lambda index: (
                targets[index] - counts[index] if ratios[index] > 0 else float("-inf"),
                ratios[index],
                -index,
            ),
        )
        counts[best_index] += 1

    return counts[0], counts[1], counts[2]


def deterministic_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split each category independently using deterministic shuffling plus
    per-class allocation that preserves rare classes where feasible.
    """
    split_frames: dict[str, list[pd.DataFrame]] = {"train": [], "val": [], "test": []}
    ratios = (train_ratio, val_ratio, test_ratio)

    ordered_categories = sorted(df["category"].unique().tolist())
    for category_index, category in enumerate(ordered_categories):
        category_df = df[df["category"] == category].sample(
            frac=1.0,
            random_state=seed + category_index,
        )
        train_count, val_count, test_count = allocate_split_counts(len(category_df), ratios)

        split_frames["train"].append(category_df.iloc[:train_count].copy())
        split_frames["val"].append(category_df.iloc[train_count : train_count + val_count].copy())
        split_frames["test"].append(
            category_df.iloc[train_count + val_count : train_count + val_count + test_count].copy()
        )

    train_df = pd.concat(split_frames["train"], ignore_index=True).sample(frac=1.0, random_state=seed)
    val_df = pd.concat(split_frames["val"], ignore_index=True).sample(frac=1.0, random_state=seed + 1)
    test_df = pd.concat(split_frames["test"], ignore_index=True).sample(frac=1.0, random_state=seed + 2)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def category_counts(df: pd.DataFrame, category_order: list[str]) -> dict[str, int]:
    counts = Counter(df["category"].tolist())
    return {category: int(counts.get(category, 0)) for category in category_order}


def validate_splits(
    cleaned_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ratios: tuple[float, float, float],
) -> None:
    raw_counts = Counter(cleaned_df["category"].tolist())
    split_counts = {
        "train": Counter(train_df["category"].tolist()),
        "val": Counter(val_df["category"].tolist()),
        "test": Counter(test_df["category"].tolist()),
    }

    positive_split_count = sum(1 for ratio in ratios if ratio > 0)
    errors: list[str] = []

    total_rows = len(train_df) + len(val_df) + len(test_df)
    if total_rows != len(cleaned_df):
        errors.append(
            f"Row-count mismatch: cleaned={len(cleaned_df)}, train+val+test={total_rows}"
        )

    for category, raw_count in sorted(raw_counts.items()):
        train_count = split_counts["train"].get(category, 0)
        val_count = split_counts["val"].get(category, 0)
        test_count = split_counts["test"].get(category, 0)
        split_total = train_count + val_count + test_count

        if split_total != raw_count:
            errors.append(
                f"{category}: split totals do not match raw count "
                f"(raw={raw_count}, train={train_count}, val={val_count}, test={test_count})"
            )

        if train_count == 0:
            errors.append(
                f"{category}: class is missing from train even though it exists in raw data "
                f"(raw={raw_count}, train={train_count}, val={val_count}, test={test_count})"
            )

        full_split_coverage_feasible = raw_count >= positive_split_count
        if ratios[1] > 0 and full_split_coverage_feasible and val_count == 0:
            errors.append(
                f"{category}: class is missing from val despite being feasible to cover all splits "
                f"(raw={raw_count}, requires>={positive_split_count}, train={train_count}, "
                f"val={val_count}, test={test_count})"
            )
        if ratios[2] > 0 and full_split_coverage_feasible and test_count == 0:
            errors.append(
                f"{category}: class is missing from test despite being feasible to cover all splits "
                f"(raw={raw_count}, requires>={positive_split_count}, train={train_count}, "
                f"val={val_count}, test={test_count})"
            )

    if errors:
        raise ValueError("Split validation failed:\n" + "\n".join(f"- {error}" for error in errors))


def build_split_manifest(
    cleaned_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    raw_path: Path,
    seed: int,
    ratios: tuple[float, float, float],
) -> dict:
    ordered_categories = sorted(cleaned_df["category"].unique().tolist())
    return {
        "source_raw_path": str(raw_path),
        "raw_file_sha256": file_sha256(raw_path),
        "seed": seed,
        "ratios": {
            "train": ratios[0],
            "val": ratios[1],
            "test": ratios[2],
        },
        "total_row_count": int(len(cleaned_df)),
        "per_split_row_counts": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "per_class_counts": {
            "raw": category_counts(cleaned_df, ordered_categories),
            "train": category_counts(train_df, ordered_categories),
            "val": category_counts(val_df, ordered_categories),
            "test": category_counts(test_df, ordered_categories),
        },
    }


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_map: dict[str, int],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_df in (train_df, val_df, test_df):
        split_df["label_id"] = split_df["category"].map(label_map)

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    with (output_dir / "label_map.json").open("w", encoding="utf-8") as file_obj:
        json.dump(label_map, file_obj, indent=2, ensure_ascii=False)


def main() -> None:
    config = load_config()
    raw_path = config.paths.raw_data
    processed_dir = config.paths.processed_data
    ratios = (
        config.data.train_ratio,
        config.data.val_ratio,
        config.data.test_ratio,
    )

    print(f"Loading canonical raw dataset from {raw_path}")
    raw_df = load_raw(raw_path)
    cleaned_df, removal_stats = clean(raw_df)

    audit = build_raw_audit(raw_df, cleaned_df, raw_path, removal_stats)
    save_json(audit, raw_audit_path(config))
    print(f"Saved raw audit to {raw_audit_path(config)}")

    label_map = build_label_map(cleaned_df, config.data.categories)
    train_df, val_df, test_df = deterministic_split(
        cleaned_df,
        train_ratio=ratios[0],
        val_ratio=ratios[1],
        test_ratio=ratios[2],
        seed=config.project.seed,
    )

    validate_splits(cleaned_df, train_df, val_df, test_df, ratios)
    save_splits(train_df, val_df, test_df, label_map, processed_dir)

    manifest = build_split_manifest(
        cleaned_df,
        train_df,
        val_df,
        test_df,
        raw_path=raw_path,
        seed=config.project.seed,
        ratios=ratios,
    )
    save_json(manifest, split_manifest_path(config))

    print(f"Saved processed splits to {processed_dir}")
    print(f"Saved split manifest to {split_manifest_path(config)}")
    print(f"Label map: {label_map}")


if __name__ == "__main__":
    main()
