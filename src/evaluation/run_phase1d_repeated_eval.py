"""
Phase 1d supplementary repeated-stratified evaluation for RoBERTa plain vs weighted CE.

This runner is intentionally separate from the frozen Phase 1 / 1b / 1c artifacts.
It reconstructs the full normalized Phase 1 dataset by concatenating the frozen locked
split CSVs, then performs repeated stratified outer splits and a class-preserving inner
validation split derived from the outer training fold only.

Outputs are written only to:
    results/phase1d_repeated_eval/
    results/analysis/phase1d_repeated_eval/
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from sklearn.model_selection import RepeatedStratifiedKFold

from src.utils.config import load_config


CONFIG = load_config()
ROOT = Path(__file__).resolve().parents[2]
LOCKED_DIR = CONFIG.paths.locked_processed_data
RESULTS_ROOT = ROOT / "results" / "phase1d_repeated_eval"
ANALYSIS_ROOT = ROOT / "results" / "analysis" / "phase1d_repeated_eval"
MINORITY_THRESHOLD = CONFIG.phase1b.minority_train_threshold

FULL_DATASET_NAME = "phase1_normalized_full_dataset"
VARIANTS = ("plain", "weighted_ce")
N_SPLITS = 3
N_REPEATS = 3
INNER_VAL_RATIO = CONFIG.data.val_ratio / (CONFIG.data.train_ratio + CONFIG.data.val_ratio)

app = typer.Typer()


@dataclass(frozen=True)
class RunSpec:
    repeat_idx: int
    fold_idx: int
    run_idx: int
    seed: int

    @property
    def run_name(self) -> str:
        return f"repeat_{self.repeat_idx}_fold_{self.fold_idx}"


def repo_relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_full_phase1_dataset() -> tuple[pd.DataFrame, dict[str, int]]:
    parts = []
    for split in ("train", "val", "test"):
        split_df = pd.read_csv(LOCKED_DIR / f"{split}.csv")
        split_df["source_split"] = split
        parts.append(split_df)

    full_df = pd.concat(parts, ignore_index=True)
    full_df["label_id"] = full_df["label_id"].astype(int)

    with (LOCKED_DIR / "label_map.json").open(encoding="utf-8") as handle:
        label_map = json.load(handle)

    expected_total = 2356
    if len(full_df) != expected_total:
        raise ValueError(f"Expected {expected_total} rows from locked split union, got {len(full_df)}.")

    expected_labels = set(label_map.values())
    actual_labels = set(full_df["label_id"].unique().tolist())
    if actual_labels != expected_labels:
        raise ValueError(f"Label id mismatch. Expected {expected_labels}, got {actual_labels}.")

    return full_df, label_map


def label_names_from_map(label_map: dict[str, int]) -> list[str]:
    id2label = {idx: label for label, idx in label_map.items()}
    return [id2label[idx] for idx in range(len(id2label))]


def validate_class_presence(df: pd.DataFrame, split_name: str, label_names: list[str]) -> None:
    counts = df["label_id"].value_counts().to_dict()
    missing = [label_names[idx] for idx in range(len(label_names)) if counts.get(idx, 0) <= 0]
    if missing:
        raise ValueError(
            f"{split_name} split is missing classes: {missing}. "
            f"Counts: {df['label_id'].value_counts().sort_index().to_dict()}"
        )


def make_inner_train_val_split(
    outer_train_df: pd.DataFrame,
    label_names: list[str],
    seed: int,
    val_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    target_val_size = max(len(label_names), int(round(len(outer_train_df) * val_ratio)))
    max_feasible_val_size = len(outer_train_df) - len(label_names)
    if max_feasible_val_size < len(label_names):
        raise ValueError(
            "Outer train fold is too small to keep every class in both train and val. "
            f"outer_train_size={len(outer_train_df)} classes={len(label_names)}"
        )
    target_val_size = min(target_val_size, max_feasible_val_size)

    by_class: dict[int, list[int]] = {}
    for label_id in range(len(label_names)):
        indices = outer_train_df.index[outer_train_df["label_id"] == label_id].tolist()
        if len(indices) < 2:
            raise ValueError(
                "Inner train/val split is infeasible because outer train has fewer than 2 "
                f"examples for class '{label_names[label_id]}'. Counts: "
                f"{outer_train_df['label_id'].value_counts().sort_index().to_dict()}"
            )
        rng.shuffle(indices)
        by_class[label_id] = indices

    val_indices: list[int] = []
    extra_pool: dict[int, list[int]] = {}
    for label_id, indices in by_class.items():
        val_indices.append(indices[0])
        remaining = indices[1:]
        extra_pool[label_id] = remaining[:-1] if len(remaining) > 1 else []
        rng.shuffle(extra_pool[label_id])

    additional_needed = target_val_size - len(label_names)
    while additional_needed > 0:
        eligible = [label_id for label_id, items in extra_pool.items() if items]
        if not eligible:
            raise ValueError(
                "Unable to allocate the requested validation size while preserving every "
                "class in train and val."
            )
        weights = np.array([len(extra_pool[label_id]) for label_id in eligible], dtype=float)
        weights /= weights.sum()
        chosen_label = int(rng.choice(eligible, p=weights))
        val_indices.append(extra_pool[chosen_label].pop())
        additional_needed -= 1

    val_index_set = set(val_indices)
    train_indices = [idx for idx in outer_train_df.index.tolist() if idx not in val_index_set]

    train_df = outer_train_df.loc[train_indices].copy().reset_index(drop=True)
    val_df = outer_train_df.loc[val_indices].copy().reset_index(drop=True)

    validate_class_presence(train_df, "inner-train", label_names)
    validate_class_presence(val_df, "inner-val", label_names)
    return train_df, val_df


def write_split_dir(
    split_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_map: dict[str, int],
    run_spec: RunSpec,
) -> None:
    if split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(split_dir / "train.csv", index=False)
    val_df.to_csv(split_dir / "val.csv", index=False)
    test_df.to_csv(split_dir / "test.csv", index=False)

    with (split_dir / "label_map.json").open("w", encoding="utf-8") as handle:
        json.dump(label_map, handle, indent=2)

    metadata = {
        "dataset_source": FULL_DATASET_NAME,
        "run_name": run_spec.run_name,
        "repeat_idx": run_spec.repeat_idx,
        "fold_idx": run_spec.fold_idx,
        "seed": run_spec.seed,
        "inner_val_ratio": INNER_VAL_RATIO,
        "counts": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
    }
    with (split_dir / "split_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def load_metrics_from_results(
    results_path: Path,
    train_df: pd.DataFrame,
    label_names: list[str],
    variant: str,
    run_spec: RunSpec,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    with results_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    report = payload["classification_report"]
    train_counts = train_df["category"].value_counts().to_dict()
    minority_classes = [
        label for label in label_names if train_counts.get(label, 0) < MINORITY_THRESHOLD
    ]
    minority_avg_f1 = float(np.mean([report[label]["f1-score"] for label in minority_classes]))

    run_row: dict[str, object] = {
        "variant": variant,
        "repeat": run_spec.repeat_idx,
        "fold": run_spec.fold_idx,
        "run_name": run_spec.run_name,
        "seed": run_spec.seed,
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "minority_avg_f1": minority_avg_f1,
        "Forced Action F1": float(report["Forced Action"]["f1-score"]),
        "Sneaking F1": float(report["Sneaking"]["f1-score"]),
        "results_path": repo_relative(results_path),
    }

    per_class_rows: list[dict[str, object]] = []
    for label in label_names:
        per_class_rows.append(
            {
                "variant": variant,
                "repeat": run_spec.repeat_idx,
                "fold": run_spec.fold_idx,
                "run_name": run_spec.run_name,
                "class": label,
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": float(report[label]["support"]),
            }
        )

    return run_row, per_class_rows


def save_normalized_confusion_matrix(
    results_path: Path,
    label_names: list[str],
    output_path: Path,
) -> None:
    with results_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    cm = np.array(payload["confusion_matrix"], dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)
    cm_df = pd.DataFrame(normalized, index=label_names, columns=label_names)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm_df.to_csv(output_path)


def run_trainer(
    variant: str,
    split_dir: Path,
    checkpoint_dir: Path,
    results_path: Path,
    seed: int,
    overwrite_existing: bool,
) -> None:
    if results_path.exists() and checkpoint_dir.exists() and not overwrite_existing:
        print(f"Skipping existing {variant} run at {repo_relative(results_path)}")
        return

    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    module = (
        "src.baselines.train_classifier"
        if variant == "plain"
        else "src.baselines.train_classifier_weighted"
    )
    command = [
        sys.executable,
        "-m",
        module,
        "--model",
        "roberta",
        "--seed",
        str(seed),
        "--processed-dir",
        repo_relative(split_dir),
        "--save-dir",
        repo_relative(checkpoint_dir),
        "--results-path",
        repo_relative(results_path),
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def make_boxplot(df: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    sns.boxplot(data=df, x="variant", y=metric, order=list(VARIANTS))
    sns.stripplot(
        data=df,
        x="variant",
        y=metric,
        order=list(VARIANTS),
        color="black",
        alpha=0.6,
        size=4,
    )
    plt.title(title)
    plt.xlabel("Variant")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def verdict_from_summary(summary_df: pd.DataFrame, run_df: pd.DataFrame) -> str:
    plain = summary_df.loc[summary_df["variant"] == "plain"].iloc[0]
    weighted = summary_df.loc[summary_df["variant"] == "weighted_ce"].iloc[0]

    wins_macro = int(
        (
            run_df.pivot(index="run_name", columns="variant", values="macro_f1")["weighted_ce"]
            > run_df.pivot(index="run_name", columns="variant", values="macro_f1")["plain"]
        ).sum()
    )
    wins_minority = int(
        (
            run_df.pivot(index="run_name", columns="variant", values="minority_avg_f1")["weighted_ce"]
            > run_df.pivot(index="run_name", columns="variant", values="minority_avg_f1")["plain"]
        ).sum()
    )

    if (
        weighted["macro_f1_mean"] > plain["macro_f1_mean"]
        and weighted["minority_avg_f1_mean"] > plain["minority_avg_f1_mean"]
        and wins_macro >= 7
        and wins_minority >= 7
    ):
        return "DIRECTIONALLY_STABLE"
    if (
        weighted["macro_f1_mean"] > plain["macro_f1_mean"]
        and weighted["minority_avg_f1_mean"] >= plain["minority_avg_f1_mean"]
    ):
        return "FRAGILE_BUT_PROMISING"
    return "INCONCLUSIVE"


def write_aggregate_markdown(summary_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 1d Aggregate Summary",
        "",
        "| Variant | Accuracy Mean | Accuracy Std | Macro F1 Mean | Macro F1 Std | Weighted F1 Mean | Weighted F1 Std | Minority Avg F1 Mean | Minority Avg F1 Std |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_df.itertuples(index=False):
        lines.append(
            f"| {row.variant} | {row.accuracy_mean:.4f} | {row.accuracy_std:.4f} | "
            f"{row.macro_f1_mean:.4f} | {row.macro_f1_std:.4f} | "
            f"{row.weighted_f1_mean:.4f} | {row.weighted_f1_std:.4f} | "
            f"{row.minority_avg_f1_mean:.4f} | {row.minority_avg_f1_std:.4f} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_stability_note(
    output_path: Path,
    verdict: str,
    summary_df: pd.DataFrame,
    run_df: pd.DataFrame,
) -> None:
    plain = summary_df.loc[summary_df["variant"] == "plain"].iloc[0]
    weighted = summary_df.loc[summary_df["variant"] == "weighted_ce"].iloc[0]
    wins_macro = int(
        (
            run_df.pivot(index="run_name", columns="variant", values="macro_f1")["weighted_ce"]
            > run_df.pivot(index="run_name", columns="variant", values="macro_f1")["plain"]
        ).sum()
    )
    wins_fa = int(
        (
            run_df.pivot(index="run_name", columns="variant", values="Forced Action F1")["weighted_ce"]
            > run_df.pivot(index="run_name", columns="variant", values="Forced Action F1")["plain"]
        ).sum()
    )

    text = f"""# Phase 1d Stability Note

Verdict: `{verdict}`

This repeated-stratified evaluation is supplementary only. The locked split remains the primary benchmark.

Key numbers:

- plain macro F1 mean/std: {plain['macro_f1_mean']:.4f} / {plain['macro_f1_std']:.4f}
- weighted CE macro F1 mean/std: {weighted['macro_f1_mean']:.4f} / {weighted['macro_f1_std']:.4f}
- plain minority_avg_f1 mean/std: {plain['minority_avg_f1_mean']:.4f} / {plain['minority_avg_f1_std']:.4f}
- weighted CE minority_avg_f1 mean/std: {weighted['minority_avg_f1_mean']:.4f} / {weighted['minority_avg_f1_std']:.4f}
- weighted CE beat plain on macro F1 in {wins_macro} of {len(run_df) // 2} paired runs
- weighted CE beat plain on Forced Action F1 in {wins_fa} of {len(run_df) // 2} paired runs

Interpretation constraints:

- Forced Action has only 4 total examples in the full dataset.
- Sneaking has only 12 total examples in the full dataset.
- Repeated stratification helps estimate instability, but it does not make the data sufficient.
- Any minority-class conclusion remains fragile, especially for Forced Action and Sneaking.
"""
    output_path.write_text(text, encoding="utf-8")


def aggregate_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    grouped = run_df.groupby("variant").agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        weighted_f1_mean=("weighted_f1", "mean"),
        weighted_f1_std=("weighted_f1", "std"),
        minority_avg_f1_mean=("minority_avg_f1", "mean"),
        minority_avg_f1_std=("minority_avg_f1", "std"),
        forced_action_f1_mean=("Forced Action F1", "mean"),
        forced_action_f1_std=("Forced Action F1", "std"),
        sneaking_f1_mean=("Sneaking F1", "mean"),
        sneaking_f1_std=("Sneaking F1", "std"),
    )
    return grouped.reset_index()


def per_class_summary(per_class_df: pd.DataFrame) -> pd.DataFrame:
    grouped = per_class_df.groupby(["variant", "class"]).agg(
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        support_mean=("support", "mean"),
        support_std=("support", "std"),
    )
    return grouped.reset_index()


@app.command()
def main(
    n_splits: int = typer.Option(N_SPLITS, help="Number of stratified folds"),
    n_repeats: int = typer.Option(N_REPEATS, help="Number of repeats"),
    seed: int = typer.Option(CONFIG.project.seed, help="Base random seed"),
    overwrite_existing: bool = typer.Option(
        False,
        help="Overwrite existing Phase 1d checkpoints and results",
    ),
) -> None:
    if n_splits != 3 or n_repeats != 3:
        print(
            "Warning: Phase 1d defaults are 3 folds x 3 repeats. "
            "You are running a non-default protocol."
        )

    full_df, label_map = load_full_phase1_dataset()
    label_names = label_names_from_map(label_map)
    validate_class_presence(full_df, "full-dataset", label_names)

    results_runs_dir = RESULTS_ROOT / "runs"
    split_root = RESULTS_ROOT / "splits"
    checkpoint_root = RESULTS_ROOT / "checkpoints"
    cm_root = ANALYSIS_ROOT / "confusion_matrices"
    for path in (RESULTS_ROOT, ANALYSIS_ROOT, results_runs_dir, split_root, checkpoint_root, cm_root):
        path.mkdir(parents=True, exist_ok=True)

    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=seed,
    )

    run_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []

    for run_idx, (outer_train_idx, test_idx) in enumerate(
        splitter.split(np.zeros(len(full_df)), full_df["label_id"].to_numpy()),
        start=1,
    ):
        repeat_idx = math.ceil(run_idx / n_splits)
        fold_idx = ((run_idx - 1) % n_splits) + 1
        run_seed = seed + repeat_idx * 100 + fold_idx
        run_spec = RunSpec(repeat_idx=repeat_idx, fold_idx=fold_idx, run_idx=run_idx, seed=run_seed)

        outer_train_df = full_df.iloc[outer_train_idx].copy().reset_index(drop=True)
        test_df = full_df.iloc[test_idx].copy().reset_index(drop=True)
        validate_class_presence(test_df, f"{run_spec.run_name}-test", label_names)

        train_df, val_df = make_inner_train_val_split(
            outer_train_df,
            label_names=label_names,
            seed=run_seed,
            val_ratio=INNER_VAL_RATIO,
        )

        split_dir = split_root / run_spec.run_name
        write_split_dir(split_dir, train_df, val_df, test_df, label_map, run_spec)

        for variant in VARIANTS:
            checkpoint_dir = checkpoint_root / variant / run_spec.run_name
            results_path = results_runs_dir / f"roberta_{variant}_{run_spec.run_name}.json"

            print(f"\n=== Running {variant} / {run_spec.run_name} ===")
            run_trainer(
                variant=variant,
                split_dir=split_dir,
                checkpoint_dir=checkpoint_dir,
                results_path=results_path,
                seed=run_seed,
                overwrite_existing=overwrite_existing,
            )

            run_row, class_rows = load_metrics_from_results(
                results_path=results_path,
                train_df=train_df,
                label_names=label_names,
                variant=variant,
                run_spec=run_spec,
            )
            run_rows.append(run_row)
            per_class_rows.extend(class_rows)

            cm_output = cm_root / f"confusion_matrix_roberta_{variant}_{run_spec.run_name}_test_norm.csv"
            save_normalized_confusion_matrix(results_path, label_names, cm_output)

    run_df = pd.DataFrame(run_rows).sort_values(["repeat", "fold", "variant"]).reset_index(drop=True)
    per_class_df = pd.DataFrame(per_class_rows).sort_values(
        ["variant", "class", "repeat", "fold"]
    ).reset_index(drop=True)
    summary_df = aggregate_summary(run_df)
    per_class_summary_df = per_class_summary(per_class_df)

    run_level_path = ANALYSIS_ROOT / "run_level_metrics.csv"
    aggregate_csv_path = ANALYSIS_ROOT / "aggregate_summary.csv"
    aggregate_md_path = ANALYSIS_ROOT / "aggregate_summary.md"
    per_class_summary_path = ANALYSIS_ROOT / "per_class_summary.csv"
    forced_action_path = ANALYSIS_ROOT / "forced_action_f1_run_table.csv"
    sneaking_path = ANALYSIS_ROOT / "sneaking_f1_run_table.csv"
    macro_boxplot_path = ANALYSIS_ROOT / "macro_f1_boxplot.png"
    minority_boxplot_path = ANALYSIS_ROOT / "minority_avg_f1_boxplot.png"
    stability_note_path = ANALYSIS_ROOT / "stability_note.md"

    run_df.to_csv(run_level_path, index=False)
    summary_df.to_csv(aggregate_csv_path, index=False)
    write_aggregate_markdown(summary_df, aggregate_md_path)
    per_class_summary_df.to_csv(per_class_summary_path, index=False)
    run_df[
        ["variant", "repeat", "fold", "run_name", "Forced Action F1"]
    ].to_csv(forced_action_path, index=False)
    run_df[
        ["variant", "repeat", "fold", "run_name", "Sneaking F1"]
    ].to_csv(sneaking_path, index=False)

    make_boxplot(run_df, "macro_f1", macro_boxplot_path, "Phase 1d Macro F1 Distribution")
    make_boxplot(
        run_df,
        "minority_avg_f1",
        minority_boxplot_path,
        "Phase 1d Minority Avg F1 Distribution",
    )

    verdict = verdict_from_summary(summary_df, run_df)
    write_stability_note(stability_note_path, verdict, summary_df, run_df)

    print("\nPhase 1d repeated evaluation complete.")
    print(f"Run-level metrics: {repo_relative(run_level_path)}")
    print(f"Aggregate summary: {repo_relative(aggregate_csv_path)}")
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    app()
