"""
Proportion Ablation: How much synthetic data is needed?
--------------------------------------------------------
Trains BERT at 4 proportions of synthetic data for rare classes
(0%, 33%, 67%, 100%) and evaluates each on test_v2.

If F1 keeps rising linearly up to 100%, that suggests the model is
memorising synthetic distribution rather than learning the pattern.
A curve that saturates early (e.g. good at 33%, flat after) indicates
genuine generalisation.

All runs use the same val_v2 / test_v2 for a fair comparison.

Usage:
    python -m src.evaluation.proportion_ablation
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
SYNTHETIC_LOG = ROOT / "results" / "augmentation" / "synthetic_examples.json"
RESULTS_DIR   = ROOT / "results" / "evaluation"

RARE_CLASSES  = ["Forced Action", "Sneaking", "Obstruction"]
PROPORTIONS   = [0.0, 0.33, 0.67, 1.0]
SEED          = 42


def load_real_train() -> list[dict]:
    rows = []
    with open(PROCESSED_DIR / "train.csv") as f:
        rows = list(csv.DictReader(f))
    return rows


def load_synthetic() -> dict[str, list[str]]:
    with open(SYNTHETIC_LOG) as f:
        return json.load(f)


def load_label_map() -> dict[str, int]:
    with open(PROCESSED_DIR / "label_map.json") as f:
        return json.load(f)


def build_split(real_rows: list[dict], synthetic: dict[str, list[str]],
                label_map: dict[str, int], proportion: float,
                rng: random.Random) -> list[dict]:
    """
    Build a training split mixing real examples with `proportion` of
    synthetic examples for each rare class.
    """
    rows = list(real_rows)  # all real training examples

    for cat, texts in synthetic.items():
        if cat not in RARE_CLASSES:
            continue
        n = max(1, round(len(texts) * proportion)) if proportion > 0 else 0
        selected = rng.sample(texts, n) if n <= len(texts) else texts
        for i, text in enumerate(selected):
            rows.append({
                "page_id":      9000 + i,
                "text":         text,
                "binary_label": 1,
                "category":     cat,
                "label_id":     label_map[cat],
            })

    rng.shuffle(rows)
    return rows


def save_split(rows: list[dict], path: Path) -> None:
    fieldnames = ["page_id", "text", "binary_label", "category", "label_id"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def run_training(split_stem: str) -> None:
    """Trigger BERT training for one split stem via subprocess."""
    import subprocess, sys
    python = sys.executable
    cmd = [
        python, "-m", "src.baselines.train_classifier",
        "--model", "bert",
        "--train-file", split_stem,
        "--val-file", "val_v2",
        "--test-file", "test_v2",
    ]
    print(f"\n  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  ⚠ Training failed for {split_stem}")


def collect_results() -> list[dict]:
    """Load F1 results for all proportion runs."""
    stems = [prop_to_stem(p) for p in PROPORTIONS]
    rows = []
    for p, stem in zip(PROPORTIONS, stems):
        path = ROOT / "results" / "baselines" / f"bert_{stem}.json"
        if not path.exists():
            print(f"  Missing: {path}")
            continue
        with open(path) as f:
            data = json.load(f)
        cr = data["classification_report"]
        entry = {
            "proportion": p,
            "stem":       stem,
            "test_macro_f1": data["test_macro_f1"],
            "val_macro_f1":  data["best_val_macro_f1"],
        }
        for cat in RARE_CLASSES + ["macro avg"]:
            c = cr.get(cat, {})
            entry[f"f1_{cat}"]      = round(c.get("f1-score", 0.0), 4)
            entry[f"support_{cat}"] = int(c.get("support", 0))
        rows.append(entry)
    return rows


def prop_to_stem(p: float) -> str:
    if p == 0.0:
        return "train_v2_p0"
    pct = int(round(p * 100))
    return f"train_v2_p{pct}"


def print_results(rows: list[dict]) -> None:
    if not rows:
        print("No results to display.")
        return

    cats = RARE_CLASSES + ["macro avg"]
    header = f"{'Proportion':>12} {'macro F1':>10}"
    for cat in RARE_CLASSES:
        short = cat.split()[0][:7]
        header += f"  {short:>8}"
    print("\n" + "=" * 65)
    print("PROPORTION ABLATION RESULTS (test_v2)")
    print("=" * 65)
    print(header)
    print("-" * 65)
    for r in rows:
        line = f"{r['proportion']:>11.0%} {r['test_macro_f1']:>10.4f}"
        for cat in RARE_CLASSES:
            line += f"  {r[f'f1_{cat}']:>8.4f}"
        print(line)
    print("-" * 65)
    print("\nInterpretation:")
    print("  F1 saturates early (e.g. 33%→67% flat)  → genuine generalisation")
    print("  F1 rises linearly to 100%                → possible memorisation")


def main() -> None:
    rng = random.Random(SEED)
    real_rows = load_real_train()
    synthetic = load_synthetic()
    label_map = load_label_map()

    print("Building proportion splits ...")
    for p in PROPORTIONS:
        stem = prop_to_stem(p)
        path = PROCESSED_DIR / f"{stem}.csv"
        if path.exists():
            print(f"  {stem}.csv already exists — skipping")
            continue
        rows = build_split(real_rows, synthetic, label_map, p, rng)
        save_split(rows, path)
        n_synth = sum(1 for r in rows if int(r["page_id"]) >= 9000)
        print(f"  {stem}.csv  total={len(rows)}  synthetic={n_synth}")

    print("\nTraining BERT for each proportion (this takes ~5 min each) ...")
    for p in PROPORTIONS:
        stem = prop_to_stem(p)
        result_path = ROOT / "results" / "baselines" / f"bert_{stem}.json"
        if result_path.exists():
            print(f"  bert_{stem}.json already exists — skipping training")
            continue
        run_training(stem)

    rows = collect_results()
    print_results(rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "proportion_ablation.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
