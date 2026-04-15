"""
Phase 1.5 — Synthetic Data Augmentation for Rare Classes
---------------------------------------------------------
Generates synthetic product-text examples for underrepresented dark pattern
categories using a similarity+composite prompting strategy (Kochanek et al., 2024).

Strategy
--------
- Similarity prompts  : seed each generation call with a real example from the
                        rare class so the LLM stays domain-grounded.
- Composite variation : vary product type across calls to avoid repetition.
- JSON output         : more reliable than Python lists (confirmed by paper).
- Post-filter         : run the saved BERT classifier on generated texts;
                        keep only examples predicted as the intended class
                        with softmax confidence >= CONFIDENCE_THRESHOLD.

Targets (chosen to balance rare classes without oversampling)
-------------------------------------------------------------
  Forced Action :  3  →  50  (generate ≈ 47)
  Sneaking      :  8  →  50  (generate ≈ 42)
  Obstruction   : 19  →  75  (generate ≈ 56)

Usage
-----
    python -m src.data.augment                        # uses qwen3:8b, bert validator
    python -m src.data.augment --model mistral-nemo:12b --validator roberta
    python -m src.data.augment --dry-run              # print prompts, skip LLM
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import typer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

from src.utils.ollama_client import chat_json

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "outputs" / "models"
RESULTS_DIR = ROOT / "results" / "augmentation"

# ---------------------------------------------------------------------------
# Augmentation config
# ---------------------------------------------------------------------------

# Classes to augment and their target counts in the training split
TARGETS: dict[str, int] = {
    "Forced Action": 50,
    "Sneaking": 50,
    "Obstruction": 75,
}

CONFIDENCE_THRESHOLD = 0.60  # minimum BERT softmax confidence to accept a generated example
EXAMPLES_PER_CALL = 3        # how many texts to request per LLM call
TEMPERATURE = 0.8            # higher temperature for output diversity (paper used 0.8)

# ---------------------------------------------------------------------------
# Category descriptions for the prompt (psychological mechanism + definition)
# ---------------------------------------------------------------------------
CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "Forced Action": (
        "Forced Action — the user is required to consent to marketing emails, "
        "terms of service, or data sharing as a condition of completing a purchase "
        "or registration, with no genuine opt-out."
    ),
    "Sneaking": (
        "Sneaking — hidden fees, automatically added items (e.g. insurance, "
        "memberships, charity donations), or costs that appear only at the final "
        "checkout step, not shown upfront."
    ),
    "Obstruction": (
        "Obstruction — cancelling a subscription or membership is made deliberately "
        "difficult: buried phone numbers, long hold times, vague instructions, "
        "or multi-step processes designed to frustrate the user into giving up."
    ),
}

# ---------------------------------------------------------------------------
# Composite variation: product types to rotate across calls
# ---------------------------------------------------------------------------
PRODUCT_TYPES: list[str] = [
    "online clothing retailer",
    "electronics store",
    "hotel booking platform",
    "streaming subscription service",
    "grocery delivery app",
    "beauty and cosmetics brand",
    "fitness and gym membership",
    "travel booking website",
    "software as a service (SaaS)",
    "food delivery platform",
    "pet supplies store",
    "home appliances retailer",
    "book subscription club",
    "meal kit delivery service",
    "online pharmacy",
    "music streaming platform",
    "gaming subscription service",
    "flower delivery website",
]


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert in dark patterns in e-commerce UX and persuasive design. "
    "You generate realistic synthetic product-page text snippets that exhibit a "
    "specific dark pattern. Output only valid JSON. Do not explain yourself."
)


def build_prompt(category: str, seed_example: str, product_type: str) -> str:
    description = CATEGORY_DESCRIPTIONS[category]
    return f"""\
Dark pattern category: {category}
Definition: {description}

Product context: {product_type}

Real example of this pattern:
\"\"\"{seed_example}\"\"\"

Generate {EXAMPLES_PER_CALL} new product-page text snippets that exhibit the same \
dark pattern. Each snippet should feel authentic, be about a {product_type}, and \
use different phrasing from the example. Keep each snippet under 60 words.

Return as JSON with this exact structure:
{{"examples": ["snippet 1", "snippet 2", "snippet 3"]}}"""


# ---------------------------------------------------------------------------
# BERT-based validator
# ---------------------------------------------------------------------------

class BERTValidator:
    """
    Wraps a saved HuggingFace classifier to filter generated examples.
    Only keeps examples predicted as the intended category with
    softmax confidence >= threshold.
    """

    def __init__(self, validator_key: str, label_map: dict[str, int], device: str):
        model_path = MODEL_DIR / validator_key
        if not model_path.exists():
            raise FileNotFoundError(
                f"No saved model at {model_path}. "
                f"Run `python -m src.baselines.train_classifier --model {validator_key}` first."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        self.model.eval()
        self.label_map = label_map
        self.device = device

    @torch.no_grad()
    def filter(
        self,
        texts: list[str],
        intended_category: str,
        threshold: float = CONFIDENCE_THRESHOLD,
    ) -> list[tuple[str, float]]:
        """
        Returns list of (text, confidence) for texts that pass the filter.
        """
        intended_id = self.label_map[intended_category]
        passed = []
        for text in texts:
            enc = self.tokenizer(
                text,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            probs = F.softmax(logits, dim=-1).squeeze()
            pred_id = probs.argmax().item()
            confidence = probs[intended_id].item()
            if pred_id == intended_id and confidence >= threshold:
                passed.append((text, round(confidence, 4)))
        return passed


# ---------------------------------------------------------------------------
# Core augmentation loop
# ---------------------------------------------------------------------------

def load_training_data() -> tuple[list[dict], dict[str, int]]:
    """Load train.csv and label_map.json."""
    rows = []
    with open(PROCESSED_DIR / "train.csv") as f:
        rows = list(csv.DictReader(f))

    with open(PROCESSED_DIR / "label_map.json") as f:
        label_map = json.load(f)

    return rows, label_map


def get_seeds(rows: list[dict], category: str) -> list[str]:
    return [r["text"] for r in rows if r["category"] == category]


def augment_category(
    category: str,
    seeds: list[str],
    target_count: int,
    current_count: int,
    validator: BERTValidator | None,
    model: str,
    dry_run: bool,
) -> list[str]:
    """
    Generate synthetic texts for one category until target_count is reached
    (or until we've made a reasonable number of attempts).

    Returns list of accepted synthetic texts.
    """
    needed = target_count - current_count
    if needed <= 0:
        print(f"  {category}: already at target ({current_count}). Skipping.")
        return []

    print(f"\n  {category}: need {needed} more (current={current_count}, target={target_count})")

    accepted: list[str] = []
    product_types = PRODUCT_TYPES.copy()
    random.shuffle(product_types)
    product_cycle = (product_types * ((needed // len(product_types)) + 2))

    max_calls = (needed // EXAMPLES_PER_CALL + 1) * 4  # allow 4x retries
    call_count = 0
    product_idx = 0

    with tqdm(total=needed, desc=f"    Generating {category}", unit="ex") as pbar:
        while len(accepted) < needed and call_count < max_calls:
            seed = random.choice(seeds)
            product_type = product_cycle[product_idx % len(product_cycle)]
            product_idx += 1
            call_count += 1

            prompt = build_prompt(category, seed, product_type)

            if dry_run:
                print(f"\n--- DRY RUN PROMPT (call {call_count}) ---\n{prompt}\n")
                accepted.extend([f"[dry-run example {i}]" for i in range(EXAMPLES_PER_CALL)])
                pbar.update(min(EXAMPLES_PER_CALL, needed - len(accepted) + EXAMPLES_PER_CALL))
                break

            try:
                result = chat_json(
                    prompt=prompt,
                    model=model,
                    system=SYSTEM_PROMPT,
                    temperature=TEMPERATURE,
                    max_retries=2,
                )
                candidates = result.get("examples", [])
                if not isinstance(candidates, list):
                    continue
                # Strip whitespace, drop empties
                candidates = [c.strip() for c in candidates if isinstance(c, str) and c.strip()]
            except (ValueError, KeyError):
                continue

            if validator is not None:
                passed = validator.filter(candidates, category)
                n_passed = len(passed)
                n_rejected = len(candidates) - n_passed
                if n_rejected:
                    tqdm.write(f"      [{category}] rejected {n_rejected}/{len(candidates)} by validator")
                candidates = [text for text, _ in passed]

            # Don't exceed target
            remaining = needed - len(accepted)
            candidates = candidates[:remaining]
            accepted.extend(candidates)
            pbar.update(len(candidates))

    print(f"    → Accepted {len(accepted)} synthetic examples for '{category}'")
    return accepted[:needed]


# ---------------------------------------------------------------------------
# Build augmented CSV rows
# ---------------------------------------------------------------------------

def build_synthetic_rows(
    category: str,
    texts: list[str],
    label_map: dict[str, int],
    start_page_id: int = 9000,
) -> list[dict]:
    label_id = label_map[category]
    rows = []
    for i, text in enumerate(texts):
        rows.append({
            "page_id": start_page_id + i,
            "text": text,
            "binary_label": 1,          # all dark pattern categories are label=1
            "category": category,
            "label_id": label_id,
            "synthetic": True,
        })
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer()


@app.command()
def main(
    model: str = typer.Option("qwen3:8b", help="Ollama model tag for generation"),
    validator: str = typer.Option("bert", help="Saved model key for filtering: bert | roberta"),
    no_validate: bool = typer.Option(False, help="Skip BERT validation step"),
    confidence: float = typer.Option(CONFIDENCE_THRESHOLD, help="Min BERT confidence to accept"),
    seed: int = typer.Option(42, help="Random seed"),
    dry_run: bool = typer.Option(False, help="Print prompts but skip LLM calls"),
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  LLM: {model}  |  Validator: {'none' if no_validate else validator}")

    # Load data
    rows, label_map = load_training_data()

    # Print current distribution
    from collections import Counter
    current_dist = Counter(r["category"] for r in rows)
    print("\nCurrent training distribution:")
    for cat, n in sorted(current_dist.items(), key=lambda x: x[1]):
        print(f"  {n:>5}  {cat}")

    # Load validator
    bert_validator = None
    if not no_validate and not dry_run:
        print(f"\nLoading validator: {validator}")
        bert_validator = BERTValidator(validator, label_map, device)

    # Augment each rare class
    all_synthetic_rows: list[dict] = []
    page_id_counter = 9000

    for category, target in TARGETS.items():
        seeds = get_seeds(rows, category)
        current_count = current_dist[category]

        new_texts = augment_category(
            category=category,
            seeds=seeds,
            target_count=target,
            current_count=current_count,
            validator=bert_validator,
            model=model,
            dry_run=dry_run,
        )

        synthetic_rows = build_synthetic_rows(category, new_texts, label_map, page_id_counter)
        all_synthetic_rows.extend(synthetic_rows)
        page_id_counter += len(synthetic_rows) + 1

    if dry_run:
        print("\nDry run complete. No files written.")
        return

    # Save augmented train.csv (original + synthetic)
    original_fieldnames = ["page_id", "text", "binary_label", "category", "label_id"]
    augmented_rows = rows + [
        {k: r[k] for k in original_fieldnames} for r in all_synthetic_rows
    ]

    out_path = PROCESSED_DIR / "train_augmented.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=original_fieldnames)
        writer.writeheader()
        writer.writerows(augmented_rows)

    # Save synthetic-only log (with confidence info)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / "synthetic_examples.json"
    log: dict[str, list[str]] = {}
    for cat in TARGETS:
        log[cat] = [r["text"] for r in all_synthetic_rows if r["category"] == cat]
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    # Print final distribution
    final_dist = Counter(r["category"] for r in augmented_rows)
    print("\nAugmented training distribution:")
    for cat, n in sorted(final_dist.items(), key=lambda x: x[1]):
        original = current_dist.get(cat, 0)
        added = n - original
        flag = f"  (+{added} synthetic)" if added > 0 else ""
        print(f"  {n:>5}  {cat}{flag}")

    print(f"\nAugmented train split saved → {out_path}")
    print(f"Synthetic examples log     → {log_path}")
    print(f"\nTotal: {len(augmented_rows)} (was {len(rows)})")

    # Remind user of the two-pass workflow
    print("\nNext steps:")
    print("  1. Re-train BERT on the augmented split:")
    print("     python -m src.baselines.train_classifier --model bert --train-file train_augmented")
    print("  2. (Optional) Re-run augmentation with the retrained model as validator")
    print("     to filter out any low-quality synthetic examples:")
    print("     python -m src.data.augment --validator bert")
    print("     (This requires the retrained checkpoint at outputs/models/bert_train_augmented/)")


if __name__ == "__main__":
    app()
