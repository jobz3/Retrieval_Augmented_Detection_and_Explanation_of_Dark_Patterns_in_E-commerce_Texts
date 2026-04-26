"""
Layer 3 — LLM-as-judge rationale quality evaluation.

Judge model: prometheus-eval/prometheus-7b-v2.0 (HuggingFace, loaded via transformers).
Prometheus-2 is a purpose-built eval model trained to score responses on user-defined
rubric criteria, making it architecture-independent from Qwen3 (the pipeline model).

Since we have no gold reference rationales, we use Prometheus in reference-free mode:
the rubric criteria stand in for the reference, and the judge scores based on them alone.

Four criteria scored [1–5] each → composite [4–20]:
  1. coherence       — reasoning_steps logically lead to the label
  2. faithfulness    — all claims grounded in input text, no hallucination
  3. specificity     — cites concrete phrases from the text, not generic descriptions
  4. non_circularity — explains WHY the pattern is manipulative, not just restates the label

Hardware note: prometheus-7b-v2.0 is ~14 GB in fp16. On this machine (RTX 5060 8 GB):
  - Load with load_in_4bit=True (bitsandbytes) to fit in ~5 GB VRAM
  - Falls back to CPU if 4-bit loading fails

Usage:
    python -m src.evaluation.rationale_judge --file results/pipelines/rag_sbert_diversity_k5_cot.jsonl
    python -m src.evaluation.rationale_judge --file results/pipelines/rag_sbert_diversity_k5_cot.jsonl --limit 50
    python -m src.evaluation.rationale_judge --file results/pipelines/rag_sbert_diversity_k5_cot.jsonl --cpu
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

ROOT        = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "evaluation"

JUDGE_MODEL = "prometheus-eval/prometheus-7b-v2.0"

# ---------------------------------------------------------------------------
# Rubric definition (reference-free mode — no gold rationale needed)
# ---------------------------------------------------------------------------

_CRITERIA = {
    "coherence": (
        "Coherence",
        "Do the reasoning_steps logically lead to the predicted label? "
        "Is the rationale internally consistent with the steps?"
    ),
    "faithfulness": (
        "Faithfulness",
        "Are all claims in the rationale grounded in the input text? "
        "The rationale must not introduce evidence not present in the product text."
    ),
    "specificity": (
        "Specificity",
        "Does the rationale cite concrete words or phrases from the input text "
        "rather than making generic statements about the label category?"
    ),
    "non_circularity": (
        "Non-Circularity",
        "Does the rationale explain WHY the tactic is manipulative to the consumer, "
        "rather than simply restating what the label name means?"
    ),
}

# Prometheus absolute scoring rubric template (reference-free)
_RUBRIC_TEMPLATE = """###Task Description:
An instruction (might include an Input inside it) and a response are given.
Evaluate the quality of the response using the criterion below.
Score the response on a scale of 1 to 5.

###Instruction:
Analyse the following product page text and identify any dark pattern present.
Provide reasoning steps and a rationale explaining your label choice.

Product text: \"\"\"{input_text}\"\"\"

###Response to evaluate:
Label: {label}

Reasoning steps:
{reasoning_steps}

Rationale: {rationale}

###Criterion:
{criterion_name}: {criterion_description}

###Score Rubric:
Score 1: The response completely fails to satisfy this criterion.
Score 2: The response partially satisfies this criterion with significant gaps.
Score 3: The response satisfies this criterion moderately well.
Score 4: The response satisfies this criterion well with only minor issues.
Score 5: The response fully satisfies this criterion without any issues.

###Feedback:"""


# ---------------------------------------------------------------------------
# Model loader (singleton)
# ---------------------------------------------------------------------------

_model   = None
_tokenizer = None


def _load_model(use_cpu: bool = False):
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"  Loading {JUDGE_MODEL}...")
    _tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)

    if use_cpu:
        _model = AutoModelForCausalLM.from_pretrained(
            JUDGE_MODEL,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
    else:
        try:
            from transformers import BitsAndBytesConfig
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            _model = AutoModelForCausalLM.from_pretrained(
                JUDGE_MODEL,
                quantization_config=bnb_cfg,
                device_map="auto",
            )
            print("  Loaded in 4-bit (bitsandbytes)")
        except Exception as e:
            print(f"  4-bit load failed ({e}), falling back to CPU fp32")
            _model = AutoModelForCausalLM.from_pretrained(
                JUDGE_MODEL,
                torch_dtype=torch.float32,
                device_map="cpu",
            )

    _model.eval()
    return _model, _tokenizer


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _extract_score(feedback_text: str) -> int:
    """
    Parse the integer score [1-5] from Prometheus feedback output.
    Prometheus appends '[RESULT] <score>' at the end of its feedback.
    Falls back to searching for the last standalone digit 1-5.
    """
    # Primary: Prometheus convention "[RESULT] N"
    m = re.search(r"\[RESULT\]\s*([1-5])", feedback_text)
    if m:
        return int(m.group(1))

    # Fallback: last occurrence of a standalone digit 1-5
    digits = re.findall(r"\b([1-5])\b", feedback_text)
    if digits:
        return int(digits[-1])

    return 1  # conservative floor if parsing fails


def _score_criterion(
    record: dict,
    criterion_key: str,
    model,
    tokenizer,
    max_new_tokens: int = 256,
) -> tuple[int, str]:
    """
    Run Prometheus on one criterion for one record.
    Returns (score, feedback_text).
    """
    import torch

    criterion_name, criterion_desc = _CRITERIA[criterion_key]

    steps = record.get("reasoning_steps", [])
    steps_str = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps)) if steps else "  (none)"

    prompt = _RUBRIC_TEMPLATE.format(
        input_text=record.get("input_text", ""),
        label=record.get("label", ""),
        reasoning_steps=steps_str,
        rationale=record.get("rationale", ""),
        criterion_name=criterion_name,
        criterion_description=criterion_desc,
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    feedback = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    score = _extract_score(feedback)
    return score, feedback


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_record(record: dict, model, tokenizer) -> dict:
    """
    Score a single prediction record across all four criteria.

    Returns a dict with per-criterion scores, composite, and feedback snippets.
    """
    scores: dict = {}
    feedbacks: dict = {}

    for key in _CRITERIA:
        try:
            sc, fb = _score_criterion(record, key, model, tokenizer)
        except Exception as exc:
            sc, fb = 1, f"error: {exc}"
        scores[key]    = sc
        feedbacks[key] = fb

    composite = sum(scores[k] for k in _CRITERIA)
    return {
        **scores,
        "composite":  composite,
        "feedbacks":  feedbacks,
        "judge_model": JUDGE_MODEL,
    }


def judge_file(
    path: Path,
    limit: int | None = None,
    use_cpu: bool = False,
) -> dict:
    """
    Score all records in a JSONL file.

    Returns a summary dict with per-record scores and aggregate statistics.
    """
    with open(path) as f:
        records = [json.loads(l) for l in f if l.strip()]

    if limit:
        records = records[:limit]

    n = len(records)
    print(f"Judging {n} records from {path.name}")
    print(f"  Judge: {JUDGE_MODEL}")

    model, tokenizer = _load_model(use_cpu=use_cpu)

    scored: list[dict] = []
    composites: list[int] = []

    for i, rec in enumerate(records):
        print(f"  [{i+1}/{n}] id={rec.get('id', i)}...", end="\r", flush=True)
        result = score_record(rec, model, tokenizer)
        composites.append(result["composite"])
        scored.append({
            "id":         rec.get("id", i),
            "label":      rec.get("label", ""),
            "gold_label": rec.get("gold_label", ""),
            "scores":     result,
        })

    print()

    summary: dict = {
        "file":          path.name,
        "n":             n,
        "judge_model":   JUDGE_MODEL,
        "mean_composite": round(statistics.mean(composites), 3),
        "std_composite":  round(statistics.stdev(composites) if n > 1 else 0.0, 3),
        "records":       scored,
    }

    for key in _CRITERIA:
        vals = [s["scores"][key] for s in scored]
        summary[f"mean_{key}"] = round(statistics.mean(vals), 3)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Prometheus-2 rationale quality judge")
    parser.add_argument("--file",  required=True,           help="Pipeline JSONL output path")
    parser.add_argument("--limit", type=int, default=None,  help="Score only first N records")
    parser.add_argument("--cpu",   action="store_true",     help="Force CPU inference (slow but no VRAM needed)")
    parser.add_argument("--out",   default=None,            help="Output JSON path")
    args = parser.parse_args()

    in_path = Path(args.file)
    summary = judge_file(in_path, limit=args.limit, use_cpu=args.cpu)

    print(f"\nComposite (mean): {summary['mean_composite']:.3f}/20  (std={summary['std_composite']:.3f})")
    for key in _CRITERIA:
        print(f"  {key:<18}: {summary[f'mean_{key}']:.3f}/5")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else RESULTS_DIR / f"rationale_judge_{in_path.stem}.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
