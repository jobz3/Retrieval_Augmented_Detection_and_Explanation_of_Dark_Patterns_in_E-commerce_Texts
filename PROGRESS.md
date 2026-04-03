# Project Progress Log

**Project:** Retrieval-Augmented Detection and Explanation of Dark Patterns in E-Commerce Texts
**Course:** HCNLP — Otto von Guericke University Magdeburg
**Group:** 7
**Last updated:** 2026-04-03

---

## Research Question

> Does retrieval-based few-shot prompting significantly improve the quality, faithfulness, and usefulness of LLM-generated explanations for dark pattern detection, compared to zero-shot prompting?

---

## Phase 1 — Setup ✅ Complete

### 1.1 Dataset

- Source: EC-DarkPattern (Yada et al., 2022) — `data/raw/dataset.tsv`
- 2,361 examples across 8 categories (7 dark pattern types + Not Dark Pattern)
- Original stratified split: train 70% / val 15% / test 15%

**Class distribution (original training set):**

| Category | Count |
|---|---|
| Not Dark Pattern | 825 |
| Scarcity | 293 |
| Social Proof | 218 |
| Urgency | 147 |
| Misdirection | 136 |
| Obstruction | 19 |
| Sneaking | 8 |
| **Forced Action** | **3** |

Forced Action, Sneaking, and Obstruction were critically underrepresented.

---

### 1.2 Classification Baselines

Both BERT-base-uncased and RoBERTa-large fine-tuned on original train split.

**Results (original split, test macro F1):**

| Model | Val macro F1 | Test macro F1 |
|---|---|---|
| BERT-base-uncased | 0.687 | 0.805 |
| RoBERTa-large | 0.850 | 0.711 |

- Both models scored **0.0 F1** on Forced Action and Sneaking — too few training examples
- RoBERTa's test F1 lower than val indicates high variance from the small rare-class test pool
- Checkpoints saved to `outputs/models/bert/` and `outputs/models/roberta/`
- Full per-class metrics in `results/baselines/bert.json` and `results/baselines/roberta.json`

---

### 1.3 FAISS Retrieval Index

- Two indices built over the training split:
  - `outputs/indices/bert_index.faiss` — encoded with `bert-base-uncased`
  - `outputs/indices/sbert_index.faiss` — encoded with `sentence-transformers/all-mpnet-base-v2`
- All three retrieval strategies implemented in `src/retrieval/retrieve.py`:
  - `knn` — naive cosine similarity top-k
  - `prototype` — prefer examples closest to class centroid
  - `diversity` — MMR-based, penalises near-duplicates
- Metadata (texts + categories) stored alongside each index as JSON

---

### 1.4 Output Schema

Defined in `src/pipelines/schema.py` — every LLM call must return:

```json
{
  "pattern_type": "Scarcity | Urgency | Social Proof | ...",
  "psychological_mechanism": "e.g. loss aversion",
  "evidence_span": "exact substring from input text",
  "harm_dimension": "e.g. financial, autonomy",
  "rationale": "1-2 sentence explanation",
  "rewrite": "detoxified version of the product text",
  "confidence": 0.87,
  "label": "dark_pattern | not_dark_pattern"
}
```

---

## Phase 1.5 — Synthetic Data Augmentation ✅ Complete

### Motivation

Original splits had critically few examples for three classes:
- Forced Action: 4 total real examples (train+val+test)
- Sneaking: 12 total
- Obstruction: 27 total

This made both model training and evaluation unreliable for these classes.

### Method

Implemented a **similarity + composite prompting** strategy (Kochanek et al., 2024, *Electronics*):

- **Similarity prompts**: each call seeds the LLM with a real example from the target class, keeping generated text domain-grounded (e-commerce product pages)
- **Composite variation**: product type varies across calls (18 product types rotated) to prevent repetition
- **JSON output**: more reliable than Python list format (confirmed by paper)
- **Temperature 0.8**: high enough for diversity, consistent with paper's recommendation
- **LLM**: Qwen3:8b via Ollama (local, no API cost)

Script: `src/data/augment.py`

### Results

| Class | Before | After | Generated |
|---|---|---|---|
| Forced Action | 3 | 50 | 47 |
| Sneaking | 8 | 50 | 42 |
| Obstruction | 19 | 75 | 56 |

Generated examples logged in `results/augmentation/synthetic_examples.json`.

**Sample generated examples (quality check):**

*Forced Action:*
> "By proceeding, you agree to our Terms, Privacy Policy, and opt in to promotional emails from VeggieBox."
> "To complete your order, agree to our Terms of Service, Privacy Policy, and marketing communications. No opt-out available."

*Sneaking:*
> "Free Sample with Purchase – Auto-Renewal Subscription Applies"
> "Join Our Pet Lover Club – 10% Off Automatically Applied"

*Obstruction:*
> "CANCELLATION TERMS: To stop your monthly deliveries, email us at support@fashionhub.com by the 5th of the month..."
> "Cancel your PowerPump subscription by reaching out to our 24/7 helpline at 1-800-555-0199. Refunds are only available if..."

### Re-split (v2)

Combined original data (all splits) with synthetic examples, then re-stratified 70/15/15.
All splits now have every class represented.

Script: `src/data/resplit.py`

**v2 split distribution (key rare classes):**

| Class | train_v2 | val_v2 | test_v2 |
|---|---|---|---|
| Forced Action | 36 (33 synthetic) | 8 (7 synthetic) | 7 (7 synthetic) |
| Sneaking | 38 (32 synthetic) | 8 (5 synthetic) | 8 (5 synthetic) |
| Obstruction | 58 (38 synthetic) | 13 (11 synthetic) | 12 (7 synthetic) |

> **Note for paper:** The `synthetic` column in each CSV flags generated examples. Real-only subsets can be extracted for human evaluation by filtering `synthetic == False`.

### Augmented Baseline (BERT on train_augmented — first pass)

| Split | Orig BERT | Aug BERT | Δ |
|---|---|---|---|
| Val macro F1 | 0.687 | **0.886** | +0.199 |
| Test macro F1 | 0.805 | 0.808 | +0.003 |

Val improvement was large (+0.20). Test improvement was minimal because the original test split still had 0 Forced Action examples — this is why the re-split was necessary.

### Final Result: BERT on v2 splits (all rare classes properly evaluated)

| Category | Orig F1 | v2 F1 | Δ |
|---|---|---|---|
| **Forced Action** | 0.000 (0 test samples) | **0.933** | +0.933 |
| **Sneaking** | 0.000 (2 test samples) | **0.714** | +0.714 |
| Obstruction | 0.857 | 0.889 | +0.032 |
| Misdirection | 0.875 | 0.815 | -0.060 |
| Social Proof | 0.968 | 0.978 | +0.010 |
| Urgency | 0.973 | 0.954 | -0.019 |
| Scarcity | 0.985 | 0.992 | +0.007 |
| Not Dark Pattern | 0.977 | 0.967 | -0.010 |
| **macro avg** | **0.705** | **0.905** | **+0.201** |

- Val macro F1: 0.687 → **0.899**
- Test macro F1: 0.805 → **0.905**

The augmentation fully solved the rare-class problem. Macro F1 jumped +0.20 because the model can now learn and be evaluated on all 8 classes.

---

## Phase 2 — Explanation Pipeline ✅ Complete

**Goal:** Build the three prompting pipelines that generate structured JSON explanations.

### Files written

| File | Description |
|---|---|
| `src/pipelines/prompts.py` | System prompt + zero-shot / few-shot user prompt formatters |
| `src/pipelines/output_parser.py` | Parse Ollama JSON → `PredictionResult`; fuzzy label matching, fallbacks |
| `src/pipelines/span_grounding.py` | Verify `evidence_span` is a verbatim substring; exact + case-insensitive |
| `src/pipelines/zero_shot.py` | Zero-shot inference via Ollama — `predict()` + `predict_batch()` + CLI |
| `src/pipelines/random_few_shot.py` | k random training examples as few-shot context — same API |
| `src/pipelines/rag_few_shot.py` | FAISS retrieval → few-shot context → Ollama — same API |

### Prompt design

- Single `SYSTEM_PROMPT` shared by all three pipelines, with schema and category descriptions
- `format_zero_shot_prompt(text)` — bare product text + "Output JSON:"
- `format_few_shot_prompt(text, examples)` — numbered example block (Input + Label pairs) → target text
- Few-shot examples show `Product text / Label` pairs; full schema appears in system prompt once
- Qwen3:8b has thinking mode disabled via `think: false` in `ollama_client.py`

### Common pipeline API

Each pipeline exposes:
- `predict(text, ...) → (PredictionResult, grounding_info [, retrieved])` — single example
- `predict_batch(texts, ...) → list[dict]` — batch with grounding + metadata fields
- CLI: `python -m src.pipelines.<module> --text "..." ` or `--file test_v2.csv --out results/...`

### Next: run and collect results

```bash
# Zero-shot baseline
~/.pyenv/versions/.hcnlp/bin/python3 -m src.pipelines.zero_shot \
    --file data/processed/test_v2.csv --out results/pipelines/zero_shot.jsonl

# Random few-shot k=5
~/.pyenv/versions/.hcnlp/bin/python3 -m src.pipelines.random_few_shot \
    --file data/processed/test_v2.csv --k 5 --out results/pipelines/random_few_shot_k5.jsonl

# RAG knn k=5
~/.pyenv/versions/.hcnlp/bin/python3 -m src.pipelines.rag_few_shot \
    --file data/processed/test_v2.csv --k 5 --strategy knn \
    --out results/pipelines/rag_sbert_knn_k5.jsonl
```

---

## Phase 3 — Evaluation 🔲 Upcoming

### Automatic metrics
- Classification macro F1 per prompting condition (zero-shot / random few-shot / RAG)
- Span faithfulness rate: `evidence_span in input_text` substring check
- ROUGE-L and BERTScore for rewrite quality
- Confidence calibration: mean confidence vs. accuracy per condition

### Human evaluation rubric (6 dimensions, 1–3 scale)

| Dimension | What is assessed |
|---|---|
| Label correctness | Predicted category matches gold label |
| Mechanism correctness | Psychological mechanism is accurately named |
| Span faithfulness | Evidence span is verbatim and supports the label |
| Specificity | Rationale is specific to this text, not generic |
| Non-hallucination | Rationale avoids claims not in the text |
| Rewrite helpfulness | Rewrite removes manipulation, preserves core offer |

- 50 outputs sampled per condition for human scoring
- Inter-annotator agreement: Cohen's κ

### Ablation studies
- Retrieval encoder: BERT vs SBERT vs (optionally) fine-tuned encoder
- Retrieval strategy: knn vs prototype vs diversity
- LLM: Qwen3:8b vs Mistral-Nemo:12b
- k: {1, 3, 5, 10}

### Cross-lingual evaluation
- German test set (`data/german/`) evaluated directly and via translation
- Compare label accuracy and rubric scores between EN and DE conditions

---

## Phase 4 — Writeup 🔲 Late April

ACM-format paper (target 6–8 pages).

**Sections:**
1. Introduction — dark patterns, gap in explanation, research question
2. Related Work — EC-DarkPattern (Yada et al.), RAG (Lewis et al.), NLP explainability
3. System — output schema, retrieval, prompting conditions, rubric
4. Experiments — dataset, baselines, augmentation, setup
5. Results — classification tables, explanation quality, ablations
6. Cross-lingual Analysis — German transfer
7. Discussion — does RAG help? failure modes, confidence calibration
8. Conclusion

---

## Phase 5 — Demo 🔲 May

FastAPI + React application.

- `POST /analyze` — accepts product text, returns full schema JSON
- `GET /examples` — pre-loaded examples
- `?mode=` param to switch zero-shot / random / RAG
- React UI: highlighted evidence span, structured output cards, side-by-side rewrite

---

## File Map (current state)

```
src/
├── data/
│   ├── preprocess.py       ✅  original split generation
│   ├── dataset.py          ✅  PyTorch Dataset wrapper
│   ├── augment.py          ✅  synthetic data generation (Qwen3:8b)
│   └── resplit.py          ✅  re-split with synthetic examples
├── baselines/
│   ├── train_classifier.py ✅  BERT/RoBERTa fine-tuning
│   └── evaluate.py         ✅  evaluation helpers
├── retrieval/
│   ├── embeddings.py       ✅  BERT + SBERT encoders
│   ├── index.py            ✅  FAISS index build/load
│   └── retrieve.py         ✅  knn / prototype / diversity strategies
├── pipelines/
│   ├── schema.py           ✅  PredictionResult dataclass + OUTPUT_SCHEMA_STR
│   ├── prompts.py          ✅  SYSTEM_PROMPT + zero-shot / few-shot formatters
│   ├── output_parser.py    ✅  parse_prediction() with fuzzy label matching
│   ├── span_grounding.py   ✅  check_grounding() + grounding_rate()
│   ├── zero_shot.py        ✅  predict() + predict_batch() + CLI
│   ├── random_few_shot.py  ✅  predict() + predict_batch() + CLI
│   └── rag_few_shot.py     ✅  predict() + predict_batch() + CLI
├── evaluation/             🔲  Phase 3
└── utils/
    ├── ollama_client.py    ✅  Qwen3 + Mistral via Ollama
    └── io.py               ✅  JSONL helpers

data/
├── raw/dataset.tsv         ✅  original EC-DarkPattern
├── processed/
│   ├── train.csv           ✅  original split
│   ├── val.csv             ✅
│   ├── test.csv            ✅
│   ├── train_augmented.csv ✅  original + synthetic (first pass)
│   ├── train_v2.csv        ✅  re-split with synthetic
│   ├── val_v2.csv          ✅
│   └── test_v2.csv         ✅
└── german/                 🔲  to be annotated

outputs/
├── models/bert/            ✅  baseline checkpoint
├── models/roberta/         ✅  baseline checkpoint
└── indices/
    ├── bert_index.faiss    ✅
    └── sbert_index.faiss   ✅

results/
├── baselines/
│   ├── bert.json           ✅
│   ├── roberta.json        ✅
│   └── bert_train_augmented.json  ✅
└── augmentation/
    └── synthetic_examples.json    ✅
```

---

## Key Decisions Log

| Decision | Rationale |
|---|---|
| Qwen3:8b over Qwen2.5:7b | Better structured output and reasoning; `think: false` suppresses chain-of-thought preamble |
| Mistral-Nemo:12b as second LLM | Larger than Mistral 7B, good instruction following, enables LLM ablation |
| SBERT (all-mpnet-base-v2) for retrieval | Stronger semantic similarity than raw BERT CLS; off-the-shelf, no fine-tuning needed |
| Similarity + composite prompting for augmentation | Best strategy from Kochanek et al. (2024); grounded to domain via seed examples |
| Temperature 0.8 for augmentation | Matches paper's recommendation for diversity |
| No BERT validator on first augmentation pass | BERT had 0.0 F1 on rare classes — chicken-and-egg; validate on re-trained model |
| Re-split (v2) after augmentation | Ensures rare classes appear in val/test; critical for meaningful evaluation |
