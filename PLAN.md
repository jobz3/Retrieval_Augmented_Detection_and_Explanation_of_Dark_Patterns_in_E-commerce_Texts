# Project Plan: Retrieval-Augmented Detection and Explanation of Dark Patterns in E-Commerce Texts

HCNLP · Otto von Guericke University Magdeburg · Group 7

---

## Research Question

> **Does retrieval-based few-shot prompting significantly improve the quality, faithfulness, and usefulness of LLM-generated explanations for dark pattern detection, compared to zero-shot prompting?**

---

## Overview

This project extends dark pattern *classification* into dark pattern *explanation*. Given a product text, the system must:

1. Predict the dark pattern category (or "no dark pattern")
2. Output a structured explanation (pattern type, psychological mechanism, evidence span, harm dimension, rationale)
3. Rewrite the text to remove the dark pattern while preserving the core offer
4. Produce a confidence score for both label and explanation

Three prompting conditions are compared: **zero-shot**, **random few-shot**, and **retrieval-augmented few-shot** (RAG). BERT-base and RoBERTa-large serve as classification baselines.

---

## Structured Output Schema

Every LLM call must return a JSON object conforming to this schema:

```json
{
  "label": "Scarcity | Social Proof | Urgency | Authority | ...",
  "confidence": 0.0,
  "psychological_mechanism": "...",
  "evidence_span": "quoted substring from input text",
  "harm_dimension": "...",
  "rationale": "one-to-two sentence natural language explanation",
  "rewrite": "detoxified version of the full product text"
}
```

Field definitions:

| Field | Description |
|---|---|
| `label` | One of the 7 EC-DarkPattern categories, or `"none"` / `"uncertain"` |
| `confidence` | Float 0–1; the model's self-assessed certainty for the label |
| `psychological_mechanism` | Named cognitive bias or persuasion principle (e.g., loss aversion, social proof) |
| `evidence_span` | Verbatim substring from the input text that most directly triggers the pattern |
| `harm_dimension` | Short phrase describing potential consumer harm (e.g., false urgency, privacy erosion) |
| `rationale` | 1–2 sentence explanation of why the text constitutes this pattern |
| `rewrite` | Full product text rewritten to remove the dark pattern while preserving the offer |

Each field is evaluated separately in the rubric (see Evaluation section).

---

## Dataset

**Primary:** EC-DarkPattern (Yada et al., 2022) — 1,818 product texts, 7 categories.

**Secondary (cross-lingual):** Small manually annotated set of German e-commerce texts. Used exclusively as a test set to study explanation-quality transfer from English training data to German product texts. Two evaluation axes:
- Direct German inference (zero-shot cross-lingual transfer)
- Translated-to-English inference (compare translation vs native)

Class distribution should be audited; apply stratified splits (train/val/test ≈ 70/15/15).

---

## Models

| Role | Model |
|---|---|
| Classification baseline | BERT-base-uncased (fine-tuned) |
| Classification baseline | RoBERTa-large (fine-tuned) |
| Explanation generation | Qwen (via Ollama, local) |
| Explanation generation | Mistral (via Ollama, local) |
| Embeddings for retrieval | BERT-base encoder (off-the-shelf) |
| Embeddings for retrieval | RoBERTa-large encoder (off-the-shelf) |
| Embeddings for retrieval | Fine-tuned encoder on EC-DarkPattern (task-specific) |

---

## Retrieval System

### Index
- Build a FAISS index over the training split of EC-DarkPattern.
- Encode each example with: BERT-base, RoBERTa-large, and one task-specifically fine-tuned encoder. This enables a **retrieval representation ablation** (which encoder finds better few-shot examples?).

### Retrieval Strategy
Three retrieval strategies (ablation within the RAG condition):
1. **Naive k-NN** — top-k by cosine similarity, no filtering.
2. **Prototype-preferring** — prefer examples whose embedding is closest to the class centroid (prototypical retrieval).
3. **Diversity-balanced** — penalize near-duplicates; ensure retrieved examples cover multiple pattern subtypes.

Default k = 5.

---

## Prompting Conditions

| Condition | Few-shot examples | Retrieval |
|---|---|---|
| Zero-shot | None | No |
| Random few-shot | k random training examples | No |
| RAG (naive k-NN) | Top-k by similarity | BERT embeddings |
| RAG (prototype) | Prototypical examples | RoBERTa embeddings |
| RAG (diversity) | Balanced diverse examples | Fine-tuned embeddings |

All conditions use the same structured output prompt template. Only the in-context examples differ.

---

## Evaluation Rubric

Every generated output is scored on six dimensions (each 0–2):

| Dimension | What is assessed |
|---|---|
| **Label correctness** | Does the predicted label match the gold label? |
| **Mechanism correctness** | Does the psychological mechanism correctly describe the pattern? |
| **Span faithfulness** | Does the evidence span appear verbatim in the input and support the label? |
| **Specificity** | Is the rationale specific to this text, not generic? |
| **Non-hallucination** | Does the rationale avoid claims not supported by the text? |
| **Rewrite usefulness** | Does the rewrite preserve the core offer while removing the manipulative element? |

Scoring method: automated where possible (label accuracy, span substring check, ROUGE/BERTScore for rewrite), human spot-check for mechanism correctness, specificity, non-hallucination, and rewrite usefulness.

---

## Phase-by-Phase Plan

### Phase 1 — Setup (March, now through ~March 28)

**Goal:** Reproducible environment; all baselines working; FAISS index operational.

#### 1.1 Repository and Environment
- [ ] Create repo structure (see below)
- [ ] `requirements.txt` / `pyproject.toml` with pinned versions
- [ ] `.env.example` for Ollama endpoint, paths
- [ ] Confirm Ollama runs locally with Qwen and Mistral

#### 1.2 Data
- [ ] Download EC-DarkPattern dataset; place in `data/raw/`
- [ ] EDA: class distribution, text length statistics, category descriptions
- [ ] Stratified train/val/test split; save to `data/processed/`
- [ ] Collect/curate German e-commerce test set; save to `data/german/`

#### 1.3 Classification Baselines
- [ ] Fine-tune BERT-base on train split; evaluate on test split (macro F1)
- [ ] Fine-tune RoBERTa-large on train split; evaluate on test split (macro F1)
- [ ] Save model checkpoints and per-class metrics to `results/baselines/`

#### 1.4 FAISS Index
- [ ] Encode training split with BERT-base encoder → build FAISS index
- [ ] Encode training split with RoBERTa-large encoder → build second FAISS index
- [ ] Write `retrieval/retrieve.py` with a `retrieve(query, k)` function
- [ ] Unit test: verify returned examples are semantically relevant
- [ ] (Deferred to Phase 2) Fine-tuned encoder index

**Midpoint deliverable:** BERT and RoBERTa baselines with reported macro F1, working FAISS retrieval, zero-shot and random few-shot pipelines returning structured JSON on a 50-example validation slice.

---

### Phase 2 — Pipeline (Early April, ~March 29 – April 11)

**Goal:** Full RAG explanation pipeline operational across all prompting conditions.

#### 2.1 Prompt Engineering
- [ ] Design the structured output prompt template (system + user message)
- [ ] Define JSON schema and add it to the prompt (or use Ollama's `format` parameter)
- [ ] Handle abstention: add `"uncertain"` as a valid label with a fallback message
- [ ] Add confidence elicitation instruction to the prompt

#### 2.2 Prompting Pipelines
- [ ] Implement `pipelines/zero_shot.py`
- [ ] Implement `pipelines/random_few_shot.py`
- [ ] Implement `pipelines/rag_pipeline.py` (wraps retrieval + prompt construction)
- [ ] All pipelines return a `PredictionResult` dataclass matching the output schema
- [ ] Run all pipelines on the full validation split; save outputs to `results/predictions/`

#### 2.3 Retrieval Strategies
- [ ] Implement naive k-NN retrieval (already from Phase 1)
- [ ] Implement prototype-preferring retrieval (precompute class centroids)
- [ ] Implement diversity-balanced retrieval (MMR or near-duplicate penalty)
- [ ] Fine-tune an encoder on EC-DarkPattern (contrastive / triplet loss); build third FAISS index

#### 2.4 Evidence Span Extraction
- [ ] Post-process model output: verify `evidence_span` is a substring of the input
- [ ] If model hallucinates a span, flag it (do not silently accept)
- [ ] Log span extraction failure rate per condition

---

### Phase 3 — Evaluation (Mid April, ~April 12 – April 22)

**Goal:** Quantitative and qualitative results across all conditions.

#### 3.1 Automatic Metrics
- [ ] Compute macro F1 per condition (label accuracy)
- [ ] Compute span faithfulness rate (substring match)
- [ ] Compute ROUGE-L and BERTScore for rewrite quality
- [ ] Compute mean confidence by condition; calibration curve (confidence vs. accuracy)
- [ ] Analyze low-confidence and uncertain cases: do explanations degrade?

#### 3.2 Human Evaluation
- [ ] Sample 50 outputs per condition for human scoring
- [ ] Apply the 6-dimension rubric; each annotator scores independently
- [ ] Compute inter-annotator agreement (Cohen's kappa or Fleiss' kappa)
- [ ] Aggregate and compare mean rubric scores across conditions

#### 3.3 Ablation Studies
- [ ] Retrieval encoder ablation: BERT vs RoBERTa vs fine-tuned → which gives best explanation quality?
- [ ] Retrieval strategy ablation: naive vs prototype vs diversity
- [ ] LLM ablation: Qwen vs Mistral (same prompts)
- [ ] k ablation: k ∈ {1, 3, 5, 10}

#### 3.4 Cross-lingual Evaluation
- [ ] Run all conditions on German test set (direct inference)
- [ ] Run all conditions on German test set translated to English
- [ ] Compare label accuracy and rubric scores between German direct vs translated
- [ ] Qualitative analysis: what fails in cross-lingual transfer?

#### 3.5 Error Analysis
- [ ] Identify categories with worst performance
- [ ] Sample 10–15 failure cases; characterize patterns of failure
- [ ] Check correlation between confidence score and rubric quality scores

---

### Phase 4 — Writeup (Late April, ~April 23 – April 30)

**Goal:** ACM-format paper (6–8 pages).

#### Suggested Structure

1. **Introduction** — dark patterns problem, gap in explanation, research question
2. **Related Work** — EC-DarkPattern dataset, RAG (Lewis et al.), explainability in NLP
3. **Method** — output schema, retrieval system, prompting conditions, rubric
4. **Experiments** — dataset, baselines, experimental setup
5. **Results** — quantitative tables, ablation tables, human evaluation
6. **Cross-lingual Analysis** — German transfer results
7. **Discussion** — does RAG help? what fails? confidence calibration
8. **Conclusion**

- [ ] Write Sections 1–3 draft
- [ ] Write Sections 4–5 with tables from Phase 3
- [ ] Write Sections 6–8
- [ ] Internal review pass; address feedback
- [ ] Final proofreading and formatting

---

### Phase 5 — Demo (May)

**Goal:** A runnable FastAPI + React application demonstrating the full pipeline on arbitrary product text input.

#### Backend (FastAPI)
- [ ] `POST /analyze` — accepts product text, returns full structured output (label, confidence, mechanism, evidence span, harm dimension, rationale, rewrite)
- [ ] `GET /examples` — returns a few pre-loaded examples
- [ ] Wire up the RAG pipeline as the default inference mode
- [ ] Add a `?mode=` query param to switch between zero-shot / random / rag

#### Frontend (React)
- [ ] Text input area for product text
- [ ] Submit button → calls `/analyze`
- [ ] Display: highlighted evidence span within original text, structured output fields, rewritten version side-by-side
- [ ] Mode selector (zero-shot / random / RAG)
- [ ] A few pre-loaded example texts for quick demo

---

## Repository Structure

```
.
├── data/
│   ├── raw/                  # Original EC-DarkPattern dataset
│   ├── processed/            # Train/val/test splits
│   └── german/               # German e-commerce test set
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baselines.ipynb
│   └── 03_results_analysis.ipynb
│
├── src/
│   ├── data/
│   │   ├── dataset.py        # Dataset loading and preprocessing
│   │   └── splits.py         # Stratified splitting logic
│   │
│   ├── baselines/
│   │   ├── bert_classifier.py
│   │   └── roberta_classifier.py
│   │
│   ├── retrieval/
│   │   ├── embeddings.py     # Encode texts with BERT/RoBERTa
│   │   ├── index.py          # FAISS index build and load
│   │   ├── retrieve.py       # retrieve(query, k, strategy)
│   │   └── strategies.py     # naive, prototype, diversity
│   │
│   ├── pipelines/
│   │   ├── schema.py         # PredictionResult dataclass + JSON schema
│   │   ├── prompts.py        # Prompt templates
│   │   ├── zero_shot.py
│   │   ├── random_few_shot.py
│   │   └── rag_pipeline.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py        # F1, span match, ROUGE, BERTScore
│   │   ├── rubric.py         # Human rubric scoring helpers
│   │   └── analysis.py       # Confidence calibration, error analysis
│   │
│   └── utils/
│       ├── ollama_client.py  # Thin wrapper around Ollama API
│       └── io.py             # Load/save predictions as JSONL
│
├── backend/
│   └── main.py               # FastAPI app
│
├── frontend/
│   └── src/                  # React application
│
├── results/
│   ├── baselines/
│   ├── predictions/
│   └── evaluation/
│
├── PLAN.md                   # This file
├── requirements.txt
└── README.md
```

---

## Key Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Ollama/LLM produces malformed JSON | Use Ollama `format=json` + output validation + retry logic |
| Model confidence is poorly calibrated | Report raw confidence; do not over-interpret; use as a ranking signal |
| German test set too small for significance | Frame as qualitative/exploratory; report exact sample sizes |
| Fine-tuning encoder takes too long | Use a small model (e.g., sentence-transformers/all-MiniLM) as fallback |
| Human evaluation bottleneck | Limit human eval to 50 samples per condition; 2 annotators minimum |

---

## Division of Labor (suggested, to be confirmed by group)

| Component | Primary owner |
|---|---|
| Data preprocessing, EDA | TBD |
| BERT/RoBERTa baselines | TBD |
| FAISS index + retrieval strategies | TBD |
| Prompt engineering + LLM pipelines | TBD |
| Evaluation metrics + rubric | TBD |
| Cross-lingual analysis | TBD |
| FastAPI backend | TBD |
| React frontend | TBD |
| Paper writing | All |
