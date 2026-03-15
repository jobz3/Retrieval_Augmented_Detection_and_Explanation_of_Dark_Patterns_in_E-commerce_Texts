# Retrieval-Augmented Detection and Explanation of Dark Patterns in E-Commerce Texts

This repository is the HCNLP research scaffold for:
- dark pattern classification on EC-DarkPattern
- structured LLM explanation generation for e-commerce texts
- comparison of zero-shot, random few-shot, and retrieval-augmented few-shot prompting

Phase 1 focuses on reproducible data preparation, BERT and RoBERTa classifier baselines, and FAISS indexing for the retrieval branch.

## Scope Boundaries

- `data/dataset_repo/` is upstream reference material only
- the canonical project-owned raw dataset is `data/raw/ec_darkpattern/dataset.tsv`
- the canonical processed split directory is `data/processed/ec_darkpattern/`
- zero-shot must not use retrieval
- random few-shot must not use retrieval
- only the RAG branch should call retrieval

## Setup

Recommended environment:
- Python 3.11
- local Ollama available at `http://localhost:11434` for later phases

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional environment file:

```powershell
Copy-Item .env.example .env
```

## Canonical Data Layout

- upstream reference clone: `data/dataset_repo/`
- canonical raw dataset: `data/raw/ec_darkpattern/dataset.tsv`
- canonical processed outputs: `data/processed/ec_darkpattern/`
- baseline metrics: `results/baselines/`
- model checkpoints: `outputs/models/`
- FAISS indices: `outputs/indices/`

## Phase 1 Commands

Bootstrap canonical raw data from the upstream clone:

```powershell
python -m src.data.bootstrap_raw
```

Audit raw data and generate validated splits:

```powershell
python -m src.data.preprocess
```

Build the Phase 1 retrieval indices:

```powershell
python -m src.retrieval.index --encoder bert
python -m src.retrieval.index --encoder roberta
```

Train and evaluate the classifier baselines:

```powershell
python -m src.baselines.train_classifier --model bert
python -m src.baselines.train_classifier --model roberta
python -m src.baselines.evaluate --model bert --split test
python -m src.baselines.evaluate --model roberta --split test
```

Helpers:

```powershell
.\run_phase1.ps1
```

```bash
./run_phase1.sh
```

## Expected Phase 1 Artifacts

After preprocessing:
- `data/processed/ec_darkpattern/raw_audit.json`
- `data/processed/ec_darkpattern/split_manifest.json`
- `data/processed/ec_darkpattern/train.csv`
- `data/processed/ec_darkpattern/val.csv`
- `data/processed/ec_darkpattern/test.csv`
- `data/processed/ec_darkpattern/label_map.json`

After indexing and training:
- `outputs/indices/*.faiss`
- `outputs/indices/*_metadata.json`
- `outputs/models/bert/`
- `outputs/models/roberta/`
- `results/baselines/bert.json`
- `results/baselines/roberta.json`

## Validation

Use the smallest meaningful validation for the area you changed. For Phase 1 stabilization, the preferred checks are:

```powershell
python -m src.data.bootstrap_raw
python -m src.data.preprocess
python -m src.retrieval.index --encoder bert
python -m src.retrieval.index --encoder roberta
python -m src.baselines.evaluate --model bert --split val
python -m src.baselines.evaluate --model roberta --split val
```

If a command cannot run because the local environment is missing dependencies, report that explicitly rather than silently skipping it.
