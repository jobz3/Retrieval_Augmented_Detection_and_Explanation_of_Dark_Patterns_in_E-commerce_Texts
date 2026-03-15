# Project Overview
This repository is the HCNLP research scaffold for dark pattern detection plus structured LLM explanation on e-commerce texts.

Phase 1 goal:
- make the EC-DarkPattern pipeline reproducible from local files
- normalize canonical raw and processed data locations
- generate audited, validated train/val/test splits
- reproduce BERT and RoBERTa classifier baselines on the canonical split
- build FAISS retrieval indices for the Phase 1 retrieval encoders

Step 1 is complete when:
- the project reads configuration from `config.yaml`
- canonical raw data exists at `data/raw/ec_darkpattern/dataset.tsv`
- audited, validated splits exist in `data/processed/ec_darkpattern/`
- `raw_audit.json` and `split_manifest.json` are written
- BERT and RoBERTa training scripts and FAISS indexing scripts consume those canonical outputs
- Windows users can run the documented Phase 1 workflow from PowerShell

# Architecture
Truth sources and ownership:
- primary dataset source of truth: `data/raw/ec_darkpattern/dataset.tsv`
- upstream reference material only: `data/dataset_repo/`
- processed split artifacts owned by this repo: `data/processed/ec_darkpattern/`
- classifier baselines: `src/baselines/`
- retrieval indexing for the RAG branch only: `src/retrieval/`
- structured output schema: `src/pipelines/schema.py`
- local Ollama wrapper: `src/utils/ollama_client.py`

Conceptual boundaries:
- zero-shot does not use retrieval
- random few-shot does not use retrieval
- only the RAG branch should call retrieval
- BERT vs RoBERTa classifier baselines are a separate comparison axis from retrieval-encoder ablations
- upstream `darkpattern-auto-detection-classical/` and `darkpattern-auto-detection-deeplearning/` are reference implementations, not separate datasets

# Setup
- Python 3.11 recommended
- create and activate a virtual environment
- install dependencies with `pip install -r requirements.txt`
- copy `.env.example` to `.env` if Ollama endpoint overrides are needed
- keep Ollama local-first; default endpoint is `http://localhost:11434`

# Canonical Data Paths
- canonical raw dataset: `data/raw/ec_darkpattern/dataset.tsv`
- canonical processed split directory: `data/processed/ec_darkpattern/`
- upstream dataset clone: `data/dataset_repo/`
- optional German set placeholder: `data/german/`
- model outputs: `outputs/models/`
- retrieval indices: `outputs/indices/`
- baseline metrics: `results/baselines/`

# Run Commands
Phase 1 bootstrap:
- `python -m src.data.bootstrap_raw`

Phase 1 preprocessing:
- `python -m src.data.preprocess`

Phase 1 retrieval indices:
- `python -m src.retrieval.index --encoder bert`
- `python -m src.retrieval.index --encoder roberta`

Phase 1 baselines:
- `python -m src.baselines.train_classifier --model bert`
- `python -m src.baselines.train_classifier --model roberta`
- `python -m src.baselines.evaluate --model bert --split test`
- `python -m src.baselines.evaluate --model roberta --split test`

Helpers:
- PowerShell: `.\run_phase1.ps1`
- Bash: `./run_phase1.sh`

# Validation
Before considering a Phase 1 task complete, run the smallest meaningful validation for the touched area.

Preferred checks for this scaffold:
- `python -m src.data.bootstrap_raw`
- `python -m src.data.preprocess`
- `python -m src.retrieval.index --encoder bert`
- `python -m src.retrieval.index --encoder roberta`
- `python -m src.baselines.evaluate --model bert --split val`
- `python -m src.baselines.evaluate --model roberta --split val`

If a heavier command is not run, say so explicitly.

# Coding Rules
- keep changes minimal and local
- preserve public APIs unless the task requires a change
- do not edit upstream files under `data/dataset_repo/`
- do not treat `classical` as a second dataset
- do not hardcode secrets
- do not break Windows paths
- do not rewrite unrelated files
- do not silently skip failing split validation
- do not add retrieval dependencies to zero-shot or random few-shot logic

# Repo Map
- preprocessing and split generation: `src/data/preprocess.py`
- processed dataset wrapper: `src/data/dataset.py`
- baseline training and evaluation: `src/baselines/`
- retrieval embeddings and FAISS index build: `src/retrieval/`
- prompt output schema: `src/pipelines/schema.py`
- Ollama utilities: `src/utils/`
- stored baseline artifacts: `results/baselines/`

# Task Workflow
- inspect relevant files first
- explain the root cause before patching
- patch the smallest surface that restores consistency
- validate after each meaningful change with the lightest useful command
- summarize changed files, validation results, and remaining risks

# Forbidden Mistakes
- do not use `data/dataset_repo/dataset/dataset.tsv` as the live pipeline path
- do not silently write splits that drop feasible classes from val/test
- do not make `config.yaml` drift away from active code
- do not implement Phase 2 or demo work during Phase 1 stabilization unless explicitly requested
