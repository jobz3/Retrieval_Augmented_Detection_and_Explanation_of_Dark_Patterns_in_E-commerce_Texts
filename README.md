# Retrieval-Augmented Detection and Explanation of Dark Patterns in E-commerce Texts

**Group 7 — Human-Centred NLP, OvGU Magdeburg (2026)**  
Jobin Roy · Akshita Sakshi · Shifin Mohammed

---

## Overview

This project detects and explains dark patterns in e-commerce product text using a retrieval-augmented generation (RAG) pipeline. Given a text snippet or a product URL, the system:

1. Retrieves the *k* most similar labelled examples from the EC-DarkPattern training set (SBERT + FAISS)
2. Passes them as few-shot demonstrations to Qwen3-8B (via Ollama)
3. Returns a structured 7-field JSON: label, confidence, evidence span, psychological mechanism, harm dimension, rationale, and a neutral rewrite

**Best configuration:** RAG SBERT KNN k=5 — macro F1 = **0.923**, Cohen's κ = **0.959**

---

## Architecture

```
User input (text / URL)
        │
        ▼
  Web scraper (BeautifulSoup)         ← URL mode only
        │
        ▼
  SBERT encoder (all-mpnet-base-v2)
        │
        ▼
  FAISS flat index  ──► top-k similar training examples
        │
        ▼
  Qwen3-8B (Ollama)  ◄── few-shot prompt + system schema
        │
        ▼
  Structured output (7 fields)
        │
        ▼
  Streamlit UI
```

---

## Prerequisites

### Docker method (recommended)
- [Docker](https://docs.docker.com/get-docker/) ≥ 24
- [Docker Compose](https://docs.docker.com/compose/) ≥ 2.20
- ~10 GB free disk (Qwen3-8B model + image layers)
- The preprocessed data must exist locally (see [Data Setup](#data-setup))

### Manual method
- Python 3.11+
- [Ollama](https://ollama.com/download) installed and running
- ~6 GB free disk for Qwen3-8B

---

## Data Setup

The processed dataset and FAISS indices are not committed to git. Before running the demo, ensure the following directories exist and contain data:

```
data/processed/
    train_v2.csv          ← augmented training split (2,501 examples)
    test_v2.csv           ← test split (376 examples)
    label_map.json

outputs/indices/
    sbert_index.faiss     ← built automatically on first Docker run if missing
    sbert_embeddings.npy
    sbert_metadata.json
```

If you have the raw dataset (`data/raw/dataset.tsv`), run preprocessing first:

```bash
PYTHONPATH=. python -m src.data.preprocess
PYTHONPATH=. python -m src.data.resplit
```

---

## Quick Start — Docker

```bash
# 1. Clone the repository
git clone <repo-url>
cd Retrieval_Augmented_Detection_and_Explanation_of_Dark_Patterns_in_E-commerce_Texts

# 2. Start everything (Ollama + app)
docker compose up
```

On **first run** the entrypoint will:
- Wait for Ollama to be ready
- Pull `qwen3:8b` (~5 GB — takes several minutes)
- Build the SBERT FAISS index if `outputs/indices/sbert_index.faiss` is missing
- Start Streamlit

Open **http://localhost:8501** in your browser.

### Stop

```bash
docker compose down
```

### Rebuild after code changes

```bash
docker compose up --build
```

---

## Quick Start — Manual

### 1. Create environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-demo.txt
```

### 2. Start Ollama and pull the model

```bash
# In a separate terminal
ollama serve

# Pull the model (one-time, ~5 GB)
ollama pull qwen3:8b
```

### 3. Build the SBERT retrieval index

```bash
PYTHONPATH=. python -m src.retrieval.index --encoder sbert
```

### 4. Run the demo

```bash
PYTHONPATH=. streamlit run demo/streamlit_app.py --server.port 8501
```

Open **http://localhost:8501**.

---

## Configuration

Copy `.env.example` to `.env` and adjust if needed:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint (use `http://ollama:11434` inside Docker) |

The main pipeline settings (encoder, k, strategy, model) are in [config.yaml](config.yaml).

---

## Demo Usage

**URL mode** — paste any e-commerce product URL (e.g. Amazon, AboutYou, ASOS). The app fetches the page, extracts up to 25 candidate snippets, and analyses each one in sequence.

**Text mode** — paste a single text snippet directly into the text area.

Each result card shows:
- Detected label and confidence score
- Highlighted evidence span in the original text
- Psychological mechanism and consumer harm dimension
- Rationale explaining why it is (or is not) a dark pattern
- A suggested neutral rewrite

---

## Project Structure

```
.
├── demo/
│   └── streamlit_app.py       ← Streamlit UI + web scraper
├── src/
│   ├── data/                  ← preprocessing & synthetic augmentation
│   ├── pipelines/             ← RAG, zero-shot, prompt templates, output parser
│   ├── retrieval/             ← FAISS index builder & retriever
│   └── utils/                 ← Ollama client
├── data/
│   ├── processed/             ← train/test splits (gitignored)
│   └── german/                ← manually annotated German test set
├── outputs/
│   └── indices/               ← FAISS indices (gitignored)
├── results/                   ← evaluation metrics (JSON)
├── paper_latex/               ← ACM paper source
├── presentation/              ← Beamer slides + presenter dossier
├── docker/
│   └── entrypoint.sh          ← container startup script
├── Dockerfile
├── docker-compose.yml
├── config.yaml
└── requirements-demo.txt      ← lean deps for the demo
```

---

## Reproducibility

To reproduce full pipeline evaluation results:

```bash
# Install all dependencies
pip install -r requirements.txt

# Build all encoder indices
PYTHONPATH=. python -m src.retrieval.index --encoder sbert
PYTHONPATH=. python -m src.retrieval.index --encoder bert
PYTHONPATH=. python -m src.retrieval.index --encoder roberta

# Run all pipeline configurations
PYTHONPATH=. python -m src.pipelines.rag_few_shot --run-all

# Evaluate
PYTHONPATH=. python -m src.evaluation.evaluate
```

---

## References

- Yada et al. (2022) — EC-DarkPattern dataset
- Lewis et al. (2020) — Retrieval-Augmented Generation
- Reimers & Gurevych (2019) — Sentence-BERT
- Kochanek et al. (2024) — Composite + similarity prompting for synthetic data
- Kim et al. (2024) — Prometheus LLM-as-judge
