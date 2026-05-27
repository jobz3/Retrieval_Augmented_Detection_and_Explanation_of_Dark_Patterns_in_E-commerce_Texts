#!/usr/bin/env bash
#
# Full from-scratch reproduction: synthetic augmentation -> English eval ->
# German + Italian cross-lingual eval.
#
# REPRODUCIBILITY NOTE
# --------------------
# This pipeline is NOT bit-for-bit reproducible, by design:
#   - `augment` generates synthetic data with an LLM at temperature 0.8, so the
#     synthetic set differs every run (similar class counts, different wording).
#   - All LLM evaluation steps use Ollama, which is non-deterministic even at
#     temperature 0 (~±0.03 macro F1 run-to-run, measured during this project).
# Deterministic stages: preprocess, resplit (given fixed synthetic input),
# index building, translated-index building.
# "Reproducible" therefore means: same conclusions and numbers within the
# measured ±0.03 variance band, not identical artifacts.
#
# AUGMENTATION IS OFF BY DEFAULT (RUN_AUGMENT=0).
# The synthetic-data step has a circular dependency: it filters LLM-generated
# rare-class examples with a BERT validator, but that validator — trained on the
# original imbalanced split — has F1=0 on Sneaking and Forced Action, so it
# rejects essentially every rare-class candidate ("Accepted 0 for Sneaking").
# The paper's accepted synthetic set is therefore committed to git at
# results/augmentation/synthetic_examples.json (47 Forced Action / 42 Sneaking /
# 56 Obstruction). The canonical reproduction REUSES that committed log and lets
# the deterministic `resplit` rebuild the exact paper train_v2.
# Set RUN_AUGMENT=1 only if you want to test the (stochastic, lossy) generator
# itself — it will NOT reproduce the paper's examples.
#
# PREREQUISITES
#   - Ollama serving qwen3:8b  (ollama serve; ollama pull qwen3:8b)
#   - .venv activated, requirements.txt installed
#   - data/raw/dataset.tsv present (EC-DarkPattern; from yamanalab/ec-darkpattern)
#   - data/italian/italian_dark_patterns.jsonl present (manual scrape; tracked in git)
#   - A CUDA GPU is strongly recommended (translate_index + classifier training)
#
# USAGE
#   bash reproduce_full_pipeline.sh
# Run stage-by-stage by commenting out sections, or copy individual commands.

set -euo pipefail
export PYTHONPATH=.

DEVICE="${DEVICE:-cuda}"   # override with DEVICE=cpu bash reproduce_full_pipeline.sh
SEED=42
RUN_AUGMENT="${RUN_AUGMENT:-0}"   # 0 = reuse committed synthetic log (canonical); 1 = regenerate (stochastic, lossy)
SYNTH_LOG="results/augmentation/synthetic_examples.json"

banner () { printf '\n============================================================\n  %s\n============================================================\n' "$1"; }

# ---------------------------------------------------------------------------
banner "STAGE 1  Preprocess raw TSV -> train/val/test splits  [deterministic]"
# Produces data/processed/{train,val,test}.csv + label_map.json
python -m src.data.preprocess

# ---------------------------------------------------------------------------
if [ "$RUN_AUGMENT" = "1" ]; then
  banner "STAGE 2  Train BERT validator on ORIGINAL split  [needed by augment]"
  # augment.py filters LLM-generated examples with this classifier.
  # Must train on 'train' (not train_v2, which does not exist yet).
  # NOTE: this validator will have F1=0 on Sneaking / Forced Action (rare in the
  # original split), which is why augment rejects most rare-class candidates.
  python -m src.baselines.train_classifier \
      --model bert --seed $SEED \
      --train-file train --val-file val --test-file test

  banner "STAGE 3  Synthetic augmentation  [STOCHASTIC, LOSSY: LLM temp 0.8]"
  # Will NOT reproduce the paper's examples. Overwrites the committed log.
  python -m src.data.augment --validator bert --seed $SEED
else
  banner "STAGE 2-3  SKIPPED  (RUN_AUGMENT=0) — reusing committed synthetic log"
  # Guard: ensure the committed paper synthetic set is intact before resplit.
  if git rev-parse --git-dir >/dev/null 2>&1; then
    git checkout -- "$SYNTH_LOG" 2>/dev/null || true
  fi
  python -c "import json,sys; d=json.load(open('$SYNTH_LOG')); c={k:len(v) for k,v in d.items()}; print('  synthetic log:', c); sys.exit(0 if c.get('Sneaking',0)>0 and c.get('Forced Action',0)>0 else 1)" \
    || { echo 'ERROR: committed synthetic log missing rare-class examples. Run: git checkout -- '"$SYNTH_LOG"; exit 1; }
fi

# ---------------------------------------------------------------------------
banner "STAGE 4  Re-split original + synthetic -> v2 splits  [deterministic]"
# Produces data/processed/{train_v2,val_v2,test_v2}.csv
python -m src.data.resplit

# ---------------------------------------------------------------------------
banner "STAGE 5  Train BERT baseline on v2 split  [paper baseline F1]"
python -m src.baselines.train_classifier \
    --model bert --seed $SEED \
    --train-file train_v2 --val-file val_v2 --test-file test_v2

# ---------------------------------------------------------------------------
banner "STAGE 6  Build FAISS retrieval indices on train_v2  [deterministic]"
python -m src.retrieval.index --encoder sbert        --split train_v2
python -m src.retrieval.index --encoder bert         --split train_v2
python -m src.retrieval.index --encoder roberta      --split train_v2
python -m src.retrieval.index --encoder multilingual --split train_v2

# ---------------------------------------------------------------------------
banner "STAGE 7  Build translated cross-lingual indices  [deterministic]"
# en->de and en->it via Helsinki-NLP/opus-mt; cached to data/processed/train_v2_{de,it}.csv
python -m src.retrieval.translate_index --lang de --device "$DEVICE"
python -m src.retrieval.translate_index --lang it --device "$DEVICE"

# ---------------------------------------------------------------------------
banner "STAGE 8  English evaluation: 8 pipeline configs  [STOCHASTIC: LLM]"
# Each writes results/pipelines/<name>.jsonl that metrics.py then scores.
python -m src.pipelines.zero_shot        --file data/processed/test_v2.csv \
    --out results/pipelines/zero_shot.jsonl
python -m src.pipelines.random_few_shot  --file data/processed/test_v2.csv --k 3 \
    --out results/pipelines/random_few_shot_k3.jsonl
python -m src.pipelines.random_few_shot  --file data/processed/test_v2.csv --k 5 \
    --out results/pipelines/random_few_shot_k5.jsonl
python -m src.pipelines.rag_few_shot     --file data/processed/test_v2.csv --k 5 --encoder sbert --strategy knn
python -m src.pipelines.rag_few_shot     --file data/processed/test_v2.csv --k 5 --encoder sbert --strategy diversity
python -m src.pipelines.rag_few_shot     --file data/processed/test_v2.csv --k 5 --encoder sbert --strategy prototype
python -m src.pipelines.rag_few_shot     --file data/processed/test_v2.csv --k 5 --encoder sbert --strategy hyde
python -m src.pipelines.rag_few_shot     --file data/processed/test_v2.csv --k 5 --encoder bert  --strategy knn

banner "STAGE 8b  Score all English configs"
python -m src.evaluation.metrics

# ---------------------------------------------------------------------------
banner "STAGE 9  German cross-lingual eval  [STOCHASTIC: LLM]"
python -m src.evaluation.cross_lingual_eval --lang de --mode both \
    --encoder multilingual --strategy knn --k 5 \
    --threshold 0.15 --diversity-alpha 0.3

# ---------------------------------------------------------------------------
banner "STAGE 10  Italian cross-lingual eval  [STOCHASTIC: LLM]"
# Shared-space (deployed) + translated index (k-sweep best).
python -m src.evaluation.cross_lingual_eval --lang it --mode both \
    --encoder multilingual --strategy knn --k 5 \
    --threshold 0.15 --diversity-alpha 0.3
python -m src.evaluation.cross_lingual_eval --lang it --mode rag \
    --encoder multilingual_it --strategy knn --k 1 \
    --threshold 0.15 --diversity-alpha 0.3

banner "STAGE 10b  Italian LaTeX table rows"
python -m src.evaluation.fill_italian_table

banner "DONE — compare results/evaluation/*.json against the paper numbers (within ±0.03)"
