#!/usr/bin/env bash
# Run all Phase 1 steps in order.
# Prerequisite: pip install -r requirements.txt
#               Place dataset.tsv in data/raw/dataset.tsv
set -e

echo "=== Step 1: Preprocess dataset ==="
python -m src.data.preprocess

echo ""
echo "=== Step 2: Build FAISS index (sbert — fast default) ==="
python -m src.retrieval.index --encoder sbert

echo ""
echo "=== Step 3: Build FAISS index (bert) ==="
python -m src.retrieval.index --encoder bert

echo ""
echo "=== Step 4: Train BERT baseline ==="
python -m src.baselines.train_classifier --model bert

echo ""
echo "=== Step 5: Train RoBERTa baseline ==="
python -m src.baselines.train_classifier --model roberta

echo ""
echo "=== Phase 1 complete. Results in results/baselines/ ==="
