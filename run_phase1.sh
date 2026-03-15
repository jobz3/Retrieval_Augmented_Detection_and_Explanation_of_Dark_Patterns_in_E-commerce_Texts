#!/usr/bin/env bash
# Run the canonical Phase 1 workflow end to end.
set -e

echo "=== Step 1: Bootstrap canonical raw dataset ==="
python -m src.data.bootstrap_raw

echo ""
echo "=== Step 2: Audit and preprocess canonical dataset ==="
python -m src.data.preprocess

echo ""
echo "=== Step 3: Build FAISS index (bert) ==="
python -m src.retrieval.index --encoder bert

echo ""
echo "=== Step 4: Build FAISS index (roberta) ==="
python -m src.retrieval.index --encoder roberta

echo ""
echo "=== Step 5: Train BERT baseline ==="
python -m src.baselines.train_classifier --model bert

echo ""
echo "=== Step 6: Train RoBERTa baseline ==="
python -m src.baselines.train_classifier --model roberta

echo ""
echo "=== Phase 1 complete ==="
