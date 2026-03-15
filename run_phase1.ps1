$ErrorActionPreference = "Stop"

Write-Host "=== Step 1: Bootstrap canonical raw dataset ==="
python -m src.data.bootstrap_raw

Write-Host ""
Write-Host "=== Step 2: Audit and preprocess canonical dataset ==="
python -m src.data.preprocess

Write-Host ""
Write-Host "=== Step 3: Build FAISS index (bert) ==="
python -m src.retrieval.index --encoder bert

Write-Host ""
Write-Host "=== Step 4: Build FAISS index (roberta) ==="
python -m src.retrieval.index --encoder roberta

Write-Host ""
Write-Host "=== Step 5: Train BERT baseline ==="
python -m src.baselines.train_classifier --model bert

Write-Host ""
Write-Host "=== Step 6: Train RoBERTa baseline ==="
python -m src.baselines.train_classifier --model roberta

Write-Host ""
Write-Host "=== Phase 1 complete ==="
