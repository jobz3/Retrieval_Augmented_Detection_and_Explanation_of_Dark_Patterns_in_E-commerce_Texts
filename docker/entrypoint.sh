#!/usr/bin/env bash
set -euo pipefail

OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
MODEL="qwen3:8b"
INDEX_FILE="outputs/indices/sbert_index.faiss"
TRAIN_CSV="data/processed/train_v2.csv"

# ── 1. Wait for Ollama ────────────────────────────────────────────────────────
echo "[entrypoint] Waiting for Ollama at ${OLLAMA_URL}…"
until curl -sf "${OLLAMA_URL}/api/tags" > /dev/null; do
    sleep 3
done
echo "[entrypoint] Ollama is ready."

# ── 2. Pull model if not present ─────────────────────────────────────────────
if curl -sf "${OLLAMA_URL}/api/tags" | grep -q "\"${MODEL}\""; then
    echo "[entrypoint] Model ${MODEL} already available."
else
    echo "[entrypoint] Pulling ${MODEL} — this may take several minutes on first run…"
    curl -X POST "${OLLAMA_URL}/api/pull" \
        -H "Content-Type: application/json" \
        -d "{\"name\":\"${MODEL}\"}" \
        --no-buffer
    echo ""
    echo "[entrypoint] Model pull complete."
fi

# ── 3. Build SBERT index if missing ──────────────────────────────────────────
if [ -f "${INDEX_FILE}" ]; then
    echo "[entrypoint] FAISS index found — skipping build."
else
    if [ ! -f "${TRAIN_CSV}" ]; then
        echo "[entrypoint] ERROR: ${TRAIN_CSV} not found."
        echo "             Mount the processed data directory before starting:"
        echo "             docker compose up  (requires ./data/processed/ on the host)"
        exit 1
    fi
    echo "[entrypoint] Building SBERT retrieval index…"
    python -m src.retrieval.index --encoder sbert
    echo "[entrypoint] Index build complete."
fi

# ── 4. Start Streamlit ────────────────────────────────────────────────────────
echo "[entrypoint] Starting Streamlit on port 8501…"
exec streamlit run demo/streamlit_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
