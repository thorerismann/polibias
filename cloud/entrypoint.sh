#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────
RUN_NAME="${RUN_NAME:-cloud_run}"
GCS_BUCKET="${GCS_BUCKET:?GCS_BUCKET env var is required}"

echo "=== polibias cloud run ==="
echo "  RUN_NAME:   $RUN_NAME"
echo "  GCS_BUCKET: $GCS_BUCKET"

# ── Start Ollama daemon ───────────────────────────────────────
echo "Starting Ollama server ..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
for i in $(seq 1 30); do
    if curl -sf http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama ready (took ${i}s)"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: Ollama failed to start within 30s"
        exit 1
    fi
    sleep 1
done

# ── Run the pipeline ──────────────────────────────────────────
echo "Running polibias pipeline ..."
polibias all \
    --run-dir "$RUN_NAME" \
    --config /app/cloud/config.toml

# ── Post-pipeline steps ──────────────────────────────────────
echo "Running stats ..."
polibias stats --run-dir "$RUN_NAME" --config /app/cloud/config.toml

echo "Running export ..."
polibias export --run-dir "$RUN_NAME" --config /app/cloud/config.toml

# ── Upload results to GCS ────────────────────────────────────
echo "Uploading results to gs://${GCS_BUCKET}/${RUN_NAME}/ ..."
polibias upload \
    --run-dir "$RUN_NAME" \
    --bucket "$GCS_BUCKET" \
    --config /app/cloud/config.toml

# ── Cleanup ──────────────────────────────────────────────────
kill $OLLAMA_PID 2>/dev/null || true
echo "=== Done ==="
