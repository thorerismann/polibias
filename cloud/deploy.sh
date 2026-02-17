#!/usr/bin/env bash
set -euo pipefail

# ── User-configurable variables ──────────────────────────────
PROJECT_ID="${GCP_PROJECT:?Set GCP_PROJECT env var}"
REGION="${GCP_REGION:-europe-west4}"       # Has L4 GPU availability
REPO_NAME="polibias"
IMAGE_NAME="polibias"
JOB_NAME="polibias-run"
GCS_BUCKET="${GCS_BUCKET:?Set GCS_BUCKET env var}"
RUN_NAME="${RUN_NAME:-cloud_run}"

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest"

# ── Resolve project root (parent of cloud/) ──────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== polibias Cloud Run deployment ==="
echo "  PROJECT:    $PROJECT_ID"
echo "  REGION:     $REGION"
echo "  IMAGE:      $IMAGE_URI"
echo "  GCS_BUCKET: $GCS_BUCKET"
echo "  RUN_NAME:   $RUN_NAME"
echo ""

# ── Step 1: Ensure Artifact Registry repo exists ─────────────
echo "[1/4] Ensuring Artifact Registry repository ..."
gcloud artifacts repositories describe "$REPO_NAME" \
    --project="$PROJECT_ID" \
    --location="$REGION" > /dev/null 2>&1 || \
gcloud artifacts repositories create "$REPO_NAME" \
    --project="$PROJECT_ID" \
    --location="$REGION" \
    --repository-format=docker \
    --description="polibias container images"

# ── Step 2: Build and push image with Cloud Build ────────────
echo "[2/4] Building and pushing image (this takes ~10 min on first build) ..."
gcloud builds submit "$PROJECT_ROOT" \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --tag="$IMAGE_URI" \
    --timeout=3600s

# ── Step 3: Create or update the Cloud Run Job ──────────────
echo "[3/4] Creating Cloud Run Job ..."
if gcloud run jobs describe "$JOB_NAME" \
    --project="$PROJECT_ID" \
    --region="$REGION" > /dev/null 2>&1; then
    echo "  Job exists — updating ..."
    gcloud run jobs update "$JOB_NAME" \
        --project="$PROJECT_ID" \
        --region="$REGION" \
        --image="$IMAGE_URI" \
        --set-env-vars="GCS_BUCKET=${GCS_BUCKET},RUN_NAME=${RUN_NAME}" \
        --gpu=1 \
        --gpu-type=nvidia-l4 \
        --no-gpu-zonal-redundancy \
        --cpu=8 \
        --memory=32Gi \
        --task-timeout=3600s \
        --max-retries=0
else
    echo "  Creating new job ..."
    gcloud run jobs create "$JOB_NAME" \
        --project="$PROJECT_ID" \
        --region="$REGION" \
        --image="$IMAGE_URI" \
        --set-env-vars="GCS_BUCKET=${GCS_BUCKET},RUN_NAME=${RUN_NAME}" \
        --gpu=1 \
        --gpu-type=nvidia-l4 \
        --no-gpu-zonal-redundancy \
        --cpu=8 \
        --memory=32Gi \
        --task-timeout=3600s \
        --max-retries=0
fi

# ── Step 4: Execute the job ──────────────────────────────────
echo "[4/4] Executing job ..."
gcloud run jobs execute "$JOB_NAME" \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --wait

echo ""
echo "=== Job complete ==="
echo "Results at: gs://${GCS_BUCKET}/${RUN_NAME}/"
echo "Download:   gsutil -m cp -r gs://${GCS_BUCKET}/${RUN_NAME}/ ./data/runs/"
