# Running polibias on Google Cloud Run (GPU)

Step-by-step guide. Run every command from your terminal.

---

## 1. Prerequisites (one-time setup)

### Install the gcloud CLI

If you don't have it yet:

```bash
# Download and install
curl https://sdk.cloud.google.com | bash

# Restart your shell, then initialize
gcloud init
```

This will open a browser to log in with your Google account.

### Create a Google Cloud project (if you don't have one)

Go to https://console.cloud.google.com and create a new project, or use an existing one.
Note your **Project ID** (e.g. `my-project-123456`).

### Enable billing

GPU workloads require a billing account. Go to:
https://console.cloud.google.com/billing

Link a billing account to your project.

---

## 2. Enable required APIs

Open a terminal and run:

```bash
export GCP_PROJECT="your-project-id-here"

gcloud config set project "$GCP_PROJECT"

gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    storage.googleapis.com
```

---

## 3. Request GPU quota (first time only)

Cloud Run GPU access may need to be requested. Check at:
https://console.cloud.google.com/iam-admin/quotas

Filter for "Cloud Run" and "NVIDIA L4 GPU". If the limit is 0, click **Edit Quotas** and request 1. Approval usually takes a few minutes to a few hours.

Alternatively, you can just try running the deploy and see if it works — some projects have GPU access enabled by default.

---

## 4. Create a GCS bucket for results

```bash
export GCS_BUCKET="polibias-results-${GCP_PROJECT}"

gsutil mb -l europe-west4 "gs://${GCS_BUCKET}"
```

Pick a region that has L4 GPUs. `europe-west4` (Netherlands) is a good option for Europe.

---

## 5. Navigate to the project

```bash
cd /home/tge/gpt/polibias
```

---

## 6. Build, deploy, and run

Everything is wrapped in the deploy script. Set your env vars and run it:

```bash
export GCP_PROJECT="your-project-id-here"
export GCS_BUCKET="polibias-results-${GCP_PROJECT}"
export GCP_REGION="europe-west4"
export RUN_NAME="cloud_run"

./cloud/deploy.sh
```

This will:
1. Create an Artifact Registry repo (if needed)
2. Build the Docker image with Cloud Build (~10 min first time, models are ~15 GB)
3. Create a Cloud Run Job with an L4 GPU
4. Execute the job and wait for it to finish (~30-45 min)

You can also monitor progress in the console at:
https://console.cloud.google.com/run/jobs

---

## 7. Download results

Once the job completes:

```bash
gsutil -m cp -r "gs://${GCS_BUCKET}/${RUN_NAME}/" ./data/runs/
```

Then view the report locally:

```bash
polibias viz --run-dir cloud_run
```

---

## 8. Re-run with a different run name

To do another run (e.g. with different articles), just change `RUN_NAME`:

```bash
export RUN_NAME="cloud_run_v2"
./cloud/deploy.sh
```

The Docker image is cached, so subsequent deploys are much faster (~1-2 min to start the job).

---

## Cost estimate

| Component | Cost |
|---|---|
| Cloud Build (image build) | ~$0.10 (first time, cached after) |
| Cloud Run Job (L4 GPU, ~30-45 min) | ~$0.50-0.80 |
| GCS storage | < $0.01 |
| **Total per run** | **~$0.50-1.00** |

---

## Troubleshooting

### "GPU quota exceeded" or "NVIDIA L4 not available"
Request quota (step 3) or try a different region:
```bash
export GCP_REGION="us-central1"
```

### "Permission denied" on Cloud Build
Grant the Cloud Build service account permissions:
```bash
PROJECT_NUM=$(gcloud projects describe "$GCP_PROJECT" --format='value(projectNumber)')
gcloud projects add-iam-policy-binding "$GCP_PROJECT" \
    --member="serviceAccount:${PROJECT_NUM}@cloudbuild.gserviceaccount.com" \
    --role="roles/artifactregistry.writer"
```

### Job fails / want to see logs
```bash
gcloud run jobs executions list --job=polibias-run --region="$GCP_REGION"
# Then pick the execution name and view logs:
gcloud run jobs executions logs read EXECUTION_NAME --region="$GCP_REGION"
```

### Want to delete everything when done
```bash
gcloud run jobs delete polibias-run --region="$GCP_REGION"
gcloud artifacts repositories delete polibias --location="$GCP_REGION"
gsutil rm -r "gs://${GCS_BUCKET}"
```
