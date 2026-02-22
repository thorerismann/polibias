# Quickstart

## Prerequisites

- [Ollama](https://ollama.com) running locally with at least one model pulled
- Python 3.10+

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Run the pipeline

```bash
python -m polibias all --run-dir baseline
```

Or step-by-step:

```bash
python -m polibias scrape
python -m polibias score --run-dir baseline
python -m polibias analyse --run-dir baseline
python -m polibias stats --run-dir baseline
python -m polibias export --run-dir baseline
python -m polibias check --run-dir baseline
```

If `--run-dir` is omitted, the default is `run_results`.

## Data flow

```text
data/input_files/rts_links.txt        -> URLs to scrape
data/webdata/*.json                        -> parsed article content (shared)
data/runs/<run_dir>/results/...            -> per-article model scores
data/runs/<run_dir>/bias_data.csv          -> aggregated scores
data/runs/<run_dir>/web_data.csv           -> cleaned article metadata
data/runs/<run_dir>/stats_report.csv       -> statistical summary
data/runs/<run_dir>/report.html            -> HTML report
data/runs/<run_dir>/errors/errors.jsonl    -> structured errors
data/runs/<run_dir>/errors/raw_outputs/*   -> raw failed model outputs
```

## Configuration

Edit `src/polibias/config.py` to change:

- `models`
- `runs`
- `ollama_options`
- `timeout` / `parse_retries`

Prompt template:

- `src/polibias/prompt.md`
