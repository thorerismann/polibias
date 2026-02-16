# Architecture

## Pipeline overview

`polibias` runs a three-stage pipeline:

1. Scraping (`polibias.scraper`)
2. Scoring (`polibias.scoring`)
3. Analysis/export (`polibias.analysis`, `polibias.stats`, `polibias.export`)

### 1. Scraping

- Reads URLs from `data/input_files/some_rts_links.csv`
- Fetches/parses RTS pages
- Writes article JSON to shared storage: `data/webdata/*.json`

### 2. Scoring

For each article, model, and run:

- Builds a prompt from `src/polibias/prompt.md`
- Calls Ollama in JSON mode
- Parses/validates output shape
- Retries parse failures with a repair prompt
- Writes score JSON to run-specific results directory

Score outputs:

- `data/runs/<run_dir>/results/<model>/<run>/<article_id>.json`

Error outputs:

- `data/runs/<run_dir>/errors/errors.jsonl`
- `data/runs/<run_dir>/errors/raw_outputs/*.txt`

### 3. Analysis/export

Reads run-specific results and writes run-specific derived outputs:

- `bias_data.csv`
- `web_data.csv`
- `stats_report.csv`
- `report.html`
- `article_summaries.csv`
- `bias_table.tex`

All under:

- `data/runs/<run_dir>/`

## Run directory model

The CLI option `--run-dir` selects the run folder name under `data/runs/`.

Examples:

- `--run-dir baseline`
- `--run-dir temp_02_ctx2k`

If omitted, run directory name defaults to `run_results`.

## Idempotency

- Scraping skips already-saved article JSON files in `data/webdata/`
- Scoring skips existing `(model, run, article)` result files in selected run dir
- Analysis/export rebuild from the selected run dir’s results
