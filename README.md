# polibias

Political bias scoring of Swiss news articles using local LLMs via [Ollama](https://ollama.com).

`polibias` scrapes RTS/Federalist/Jacobin articles, scores them across four bias dimensions with local models, and produces CSV/statistics/HTML outputs.

## Bias Dimensions

| Dimension | What it measures |
|---|---|
| `subject_bias` | Does the topic selection itself lean left or right? |
| `framing_bias` | Is the narrative or tone left- or right-leaning? |
| `treatment_bias` | Does the article treat one side more favorably? |
| `guests_bias` | Are quoted voices more left or more right? |

Scores are in `[-1.0, +1.0]` where negative means left-leaning and positive means right-leaning.

## Setup

### Prerequisites

- Ollama running locally (`ollama serve`)
- Python 3.10+

Pull the default models from `src/polibias/config.py` (example):

```bash
ollama pull llama3.2:latest
ollama pull gemma2:latest
ollama pull phi3:mini
ollama pull qwen2.5:3b-instruct
ollama pull gemma3:4b
```

### Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Runbook

`--run-dir` selects the output folder name under `data/runs/`.

If omitted, it defaults to `run_results`.

```bash
polibias run --run-dir temp_02_ctx2k
polibias run --run-dir temp_08_ctx4k
```

You can run individual grouped commands too:

```bash
polibias scrape --source rts
polibias scrape --source the_federalist --limit 20
polibias scrape --source jacobin --limit 20

polibias score --run-dir exp_a --source all
polibias score --run-dir exp_a --source rts
polibias score --run-dir exp_a --source the_federalist
polibias score --run-dir exp_a --source jacobin

polibias analyze --run-dir exp_a
polibias stats --run-dir exp_a
polibias export --run-dir exp_a
polibias bambi analyze --run-dir exp_a
polibias bambi viz --run-dir exp_a

polibias viz --run-dir exp_a                     # report.html
polibias viz --run-dir exp_a --source rts
polibias viz --run-dir exp_a --source the_federalist
polibias viz --run-dir exp_a --source jacobin
polibias viz --run-dir exp_a --source all        # report_all.html

polibias check --run-dir exp_a
```

Legacy commands (`all`, `analyse`, `score-rts`, `viz-fed`, etc.) are still accepted.

Bayesian audit extras (optional):

```bash
pip install -e "[bayes]"
polibias bambi --run-dir exp_a
```

Streamlit control panel (optional):

```bash
pip install -e "[ui]"
streamlit run app/streamlit_app.py
```

## Output Layout

Shared scraped content:

- `data/webdata/rts/*.json`
- `data/webdata/the_federalist/*.json`
- `data/webdata/jacobin/*.json`

Per-run outputs:

- `data/runs/<run_dir>/rts_results/<model>/<run>/*.json`
- `data/runs/<run_dir>/the_federalist_results/<model>/<run>/*.json`
- `data/runs/<run_dir>/jacobin_results/<model>/<run>/*.json`
- `data/runs/<run_dir>/errors/errors.jsonl`
- `data/runs/<run_dir>/errors/raw_outputs/*.txt` (raw failed model responses)
- `data/runs/<run_dir>/bias_data.csv`
- `data/runs/<run_dir>/web_data.csv`
- `data/runs/<run_dir>/stats_report.csv`
- `data/runs/<run_dir>/report.html`
- `data/runs/<run_dir>/report_rts.html`
- `data/runs/<run_dir>/report_fed.html`
- `data/runs/<run_dir>/report_jacobin.html`
- `data/runs/<run_dir>/report_all.html`
- `data/runs/<run_dir>/article_summaries.csv`
- `data/runs/<run_dir>/bias_table.tex`

## Configuration

Edit `src/polibias/config.py` to change models, runs, timeouts, and Ollama options.

Prompt template:

- `src/polibias/prompt.md`

Input URL file:

- `data/input_files/some_rts_links.csv`

## Tests

Install dev deps and run:

```bash
pip install -e "[dev]"
python -m pytest -q
```

## License

MIT
