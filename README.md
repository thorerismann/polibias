# polibias

Political bias scoring of Swiss news articles using local LLMs via [Ollama](https://ollama.com).

`polibias` scrapes RTS articles, scores them across four bias dimensions with local models, and produces CSV/statistics/HTML outputs.

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
python -m polibias all --run-dir temp_02_ctx2k
python -m polibias all --run-dir temp_08_ctx4k
```

You can run individual steps too:

```bash
python -m polibias scrape
python -m polibias score --run-dir exp_a
python -m polibias analyse --run-dir exp_a
python -m polibias stats --run-dir exp_a
python -m polibias export --run-dir exp_a
python -m polibias bambi --run-dir exp_a
python -m polibias viz --run-dir exp_a
python -m polibias check --run-dir exp_a
```

Bayesian audit extras (optional):

```bash
pip install -e "[bayes]"
python -m polibias bambi --run-dir exp_a
```

## Output Layout

Shared scraped content:

- `data/webdata/*.json`

Per-run outputs:

- `data/runs/<run_dir>/results/<model>/<run>/*.json`
- `data/runs/<run_dir>/errors/errors.jsonl`
- `data/runs/<run_dir>/errors/raw_outputs/*.txt` (raw failed model responses)
- `data/runs/<run_dir>/bias_data.csv`
- `data/runs/<run_dir>/web_data.csv`
- `data/runs/<run_dir>/stats_report.csv`
- `data/runs/<run_dir>/report.html`
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
