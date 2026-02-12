# polibias

Political bias scoring of Swiss news articles using local LLMs via [Ollama](https://ollama.com).

polibias scrapes articles from [RTS](https://www.rts.ch) (Radio Television Suisse), sends them through multiple small language models, and scores each article for political bias along four dimensions. Running each model multiple times reveals how **consistent** (or volatile) each model's judgments are.

## Bias dimensions

| Dimension | What it measures |
|---|---|
| `subject_bias` | Does the topic selection itself lean left or right? |
| `framing_bias` | Is the narrative or tone left- or right-leaning? |
| `treatment_bias` | Does the article treat one side more favorably? |
| `guests_bias` | Are quoted voices more left or more right? |

Scores range from **-1.0** (left-leaning) to **+1.0** (right-leaning). An `overall_bias` is computed as the mean of all four.

## Setup

### Prerequisites

- **Ollama** running locally (`ollama serve`) with models pulled:
  ```bash
  ollama pull llama3.2:latest
  ollama pull gemma2:latest
  ollama pull phi3:mini
  ollama pull qwen2.5:3b-instruct
  ollama pull gemma3:4b
  ```

### Install with mamba (recommended)

```bash
mamba env create -f environment.yml
mamba activate polibias
pip install -e .
```

### Install with pip only

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Runbook

Edit only these files:
- `src/polibias/config.py` for model/runtime settings
- `data/input_files/some_rts_links.csv` for RTS URLs
- `src/polibias/prompt.md` if you want to change the scoring prompt

Run only these commands:

```bash
python -m polibias scrape
python -m polibias score
python -m polibias analyse
python -m polibias check
python -m polibias viz
```

The pipeline is **idempotent** — it skips any work whose output already exists on disk. Re-running is safe and fast.

Get outputs here:
- `data/webdata/*.json` scraped article content
- `data/results/{model}/{run}/*.json` model scores
- `data/bias_data.csv` aggregated bias rows
- `data/web_data.csv` cleaned article metadata

## Project structure

```
polibias/
├── src/polibias/           # Python package
│   ├── __main__.py         # CLI entry point
│   ├── config.py           # Settings dataclass
│   ├── scraper.py          # RTS article fetcher + parser
│   ├── scoring.py          # Ollama model scoring
│   ├── analysis.py         # DataFrame aggregation
│   ├── dashboard.py        # Streamlit visualization
│   └── prompt.md           # Prompt template
├── data/
│   ├── input_files/        # URL list (some_rts_links.csv)
│   ├── webdata/            # Scraped article JSONs
│   ├── results/            # Model output JSONs (model/run/article.json)
│   ├── bias_data.csv       # Aggregated bias scores
│   └── web_data.csv        # Cleaned article metadata
├── docs/                   # Sphinx documentation (ReadTheDocs-ready)
├── environment.yml         # Mamba environment spec
├── pyproject.toml          # Package metadata
└── requirements.txt        # Pip dependencies
```

## Configuration

Edit `src/polibias/config.py` to change models, number of runs, Ollama options (temperature, context length), or timeout. The prompt template lives at `src/polibias/prompt.md`.

Legacy prototype code has been moved to `archive/app_legacy/` and is not used by the packaged pipeline.

## Dashboard

The Streamlit dashboard provides interactive visualizations:

- Bias scatter plots by model and by article
- Box plots showing per-model variance
- Heatmap of mean bias across models and articles
- Confidence vs. absolute bias scatter

## Documentation

Full docs are built with Sphinx and hosted on ReadTheDocs:

```bash
pip install -e ".[docs]"
cd docs && make html
```

## License

MIT
