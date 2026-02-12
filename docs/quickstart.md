# Quickstart

## Prerequisites

- [Ollama](https://ollama.com) running locally with at least one model pulled
- Python 3.10+ (recommended: use the provided mamba environment)

## Installation

```bash
# Create and activate the environment
mamba env create -f environment.yml
mamba activate polibias

# Install the package in editable mode
pip install -e .
```

## Running the pipeline

```bash
python -m polibias scrape    # fetch articles from RTS
python -m polibias score     # send to Ollama models
python -m polibias analyse   # build CSV datasets
python -m polibias check     # verify pipeline save paths
python -m polibias viz       # launch Streamlit dashboard
```

## Data flow

```
data/input_files/some_rts_links.csv   → URLs to scrape
data/webdata/*.json                   → parsed article content
data/results/{model}/{run}/*.json     → per-article bias scores
data/bias_data.csv                    → aggregated bias results
data/web_data.csv                     → cleaned article metadata
```

`src/polibias` uses `data/` as the only canonical output location.

## Configuration

Edit `src/polibias/config.py` to change:

- **models**: which Ollama models to use
- **runs**: number of scoring runs per model (default 6)
- **ollama_options**: temperature, context length, etc.
- **prompt template**: `src/polibias/prompt.md`
