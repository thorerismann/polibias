# Architecture

## Pipeline overview

polibias runs a three-stage pipeline:

### 1. Scraping (`polibias.scraper`)

Fetches HTML from RTS article URLs, extracts structured content using
BeautifulSoup and JSON-LD metadata, and saves each article as a JSON file.

### 2. Scoring (`polibias.scoring`)

For each article, for each model, for N independent runs:

- Loads the article body text
- Fills the prompt template with the article
- Sends the prompt to Ollama's `/api/generate` endpoint
- Parses the JSON response (with fallback repair for malformed output)
- Saves the per-article score to disk

Results are organized as `data/results/{model}/{run}/{article_id}.json`.

### 3. Analysis (`polibias.analysis`)

Collects all score JSONs into a single pandas DataFrame, computes the
`overall_bias` as the mean of the four bias dimensions, and exports CSVs.

Canonical aggregate outputs are `data/bias_data.csv` and `data/web_data.csv`.
Use `python -m polibias check` to verify expected files exist.

## Bias dimensions

| Dimension        | Question                                        |
|------------------|-------------------------------------------------|
| `subject_bias`   | Does the topic selection lean left or right?     |
| `framing_bias`   | Is the narrative left- or right-leaning?         |
| `treatment_bias` | Does the article treat one side more favorably?  |
| `guests_bias`    | Are quoted voices more left or more right?       |

All scores are in `[-1.0, +1.0]` where negative = left, positive = right.

## Idempotency

Every step checks for existing outputs before doing work:

- **Scraping** skips articles whose JSON already exists in `webdata/`
- **Scoring** skips article/model/run combinations whose JSON already exists
- **Analysis** always rebuilds CSVs from whatever results are on disk
