# Polibias — Codebase Review

A candid review of the polibias module: what works well, what doesn't, and where it could go next.

---

## Things Done Well

### Reproducibility is first-class

Every scoring result includes a SHA-256 hash of the prompt template and the model digest. Combined with timestamped error logs and run-directory isolation, any result can be traced back to the exact prompt, model version, and conditions that produced it. This is rare in hobby projects and sets a strong foundation for credible analysis.

### Clean pipeline architecture

The scrape → score → analyse → stats → export pipeline is well-decomposed. Each module has a single responsibility, no circular dependencies, and can be run independently via the CLI. The `Settings` frozen dataclass centralises all configuration in one immutable object, preventing accidental mutation across threads.

### Idempotent operations

Scraping skips already-fetched articles. Scoring skips already-scored (model, run, article) combinations. Analysis rebuilds from raw sources. This means a crash mid-run doesn't lose work, and re-runs are cheap. Good design choice for a tool that depends on slow LLM inference.

### Thoughtful error resilience in scoring

The three-tier fallback (call → JSON-repair retry → blank fallback) with structured JSONL error logging and raw output preservation is well-engineered. When a model returns garbage, the pipeline doesn't crash — it logs everything needed for a post-mortem and moves on.

### Correct statistical methods

ICC(1,1) for within-model consistency, Fleiss' kappa for inter-model agreement, and t-distribution confidence intervals are all correctly implemented. The discretisation into 5 equal-width bins for kappa is a reasonable choice for a [-1, 1] scale.

### Parallel scoring with memory awareness

Models are loaded sequentially (one in VRAM at a time), while articles within each model-run are scored in parallel with ThreadPoolExecutor. This balances throughput against the constraint that local LLMs are memory-heavy.

### Professional HTML report

The standalone HTML report with interactive Plotly charts, styled tables, kappa interpretation legend, and forest plot is genuinely useful output. It's self-contained, mobile-responsive, and opens in the browser automatically.

---

## Things Done Poorly

### Almost no tests

Eight tests across three files. The scraper has zero tests — parsing HTML from a live news site is the most fragile part of the system and the most in need of regression tests. The stats module (ICC, kappa) has no unit tests despite implementing non-trivial formulas. Config validation, edge cases in analysis, and the export module are all untested.

### Scraper is brittle with no retry logic

The scraper uses raw `urllib` with no connection pooling, no retry logic, and no exponential backoff. A single transient network error kills that article's scrape. Meanwhile, the scoring module has a robust retry mechanism — the scraper deserves the same treatment.

### `prompt_template` re-reads disk on every call

`Settings.prompt_template` is a property that reads `prompt.md` from disk each time it's accessed. Since it's called once per article per model per run inside a ThreadPoolExecutor, this means hundreds of redundant disk reads. It should be read once and cached.

### No timeout on ollama.Client

The `REVIEW.md` flags this and it's still unfixed: `ollama.Client` is constructed without `timeout=settings.timeout`, so a hung model blocks a worker thread indefinitely. The timeout setting exists but isn't wired in.

### Magic numbers scattered through the code

`6000` chars for article truncation, `2048` for context window, `5` for kappa bins, `"5m"` for keep_alive — these are buried in function bodies instead of living in `Settings` or as named constants. Makes tuning harder than it should be.

### No external config file support

All configuration is hardcoded in `config.py`. Changing models, timeouts, or worker counts requires editing source code. A `polibias.toml` or `--config` flag would let users run experiments with different settings without touching the codebase.

### Dead code and imports

`from plotly.subplots import make_subplots` is unused in `export.py`. The `requests` package was removed from imports but may still linger in `requirements.txt` or `environment.yml`. Small things, but they signal incomplete cleanup.

### HTML report depends on CDN

Plotly charts load JavaScript from a CDN, so the report requires internet access to render properly. For a tool focused on reproducibility and local execution, this is an odd dependency. Plotly supports embedding the JS bundle for offline use.

### Scraper is tightly coupled to RTS layout

CSS selectors like `.article-lead`, class-based stop conditions, and JSON-LD parsing assumptions are all specific to the current RTS website structure. Any layout change breaks scraping silently (returns partial/empty data rather than erroring). There's no schema validation on scraped output.

### CLI lacks `--verbose` / `--quiet` flags

The output is fixed at one verbosity level. No way to get debug logging or suppress output for scripted use. Logging uses `print()` and `warnings.warn()` rather than Python's `logging` module.

---

## Feature Ideas

**Multi-source scraping** — Extend beyond RTS to other Swiss news outlets (SRF, Swissinfo, Le Temps, 20 Minutes). A pluggable scraper interface with per-source parsers would make bias comparisons across outlets possible.

**Prompt experimentation framework** — Support multiple prompt templates per run and track which prompt produced which scores. Would enable systematic prompt engineering — e.g. testing whether longer instructions improve model agreement.

**Temporal analysis** — Score the same outlet's articles over weeks/months and plot bias trends over time. The run-directory structure already supports this; it just needs a comparison/aggregation layer.

**Model calibration benchmarks** — Score a set of articles with known political leanings (party press releases, editorials with declared positions) to calibrate each model's bias scale. Would help interpret whether a score of 0.3 is meaningful.

**Live dashboard** — A lightweight web server (FastAPI or similar) that watches the run directory and updates charts in real-time as scoring progresses. More useful than the current post-hoc HTML report for long runs.

**Confidence-weighted aggregation** — Models report a confidence score but it's only displayed, never used. Weight each model's contribution to the aggregate by its self-reported confidence, or filter out low-confidence scores.

**Comparative report mode** — Generate a side-by-side report comparing two runs (different models, different prompts, different time periods). Highlight where models disagree most and which articles are most contentious.

**Offline Plotly embedding** — Bundle plotly.min.js into the HTML report so it works without internet. One config flag, significant usability improvement.
