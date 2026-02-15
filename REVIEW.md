# Change Review — feat/improvements

Honest assessment of every change, including bugs found on re-read.

---

## 1. Switch from `requests` to `ollama` Python package

**Status: MOSTLY DONE — has a bug**

What works:
- `ollama.Client(host=settings.ollama_host).generate()` replaces the raw `requests.post()` correctly
- `format="json"` forces Ollama to output valid JSON, which eliminates the entire `parse_json_from_model` heuristic repair chain (leading zeros, truncation repair, markdown fence stripping)
- The old 3-tier fallback (parse → fix → repair) is replaced by a simple `json.loads()` since JSON mode guarantees syntactically valid output
- `keep_alive="5m"` prevents Ollama from unloading the model between articles

Bug found:
- **No timeout on the ollama Client.** The old code passed `timeout=settings.timeout` (200s) to `requests.post()`. The new `call_ollama()` doesn't set any timeout on the client or the generate call. If Ollama hangs, the worker thread blocks forever. Fix: pass `timeout=settings.timeout` to `ollama.Client()` constructor.

Note:
- `format="json"` guarantees *syntactically* valid JSON but not *schema-correct* JSON. A model can still return `{"lol": true}` instead of the expected keys. The code handles this correctly — `_format_result()` fills missing required keys with `None` and the `status` field tracks whether it was `ok` or `fallback`.

**Verdict: Needs the timeout fix, otherwise solid.**

---

## 2. JSON mode (`format="json"`)

**Status: ACCEPTED**

This is the single biggest quality-of-life improvement. With the old raw-requests approach, every model could return markdown-wrapped JSON, truncated JSON, JSON with leading zeros, or just garbage. The `parse_json_from_model` function was a fragile best-effort repair tool.

With `format="json"`, Ollama constrains the model's output at the token-sampling level to produce valid JSON. The retry loop is still there as a belt-and-suspenders fallback, but it should almost never fire.

The old code this replaced:
- `_strip_markdown_json()` — gone (Ollama won't wrap in fences)
- `_fix_leading_zeros()` — gone (Ollama constrains number tokens)
- truncation repair with quote/brace balancing — gone
- 3-stage parse cascade — gone

All replaced by `json.loads(raw)`. Correct.

**Verdict: Accepted. Clean improvement.**

---

## 3. Article truncation guard

**Status: ACCEPTED**

`_truncate_article(body, max_article_chars=6000)` caps article text before it's inserted into the prompt. With `num_ctx: 2048` tokens and ~4 chars/token for French text, 6000 chars ≈ 1500 tokens, leaving ~500 tokens for the prompt template and response.

This is a rough heuristic — character count isn't token count. But it's the right tradeoff: cheap, no extra dependency, prevents the real failure mode (article exceeding context window and getting silently truncated by Ollama, which corrupts the scoring).

**Verdict: Accepted. Could be improved with actual tokenization later but works.**

---

## 4. Scraper timeout fix

**Status: ACCEPTED — but design note**

`fetch_rts_soup()` now takes a `timeout` parameter (default 30s) instead of hardcoded 15s. `scrape_articles()` passes `settings.timeout` through.

Design concern: `settings.timeout` is 200 seconds — designed for Ollama inference, not HTTP scraping. A 200s timeout on a web request is very generous. Should probably have a separate `scrape_timeout` field in Settings, or just use a smaller default. Not a bug — it works — but you'd never want to wait 200s for an RTS page.

**Verdict: Accepted. Consider adding a separate `scrape_timeout` to Settings later.**

---

## 5. Parallel article scoring (ThreadPoolExecutor)

**Status: ACCEPTED — two minor thread-safety notes**

The parallelization pattern is correct:
```
for model in models:          # sequential → one model in RAM
    for run in runs:          # sequential
        ThreadPoolExecutor    # parallel articles → same model, queued
```

This is the safe pattern we discussed. `max_workers=4` is conservative. Ollama handles concurrent requests to the same loaded model via its internal queue.

Thread-safety notes:
1. **`_log_error()` file appends**: Multiple threads can call `_log_error()` simultaneously. The `open("a") + write()` pattern relies on OS-level atomicity for small writes (< PIPE_BUF = 4096 bytes on Linux). In practice this won't corrupt data, but it's technically a race condition. A threading Lock would be cleaner.
2. **`settings.prompt_template` re-reads disk**: It's a property that calls `read_text()` every time. With 4 workers scoring simultaneously, each reads prompt.md from disk on every article. Not dangerous (read-only) but wasteful. Could cache the template string once.

Neither of these will cause actual failures in practice.

**Verdict: Accepted. Works correctly under real-world conditions.**

---

## 6. Validation module (`validation.py`)

**Status: ACCEPTED — one fragile spot**

`check_ollama_reachable()`, `check_models_available()`, `check_input_data()` — all straightforward and correct.

Fragile spot: `check_model_digests()` uses `getattr(info, "digest", None)` on the response from `ollama.show()`. The response schema varies by ollama package version. It might return "unknown" for every model instead of the actual digest. Not harmful — it's just a reproducibility log — but might not produce useful output without testing against the actual installed version.

**Verdict: Accepted. Digest extraction may need adjustment based on actual package version.**

---

## 7. Stats module (`stats.py`)

**Status: HAS BUGS — needs rework**

The confidence interval computation (`compute_model_ci`) is correct — standard t-distribution CI, scipy implementation is proper.

**Bug 1 — ICC implementation is degenerate:**

`icc_oneway()` receives a 1-D array of scores from *one model on one article* (e.g., 4 runs = 4 values). ICC(1,1) is designed for n subjects rated by k raters. With 1 subject and k raters, the formula is degenerate (n_sub=1 makes the between-subjects variance meaningless).

The correct approach: for each model, build a 2-D matrix (articles × runs) and compute ICC across all articles. Articles are the subjects, runs are the raters. This gives one ICC value per model that actually measures within-model consistency across the full dataset.

**Bug 2 — Fleiss' kappa per article is degenerate:**

`compute_fleiss_kappa_per_article()` creates a 1-row rating matrix per article (1 subject, n models as raters). Fleiss' kappa with 1 subject collapses to a trivial calculation. The result is technically computable but statistically meaningless.

The correct approach: build the full (articles × bins) rating matrix where each model contributes one rating per article, then compute one global Fleiss' kappa. Or compute per-bias-dimension across all articles.

**Verdict: CI computation is correct. ICC and Fleiss' kappa need to be restructured to operate on the full (articles × raters) matrix, not per-article single-row slices.**

---

## 8. Export module (`export.py`)

**Status: ACCEPTED — minor issues**

The HTML report, article summaries CSV, and LaTeX table all work correctly.

Minor issues:
1. **Dead import**: `from plotly.subplots import make_subplots` is imported but never used.
2. **HTML needs internet**: `include_plotlyjs="cdn"` loads Plotly.js from CDN. For a truly standalone report (e.g., emailing to someone offline), use `include_plotlyjs=True` to embed the full library (~3MB larger but self-contained).
3. **LaTeX escaping**: Model names are inserted raw into the LaTeX table. If a model name contains `_` or `&` (LaTeX special chars), the `.tex` file won't compile. Current model names are safe (`llama3.2:latest`, `gemma2:latest`, etc.) but this is fragile.

**Verdict: Accepted. Fix the dead import. Consider `include_plotlyjs=True` for truly offline reports.**

---

## 9. Dashboard errors tab

**Status: ACCEPTED**

Clean implementation. Loads `errors.jsonl`, shows filterable dataframe, bar charts by stage and model. Handles the no-errors case correctly (shows "No errors recorded").

**Verdict: Accepted. No issues found.**

---

## 10. Config changes

**Status: ACCEPTED**

- `ollama_url` → `ollama_host`: correct (ollama package uses host, not full URL)
- `max_workers: int = 4`: conservative default
- `max_article_chars: int = 6000`: reasonable heuristic
- `prompt_hash` property: SHA-256 of prompt template, stored in every result JSON for reproducibility

**Verdict: Accepted.**

---

## 11. CLI updates (`__main__.py`)

**Status: ACCEPTED — one behavior change to note**

New commands `validate`, `stats`, `export` all wire up correctly. The `all` command now runs validation first, which means `python -m polibias` will fail fast if Ollama isn't running. This is a good behavior change.

Note: validation failure calls `sys.exit(1)`, so if Ollama is down, `all` won't proceed to scraping (which doesn't need Ollama). This might surprise someone who just wants to scrape articles. Minor — they can just run `python -m polibias scrape` directly.

**Verdict: Accepted.**

---

## 12. Streamlit — NOT removed

**Status: NOT ADDRESSED**

The user expressed interest in removing Streamlit in favor of the standalone HTML report. Streamlit is still in the codebase and in the dependencies. The HTML export in `export.py` duplicates most of the dashboard's charts. If you want to drop Streamlit:
- Remove `dashboard.py`
- Remove `streamlit>=1.30` from all three dependency files
- Remove the `viz` CLI command
- The HTML report from `export.py` becomes the primary visualization

This is a clean cut since nothing else depends on Streamlit.

**Verdict: Pending your decision. Both can coexist, or Streamlit can be dropped cleanly.**

---

## 13. Dead dependency: `requests`

**Status: NOTED**

After switching `scoring.py` to the `ollama` package, nothing imports `requests` anymore. `scraper.py` uses `urllib.request` (stdlib). The `ollama` package uses `httpx` internally. `requests>=2.31` is now dead weight in the dependency files.

Not harmful, but could be cleaned up.

**Verdict: Can be removed from pyproject.toml, requirements.txt, and environment.yml.**

---

## Summary table

| Change | Status | Action needed |
|---|---|---|
| ollama package switch | Bug | Add `timeout=settings.timeout` to Client |
| JSON mode | Accepted | None |
| Article truncation | Accepted | None |
| Scraper timeout | Accepted | Consider separate `scrape_timeout` |
| Parallel scoring | Accepted | Optional: add Lock for error log |
| Validation module | Accepted | Digest extraction may need version check |
| Stats module | Buggy | Rework ICC and Fleiss' kappa granularity |
| Export module | Accepted | Remove dead import, consider embedded plotly.js |
| Dashboard errors tab | Accepted | None |
| Config changes | Accepted | None |
| CLI updates | Accepted | None |
| Streamlit removal | Not done | Pending decision |
| `requests` dependency | Dead | Can remove |
