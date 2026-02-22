"""Send articles to Ollama models and collect bias scores.

Uses the ``ollama`` Python package with ``format='json'`` for guaranteed
valid JSON output.  Articles within a single model/run are scored in
parallel via ThreadPoolExecutor (safe — single model loaded in Ollama,
no extra RAM).
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import ollama
from tqdm import tqdm

_REQUIRED_KEYS = [
    "subject_bias",
    "framing_bias",
    "treatment_bias",
    "guests_bias",
    "confidence",
    "comment",
]


# ---------- Ollama interaction ----------

def _get_client(settings) -> ollama.Client:
    return ollama.Client(host=settings.ollama_host, timeout=settings.timeout)


def call_ollama(
    client: ollama.Client, model: str, *, system: str, prompt: str, settings,
) -> str:
    """Call Ollama with JSON mode enabled.

    *system* carries the scoring instructions (identity + rules).
    *prompt* carries only the article text to evaluate.
    """
    resp = client.generate(
        model=model,
        system=system,
        prompt=prompt,
        format="json",
        options=settings.ollama_options,
        keep_alive=settings.keep_alive,
    )
    return resp.response


# ---------- Article truncation ----------

def _truncate_article(body: str, max_chars: int) -> str:
    """Truncate article text to stay within context limits."""
    if len(body) <= max_chars:
        return body
    return body[:max_chars] + "\n\n[... article truncated for model context limit ...]"


# ---------- Result formatting ----------

def _blank_result() -> dict[str, Any]:
    return {k: None for k in _REQUIRED_KEYS}


def _format_result(
    data: dict[str, Any],
    *,
    status: str,
    model: str,
    article: str,
    prompt_hash: str = "",
) -> dict[str, Any]:
    out = _blank_result()
    for k in _REQUIRED_KEYS:
        if k in data:
            out[k] = data.get(k)
    out["status"] = status
    out["model"] = model
    out["article"] = article
    out["prompt_hash"] = prompt_hash
    return out


# ---------- Error logging ----------

def _log_error(settings, payload: dict[str, Any]) -> None:
    settings.errors_dir.mkdir(parents=True, exist_ok=True)
    log_path = settings.errors_dir / "errors.jsonl"
    payload = dict(payload)
    payload["ts"] = datetime.now(timezone.utc).isoformat()
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload))
        f.write("\n")


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)


def _save_failed_raw(
    settings,
    *,
    stage: str,
    model: str,
    article: str,
    run: int,
    raw: str,
    attempt: int | None = None,
) -> Path:
    raw_dir = settings.errors_dir / "raw_outputs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    attempt_part = f"_attempt{attempt}" if attempt is not None else ""
    fname = (
        f"{ts}_{_safe_name(stage)}_{_safe_name(model)}_run{run}_"
        f"{_safe_name(article)}{attempt_part}.txt"
    )
    out_path = raw_dir / fname
    out_path.write_text(raw, encoding="utf-8")
    return out_path


def _truncate_raw(raw: str, limit: int = 4000) -> str:
    if len(raw) <= limit:
        return raw
    head = raw[:2000]
    tail = raw[-1500:]
    return f"{head}\n...\n{tail}"


def _retry_prompt(raw: str) -> str:
    return (
        "You returned invalid JSON previously. "
        "Return ONLY a valid JSON object with keys: "
        "subject_bias, framing_bias, treatment_bias, guests_bias, confidence, comment. "
        "If you are unsure, set values to null. No markdown, no extra text.\n\n"
        "INVALID_OUTPUT:\n"
        f"{_truncate_raw(raw)}"
    )


# ---------- Single-article scoring ----------

def score_one_article(
    client: ollama.Client,
    article_path: Path,
    model: str,
    run: int,
    settings,
) -> dict[str, Any]:
    """Score a single article. Returns a result dict."""
    body = json.loads(article_path.read_text(encoding="utf-8"))["body"]
    body = _truncate_article(body, settings.max_article_chars)
    system_msg = settings.prompt_template
    user_msg = f"Article:\n<<<ARTICLE_START>>>\n{body}\n<<<ARTICLE_END>>>"
    prompt_hash = settings.prompt_hash

    # First attempt
    try:
        raw = call_ollama(client, model, system=system_msg, prompt=user_msg, settings=settings)
    except Exception as e:
        _log_error(settings, {
            "stage": "call_ollama_failed",
            "model": model,
            "article": article_path.name,
            "run": run,
            "error": repr(e),
        })
        return _format_result(
            {}, status="fallback", model=model,
            article=article_path.name, prompt_hash=prompt_hash,
        )

    # JSON mode should give us valid JSON, but validate structure
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Model output parsed to a non-object JSON value.")
        return _format_result(
            data, status="ok", model=model,
            article=article_path.name, prompt_hash=prompt_hash,
        )
    except Exception as e:
        raw_str = raw if isinstance(raw, str) else repr(raw)
        raw_path = _save_failed_raw(
            settings,
            stage="json_parse_failed",
            model=model,
            article=article_path.name,
            run=run,
            raw=raw_str,
        )
        _log_error(settings, {
            "stage": "json_parse_failed",
            "model": model,
            "article": article_path.name,
            "run": run,
            "error": repr(e),
            "raw_len": len(raw_str),
            "raw_file": str(raw_path),
        })

    # Retry loop (should be rare with format='json')
    last_raw = raw
    for attempt in range(1, settings.parse_retries + 1):
        try:
            retry_raw = call_ollama(
                client, model,
                system=system_msg, prompt=_retry_prompt(last_raw),
                settings=settings,
            )
        except Exception as e:
            _log_error(settings, {
                "stage": "retry_call_failed",
                "model": model,
                "article": article_path.name,
                "run": run,
                "attempt": attempt,
                "error": repr(e),
            })
            continue

        try:
            data = json.loads(retry_raw)
            if not isinstance(data, dict):
                raise ValueError("Model output parsed to a non-object JSON value.")
            return _format_result(
                data, status="recovered", model=model,
                article=article_path.name, prompt_hash=prompt_hash,
            )
        except Exception as e:
            retry_raw_str = retry_raw if isinstance(retry_raw, str) else repr(retry_raw)
            raw_path = _save_failed_raw(
                settings,
                stage="retry_parse_failed",
                model=model,
                article=article_path.name,
                run=run,
                raw=retry_raw_str,
                attempt=attempt,
            )
            _log_error(settings, {
                "stage": "retry_parse_failed",
                "model": model,
                "article": article_path.name,
                "run": run,
                "attempt": attempt,
                "error": repr(e),
                "raw_len": len(retry_raw_str),
                "raw_file": str(raw_path),
            })
            last_raw = retry_raw

    return _format_result(
        {}, status="fallback", model=model,
        article=article_path.name, prompt_hash=prompt_hash,
    )


# ---------- Parallel article scoring ----------

def _score_and_save(
    client: ollama.Client,
    article_path: Path,
    out_file: Path,
    model: str,
    run: int,
    settings,
) -> str:
    """Score one article and write result to disk. Returns status string."""
    result = score_one_article(client, article_path, model, run, settings)
    out_file.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result.get("status", "ok")


def score_all(settings, sources: Iterable[str] | None = None) -> None:
    """Score every article with every model for N runs.

    Models are processed sequentially (only one model loaded at a time —
    safe for systems with no swap).  Articles within each model/run are
    scored in parallel via ThreadPoolExecutor.

    Results are written to per-source subdirectories:
      <run_dir>/<source>_results/<model>/<run>/
    """
    client = _get_client(settings)
    selected = set(sources) if sources is not None else None

    for model in settings.models:
        print(f"\n{'=' * 50}")
        print(f"MODEL: {model}")
        print(f"{'=' * 50}")

        for run in range(1, settings.runs + 1):
            todo = []
            skipped = 0
            source_counts: dict[str, int] = {}
            for src_dir in settings.all_source_webdata_dirs:
                source = src_dir.name
                if selected is not None and source not in selected:
                    continue
                files = sorted(src_dir.glob("*.json"))
                source_counts[source] = len(files)
                out_dir = (
                    settings.source_results_dir(source)
                    / settings.model_output_dirname(model)
                    / str(run)
                )
                out_dir.mkdir(parents=True, exist_ok=True)
                for p in files:
                    out_file = out_dir / f"{p.stem}.json"
                    if out_file.exists():
                        skipped += 1
                    else:
                        todo.append((source, p, out_file))

            if skipped:
                print(f"  Run {run}/{settings.runs}: {skipped} already scored, {len(todo)} to do")
            else:
                print(f"  Run {run}/{settings.runs}: {len(todo)} articles")
            if source_counts:
                by_source = ", ".join(f"{src}={n}" for src, n in sorted(source_counts.items()))
                print(f"    Sources: {by_source}")

            if not todo:
                continue

            bar = tqdm(
                total=len(todo),
                desc=f"  run {run}",
                unit="article",
                leave=True,
            )

            with ThreadPoolExecutor(max_workers=settings.max_workers) as pool:
                futures = {
                    pool.submit(
                        _score_and_save, client, p, out_file, model, run, settings
                    ): f"{source}/{p.stem}"
                    for source, p, out_file in todo
                }
                for future in as_completed(futures):
                    stem = futures[future]
                    try:
                        status = future.result()
                        bar.set_postfix_str(f"{stem} [{status}]")
                    except Exception as e:
                        bar.set_postfix_str(f"{stem} [error]")
                        _log_error(settings, {
                            "stage": "worker_exception",
                            "model": model,
                            "article": stem,
                            "run": run,
                            "error": repr(e),
                        })
                    bar.update(1)

            bar.close()
