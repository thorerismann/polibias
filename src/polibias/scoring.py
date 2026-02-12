"""Send articles to Ollama models and collect bias scores."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

_LEADING_ZERO_NUM = re.compile(r"(:\s*)(-?)00(?=[\d.])")
_REQUIRED_KEYS = [
    "subject_bias",
    "framing_bias",
    "treatment_bias",
    "guests_bias",
    "confidence",
    "comment",
]


def _fix_leading_zeros(raw: str) -> str:
    return _LEADING_ZERO_NUM.sub(r"\1\20", raw)


def call_ollama(model: str, prompt: str, settings) -> str:
    r = requests.post(
        settings.ollama_url,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": settings.ollama_options,
        },
        timeout=settings.timeout,
    )
    r.raise_for_status()
    return r.json().get("response", "")


def _strip_markdown_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("\n", 1)[0]
    return text.strip()


def _blank_result() -> dict[str, Any]:
    return {k: None for k in _REQUIRED_KEYS}


def _format_result(
    data: dict[str, Any],
    *,
    status: str,
    model: str,
    article: str,
) -> dict[str, Any]:
    out = _blank_result()
    for k in _REQUIRED_KEYS:
        if k in data:
            out[k] = data.get(k)
    out["status"] = status
    out["model"] = model
    out["article"] = article
    return out


def _log_error(settings, payload: dict[str, Any]) -> None:
    settings.errors_dir.mkdir(parents=True, exist_ok=True)
    log_path = settings.errors_dir / "errors.jsonl"
    payload = dict(payload)
    payload["ts"] = datetime.now(timezone.utc).isoformat()
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload))
        f.write("\n")


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


def parse_json_from_model(text: str) -> dict[str, Any]:
    raw = _strip_markdown_json(text)

    # Normal parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e1:
        err1 = e1

    # Leading-zero fix
    try:
        return json.loads(_fix_leading_zeros(raw))
    except json.JSONDecodeError as e2:
        err2 = e2

    # Truncation repair (last resort)
    if "{" not in raw:
        raise ValueError("Model output doesn't contain a JSON object.") from err2

    repaired = raw.strip()
    if repaired.count('"') % 2 == 1:
        repaired += '"'
    if not repaired.endswith("}"):
        repaired += "\n}"

    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e3:
        msg = (
            "Failed to parse model output as JSON.\n"
            f"normal: {err1}\nnumber-fix: {err2}\nrepair: {e3}"
        )
        raise ValueError(msg) from e3


def score_one_article(
    article_path: Path,
    model: str,
    run: int,
    settings,
) -> dict[str, Any]:
    body = json.loads(article_path.read_text(encoding="utf-8"))["body"]
    prompt = settings.prompt_template.replace("{{ARTICLE_TEXT}}", body)

    try:
        raw = call_ollama(model, prompt, settings)
    except Exception as e:
        _log_error(
            settings,
            {
                "stage": "call_ollama_failed",
                "model": model,
                "article": article_path.name,
                "run": run,
                "error": repr(e),
            },
        )
        return _format_result(
            {},
            status="fallback",
            model=model,
            article=article_path.name,
        )

    try:
        data = parse_json_from_model(raw)
        if not isinstance(data, dict):
            raise ValueError("Model output parsed to a non-object JSON value.")
        return _format_result(
            data,
            status="ok",
            model=model,
            article=article_path.name,
        )
    except Exception as e:
        _log_error(
            settings,
            {
                "stage": "json_parse_failed",
                "model": model,
                "article": article_path.name,
                "run": run,
                "error": repr(e),
                "raw_len": len(raw),
            },
        )

    last_raw = raw
    for attempt in range(1, settings.parse_retries + 1):
        try:
            retry_raw = call_ollama(model, _retry_prompt(last_raw), settings)
        except Exception as e:
            _log_error(
                settings,
                {
                    "stage": "retry_call_failed",
                    "model": model,
                    "article": article_path.name,
                    "run": run,
                    "attempt": attempt,
                    "error": repr(e),
                },
            )
            continue

        try:
            data = parse_json_from_model(retry_raw)
            if not isinstance(data, dict):
                raise ValueError("Model output parsed to a non-object JSON value.")
            return _format_result(
                data,
                status="recovered",
                model=model,
                article=article_path.name,
            )
        except Exception as e:
            _log_error(
                settings,
                {
                    "stage": "retry_parse_failed",
                    "model": model,
                    "article": article_path.name,
                    "run": run,
                    "attempt": attempt,
                    "error": repr(e),
                    "raw_len": len(retry_raw),
                },
            )
            last_raw = retry_raw

    return _format_result(
        {},
        status="fallback",
        model=model,
        article=article_path.name,
    )


def score_all(settings) -> None:
    """Score every article with every model for N runs.

    Skips individual article/model/run combinations whose output
    JSON already exists on disk.
    """
    results_dir = settings.results_dir
    for model in settings.models:
        print(f"\n{'='*50}")
        print(f"MODEL: {model}")
        print(f"{'='*50}")
        for run in range(1, settings.runs + 1):
            print(f"\n--- Run {run}/{settings.runs} ---")
            out_dir = results_dir / model.replace(":", "_") / str(run)
            out_dir.mkdir(parents=True, exist_ok=True)
            for p in sorted(settings.webdata_dir.glob("*.json")):
                out_file = out_dir / f"{p.stem}.json"
                if out_file.exists():
                    print(f"  [skip] {p.stem}")
                    continue
                print(f"  [score] {p.stem} ...", end=" ", flush=True)
                result = score_one_article(p, model, run, settings)
                out_file.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f"[{result.get('status', 'ok')}]")
