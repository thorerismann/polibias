"""Post-processing: aggregate model results and web-data into DataFrames."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd

_WS_RE = re.compile(r"\s+")
_TAG_RE = re.compile(r"<[^>]+>")


# ---------- Bias results ----------

def build_bias_frame(settings) -> pd.DataFrame:
    """Collect all per-article score JSONs into a single DataFrame.

    Looks in both the legacy results/ dir and the per-source dirs
    (rts_results/, jacobin_results/, the_federalist_results/, etc.).
    Adds a 'source' column derived from the directory name.
    """
    rows = []

    def _append_rows_from_model_dir(source: str, model: str, run: int, model_dir: Path) -> None:
        for p in model_dir.glob("*.json"):
            data = json.loads(p.read_text(encoding="utf-8"))
            rows.append({
                "source": source,
                "model": model,
                "article_id": p.stem,
                "subject_bias": data.get("subject_bias"),
                "framing_bias": data.get("framing_bias"),
                "treatment_bias": data.get("treatment_bias"),
                "guests_bias": data.get("guests_bias"),
                "confidence": data.get("confidence"),
                "comment": data.get("comment"),
                "status": data.get("status", "ok"),
                "run": run,
            })

    # Collect all candidate (source, model_dir) pairs to scan
    def _iter_model_dirs(base_dir, source):
        for run in range(1, settings.runs + 1):
            for model in settings.models:
                canonical = base_dir / settings.model_output_dirname(model) / str(run)
                legacy = base_dir / model.replace(":", "_") / str(run)
                for d in {canonical, legacy}:
                    if d.is_dir():
                        yield source, run, model, d

    # Legacy flat results/ dir (pre-source-split runs)
    for source, run, model, model_dir in _iter_model_dirs(settings.results_dir, "rts"):
        _append_rows_from_model_dir(source, model, run, model_dir)

    # Per-source results dirs (<source>_results/)
    for src_dir in sorted(settings.run_dir.glob("*_results")):
        if not src_dir.is_dir():
            continue
        source = src_dir.name.removesuffix("_results")
        for source, run, model, model_dir in _iter_model_dirs(src_dir, source):
            _append_rows_from_model_dir(source, model, run, model_dir)

    df = pd.DataFrame(rows)
    if not df.empty:
        bias_cols = ["subject_bias", "framing_bias", "treatment_bias", "guests_bias"]
        numeric_cols = [*bias_cols, "confidence"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["overall_bias"] = df[bias_cols].mean(axis=1)
    return df


# ---------- Web-data cleaning ----------

def _clean_text(x: Any) -> Optional[str]:
    if x is None:
        return None
    if not isinstance(x, str):
        x = str(x)
    x = _TAG_RE.sub(" ", x)
    x = _WS_RE.sub(" ", x).strip()
    return x or None


def _ensure_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [v for v in (_clean_text(v) for v in x) if v]
    cv = _clean_text(x)
    return [cv] if cv else []


def _word_count(text: Optional[str]) -> int:
    return len(text.split()) if text else 0


def build_webdata_frame(settings) -> pd.DataFrame:
    """Build a cleaned DataFrame from the scraped web-data JSONs."""
    records = []
    all_json = []
    for src_dir in settings.all_source_webdata_dirs:
        all_json.extend(sorted(src_dir.glob("*.json")))
    for p in all_json:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [warn] {p}: {e}")
            continue
        data["article_id"] = p.stem
        records.append(data)

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df

    # Clean text columns
    text_cols = [
        "title", "headline", "alternative_headline",
        "lead", "body", "description",
        "canonical_url", "publisher_name", "article_section", "in_language",
    ]
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].map(_clean_text)

    for c in ["keywords", "sources", "credit"]:
        if c in df.columns:
            df[c] = df[c].map(_ensure_list)

    for c in ["date_published", "date_accessed"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=False)

    # Word counts
    for fld in ["title", "lead", "body"]:
        if fld in df.columns:
            df[f"{fld}_words"] = df[fld].map(_word_count).astype(int)

    df["text_words_total"] = sum(
        df.get(f"{f}_words", 0) for f in ["title", "lead", "body"]
    )

    return df
