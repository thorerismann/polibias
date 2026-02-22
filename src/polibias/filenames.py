"""Stable JSON filename helpers for scraped articles."""

from __future__ import annotations

import hashlib
import re
from typing import Any


_DIGITS_RE = re.compile(r"(\d{6,})")


def _extract_numeric_id(url: str) -> str | None:
    matches = _DIGITS_RE.findall(url)
    if not matches:
        return None
    # Prefer the longest digit run; tie-break by right-most match.
    matches.sort(key=len)
    return matches[-1]


def stable_article_filename(article: dict[str, Any], source: str) -> str:
    """Return stable ``<source>_<number>.json`` filename for any article dict.

    Number extraction order:
    1) longest 6+ digit run from canonical URL
    2) deterministic 12-digit numeric hash of canonical URL (or fallback text)
    """
    canonical = str(article.get("canonical_url") or article.get("url") or "").strip()
    numeric = _extract_numeric_id(canonical)
    if numeric is None:
        seed = canonical or str(article.get("headline") or article.get("title") or "article")
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()
        numeric = f"{int(digest[:15], 16) % 10**12:012d}"
    return f"{source}_{numeric}.json"

