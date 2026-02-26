"""Shared helpers for source-specific scrapers."""

from __future__ import annotations

import html
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Optional
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

from polibias.filenames import stable_article_filename


def safe_strip(value: Any, *, unescape: bool = False) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    if unescape:
        text = html.unescape(text)
    text = text.strip()
    return text or None


def parse_iso_datetime(value: Any) -> Optional[str]:
    raw = safe_strip(value)
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return raw


def fetch_soup(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: int = 30,
    require_html_content_type: bool = False,
) -> BeautifulSoup:
    req = Request(url, headers=dict(headers or {}))
    with urlopen(req, timeout=timeout) as response:
        if require_html_content_type:
            ctype = (response.headers.get("Content-Type") or "").lower()
            if "text/html" not in ctype:
                raise ValueError(f"Not HTML: {ctype}")
        html_text = response.read().decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html_text, "html.parser")
    if not soup.html or not soup.body:
        raise ValueError("Malformed or non-document HTML")
    return soup


def extract_jsonld_article(soup: BeautifulSoup) -> dict[str, Any]:
    for script in soup.select('script[type="application/ld+json"]'):
        raw = script.string
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        nodes = data if isinstance(data, list) else data.get("@graph", [data])
        if not isinstance(nodes, list):
            nodes = [nodes]
        for node in nodes:
            if isinstance(node, dict) and node.get("@type") in {"NewsArticle", "Article"}:
                return node
    return {}


def extract_keywords(node: dict[str, Any]) -> list[str]:
    keywords = node.get("keywords")
    if isinstance(keywords, list):
        return [str(k).strip() for k in keywords if str(k).strip()]
    if isinstance(keywords, str):
        return [part.strip() for part in keywords.split(",") if part.strip()]
    return []


def extract_author(node: dict[str, Any]) -> Optional[str]:
    author = node.get("author")
    if isinstance(author, dict):
        return safe_strip(author.get("name"))
    if isinstance(author, list) and author:
        first = author[0]
        if isinstance(first, dict):
            return safe_strip(first.get("name"))
        return safe_strip(first)
    return safe_strip(author)


def scrape_urls(
    urls: list[str],
    out_dir: Path,
    source: str,
    parse_article: Callable[[str, int], Any],
    *,
    timeout: int = 30,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for url in urls:
        url = url.strip()
        if not url.startswith("http"):
            continue
        try:
            data = parse_article(url, timeout)
            if is_dataclass(data):
                data = asdict(data)
            filename = stable_article_filename(data, source)
            out_path = out_dir / filename
            if out_path.exists():
                print(f"  [skip] {filename}")
                continue
            out_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  [ok]   {filename}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [err]  {url}: {exc}")
