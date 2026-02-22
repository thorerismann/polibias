"""Fetch and parse Protestinfo articles into structured JSON."""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
from polibias.filenames import stable_article_filename


@dataclass
class ProtestinfoArticle:
    title: Optional[str]
    body: Optional[str]
    description: Optional[str]
    headline: Optional[str]
    author: Optional[str]
    keywords: List[str]
    article_section: Optional[str]
    in_language: Optional[str]
    canonical_url: Optional[str]
    publisher_name: str
    date_published: Optional[str]
    date_modified: Optional[str]
    date_accessed: str
    source: str = "protestinfo"


_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "fr-CH,fr;q=0.9,de;q=0.8,en;q=0.7",
}


def _safe_strip(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def fetch_soup(url: str, timeout: int = 30) -> BeautifulSoup:
    req = Request(url, headers=_HEADERS)
    with urlopen(req, timeout=timeout) as r:
        html = r.read().decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    if not soup.html or not soup.body:
        raise ValueError("Malformed or non-document HTML")
    return soup


def _extract_jsonld_article(soup: BeautifulSoup) -> dict:
    for sc in soup.select('script[type="application/ld+json"]'):
        raw = sc.string
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


def _extract_body(soup: BeautifulSoup) -> Optional[str]:
    selectors = [
        "article .entry-content",
        "article .post-content",
        "article .article-content",
        "main article",
        "article",
        "main .content",
        "main",
        "#content",
    ]
    for sel in selectors:
        container = soup.select_one(sel)
        if container is None:
            continue
        parts = [el.get_text(" ", strip=True) for el in container.select("p, h2, h3, li, blockquote")]
        parts = [p for p in parts if p]
        if parts:
            return "\n\n".join(parts)

        # Fallback: plain container text if semantic tags are absent.
        plain = container.get_text(" ", strip=True)
        plain = re.sub(r"\s+", " ", plain).strip()
        if plain and len(plain) > 120:
            return "\n".join(textwrap.wrap(plain, width=120))
    return None


def _extract_keywords(node: dict) -> List[str]:
    kw = node.get("keywords")
    if isinstance(kw, list):
        return [str(k).strip() for k in kw if str(k).strip()]
    if isinstance(kw, str):
        return [p.strip() for p in kw.split(",") if p.strip()]
    return []


def _extract_author(node: dict) -> Optional[str]:
    author = node.get("author")
    if isinstance(author, dict):
        return _safe_strip(author.get("name"))
    if isinstance(author, list) and author:
        first = author[0]
        if isinstance(first, dict):
            return _safe_strip(first.get("name"))
        return _safe_strip(first)
    return _safe_strip(author)


def _extract_date(value: Any) -> Optional[str]:
    raw = _safe_strip(value)
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return raw


def parse_article(url: str, timeout: int = 30) -> ProtestinfoArticle:
    soup = fetch_soup(url, timeout=timeout)
    jsonld = _extract_jsonld_article(soup)
    title = _safe_strip(jsonld.get("headline")) or _safe_strip(soup.title.string if soup.title else None)
    canonical = (
        _safe_strip(jsonld.get("mainEntityOfPage"))
        or _safe_strip((soup.select_one('link[rel="canonical"]') or {}).get("href"))
        or url
    )

    description = _safe_strip(jsonld.get("description")) or _safe_strip(
        (soup.select_one('meta[name="description"]') or {}).get("content")
    )
    body = _extract_body(soup) or description

    return ProtestinfoArticle(
        title=title,
        body=body,
        description=description,
        headline=_safe_strip(jsonld.get("headline")) or title,
        author=_extract_author(jsonld),
        keywords=_extract_keywords(jsonld),
        article_section=_safe_strip(jsonld.get("articleSection")),
        in_language=_safe_strip(jsonld.get("inLanguage")),
        canonical_url=canonical,
        publisher_name="Protestinfo",
        date_published=_extract_date(jsonld.get("datePublished")),
        date_modified=_extract_date(jsonld.get("dateModified")),
        date_accessed=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    )


def fetch_article_links(limit: int = 20, timeout: int = 20) -> List[str]:
    seeds = [
        "https://www.protestinfo.ch/",
        "https://www.protestinfo.ch/actualites.html",
    ]
    links: List[str] = []
    seen: set[str] = set()
    for seed in seeds:
        soup = fetch_soup(seed, timeout=timeout)
        for a in soup.select("a[href]"):
            href = a.get("href", "")
            full = urljoin(seed, href).split("?", 1)[0].rstrip("/")
            if "protestinfo.ch" not in full:
                continue
            if not re.search(r"/[a-z0-9][a-z0-9-]{10,}\.html$", full):
                continue
            if full in seen:
                continue
            seen.add(full)
            links.append(full)
            if len(links) >= limit:
                return links
    return links


def _make_filename(article: dict) -> str:
    return stable_article_filename(article, "protestinfo")


def scrape_protestinfo(urls: List[str], out_dir: Path, timeout: int = 30) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for url in urls:
        url = url.strip()
        if not url.startswith("http"):
            continue
        try:
            data = parse_article(url, timeout=timeout)
            if is_dataclass(data):
                data = asdict(data)
            fname = _make_filename(data)
            path = out_dir / fname
            if path.exists():
                print(f"  [skip] {fname}")
                continue
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  [ok]   {fname}")
        except Exception as e:  # noqa: BLE001
            print(f"  [err]  {url}: {e}")
