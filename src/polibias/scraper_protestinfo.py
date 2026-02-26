"""Fetch and parse Protestinfo articles into structured JSON."""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from polibias.scraper_utils import (
    extract_author,
    extract_jsonld_article,
    extract_keywords,
    fetch_soup,
    parse_iso_datetime,
    safe_strip,
    scrape_urls,
)


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
            return textwrap.fill(plain, width=120)
    return None


def parse_article(url: str, timeout: int = 30) -> ProtestinfoArticle:
    soup = fetch_soup(url, headers=_HEADERS, timeout=timeout)
    jsonld = extract_jsonld_article(soup)
    title = safe_strip(jsonld.get("headline")) or safe_strip(soup.title.string if soup.title else None)
    canonical = (
        safe_strip(jsonld.get("mainEntityOfPage"))
        or safe_strip((soup.select_one('link[rel="canonical"]') or {}).get("href"))
        or url
    )

    description = safe_strip(jsonld.get("description")) or safe_strip(
        (soup.select_one('meta[name="description"]') or {}).get("content")
    )
    body = _extract_body(soup) or description

    return ProtestinfoArticle(
        title=title,
        body=body,
        description=description,
        headline=safe_strip(jsonld.get("headline")) or title,
        author=extract_author(jsonld),
        keywords=extract_keywords(jsonld),
        article_section=safe_strip(jsonld.get("articleSection")),
        in_language=safe_strip(jsonld.get("inLanguage")),
        canonical_url=canonical,
        publisher_name="Protestinfo",
        date_published=parse_iso_datetime(jsonld.get("datePublished")),
        date_modified=parse_iso_datetime(jsonld.get("dateModified")),
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


def scrape_protestinfo(urls: List[str], out_dir: Path, timeout: int = 30) -> None:
    scrape_urls(urls, out_dir, "protestinfo", parse_article, timeout=timeout)
