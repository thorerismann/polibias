"""Fetch and parse Cathinfo articles into structured JSON."""

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
class CathinfoArticle:
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
    source: str = "cathinfo"


_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "fr-CH,fr;q=0.9,en;q=0.8",
}


def _extract_body(soup: BeautifulSoup) -> Optional[str]:
    # Cath.ch pages often store the real article in a dedicated div.text container.
    primary = soup.select_one("div.text")
    if primary is not None:
        raw = primary.get_text("\n", strip=True)
        raw = re.sub(r"\n{3,}", "\n\n", raw).strip()
        if "©" in raw:
            raw = raw.split("©", 1)[0].strip()
        if raw:
            return raw

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

        # Fallback: plain container text when there are no paragraph nodes.
        plain = container.get_text(" ", strip=True)
        plain = re.sub(r"\s+", " ", plain).strip()
        if plain and len(plain) > 120:
            if "©" in plain:
                plain = plain.split("©", 1)[0].strip()
            return textwrap.fill(plain, width=120)
    return None


def parse_article(url: str, timeout: int = 30) -> CathinfoArticle:
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

    return CathinfoArticle(
        title=title,
        body=body,
        description=description,
        headline=safe_strip(jsonld.get("headline")) or title,
        author=extract_author(jsonld),
        keywords=extract_keywords(jsonld),
        article_section=safe_strip(jsonld.get("articleSection")),
        in_language=safe_strip(jsonld.get("inLanguage")) or "fr",
        canonical_url=canonical,
        publisher_name="Cathinfo",
        date_published=parse_iso_datetime(jsonld.get("datePublished")),
        date_modified=parse_iso_datetime(jsonld.get("dateModified")),
        date_accessed=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    )


def fetch_article_links(limit: int = 20, timeout: int = 20) -> List[str]:
    seeds = [
        "https://www.cathinfo.ch/",
        "https://www.cathinfo.ch/category/actualite/",
    ]
    links: List[str] = []
    seen: set[str] = set()
    for seed in seeds:
        soup = fetch_soup(seed, timeout=timeout)
        for a in soup.select("a[href]"):
            href = a.get("href", "")
            full = urljoin(seed, href).split("?", 1)[0].rstrip("/")
            if "cathinfo.ch" not in full:
                continue
            if not re.search(r"/\d{4}/\d{2}/[a-z0-9-]{8,}$", full):
                continue
            if full in seen:
                continue
            seen.add(full)
            links.append(full)
            if len(links) >= limit:
                return links
    return links


def scrape_cathinfo(urls: List[str], out_dir: Path, timeout: int = 30) -> None:
    scrape_urls(urls, out_dir, "cathinfo", parse_article, timeout=timeout)
