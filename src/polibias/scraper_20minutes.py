"""Fetch and parse 20 Minutes FR articles into structured JSON."""

from __future__ import annotations

import re
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
class TwentyMinutesArticle:
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
    source: str = "20minutes"


_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "fr-CH,fr;q=0.9,en;q=0.8",
}


def _extract_body(soup: BeautifulSoup) -> Optional[str]:
    selectors = [
        "article p",
        "main article p",
        '[data-testid="article-content"] p',
        "div.article-body p",
        "div.story p",
        "main p",
    ]
    for sel in selectors:
        parts = [p.get_text(" ", strip=True) for p in soup.select(sel)]
        parts = [p for p in parts if p]
        if parts:
            return "\n\n".join(parts)
    return None


def parse_article(url: str, timeout: int = 30) -> TwentyMinutesArticle:
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

    return TwentyMinutesArticle(
        title=title,
        body=body,
        description=description,
        headline=safe_strip(jsonld.get("headline")) or title,
        author=extract_author(jsonld),
        keywords=extract_keywords(jsonld),
        article_section=safe_strip(jsonld.get("articleSection")),
        in_language=safe_strip(jsonld.get("inLanguage")) or "fr",
        canonical_url=canonical,
        publisher_name="20 Minutes",
        date_published=parse_iso_datetime(jsonld.get("datePublished")),
        date_modified=parse_iso_datetime(jsonld.get("dateModified")),
        date_accessed=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    )


def fetch_article_links(limit: int = 20, timeout: int = 20) -> List[str]:
    seeds = [
        "https://www.20min.ch/fr",
        "https://www.20min.ch/fr/news",
    ]
    links: List[str] = []
    seen: set[str] = set()
    for seed in seeds:
        soup = fetch_soup(seed, headers=_HEADERS, timeout=timeout)
        for a in soup.select("a[href]"):
            href = a.get("href", "")
            full = urljoin(seed, href).split("?", 1)[0].rstrip("/")
            if "20min.ch/fr/story/" not in full:
                continue
            if re.search(r"/(video|podcast|live)(/|$)", full):
                continue
            if not re.search(r"/fr/story/[a-z0-9-]+-\d{6,}$", full):
                continue
            if full in seen:
                continue
            seen.add(full)
            links.append(full)
            if len(links) >= limit:
                return links
    return links


def scrape_20minutes(urls: List[str], out_dir: Path, timeout: int = 30) -> None:
    scrape_urls(urls, out_dir, "20minutes", parse_article, timeout=timeout)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape 20 Minutes FR articles into JSON files.")
    parser.add_argument("--limit", type=int, default=20, help="Max articles to discover.")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "webdata" / "20minutes",
        help="Directory to write JSON files into.",
    )
    parser.add_argument(
        "--urls-file",
        type=Path,
        default=None,
        help="Plain-text file with one URL per line. If omitted, discover from 20 Minutes FR.",
    )
    args = parser.parse_args()

    if args.urls_file:
        urls = args.urls_file.read_text().splitlines()
    else:
        print("Fetching article links from 20 Minutes FR...")
        urls = fetch_article_links(limit=args.limit, timeout=args.timeout)
        print(f"Found {len(urls)} article links.")

    scrape_20minutes(urls, args.out_dir, timeout=args.timeout)
