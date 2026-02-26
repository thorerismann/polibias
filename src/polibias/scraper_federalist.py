"""Fetch and parse The Federalist articles into structured JSON.

The Federalist (thefederalist.com) is a right-wing political commentary site.
This scraper mirrors the interface of scraper.py for RTS articles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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


# ---------- Schema ----------

@dataclass
class FederalistArticle:
    title: Optional[str]
    body: Optional[str]
    description: Optional[str]
    headline: Optional[str]
    author: Optional[str]
    keywords: List[str]
    article_section: Optional[str]
    in_language: Optional[str]
    canonical_url: Optional[str]
    publisher_name: Optional[str]
    date_published: Optional[str]
    date_modified: Optional[str]
    date_accessed: str
    word_count: Optional[int]
    source: str = "the_federalist"


# ---------- Fetch ----------

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# ---------- Field extractors ----------

def _safe_strip(s: Any) -> Optional[str]:
    return safe_strip(s, unescape=True)


def _extract_body(soup: BeautifulSoup) -> Optional[str]:
    """Extract article body text from .article-body paragraphs."""
    container = soup.select_one(".article-body")
    if not container:
        return None
    parts: List[str] = []
    for el in container.select("p, h2, h3"):
        txt = el.get_text(" ", strip=True)
        if txt:
            parts.append(txt)
    return "\n\n".join(parts) if parts else None


def _extract_author(soup: BeautifulSoup, jsonld: Dict[str, Any]) -> Optional[str]:
    # Prefer JSON-LD author name
    author = extract_author(jsonld)
    if author:
        return author
    # Fallback: HTML element
    el = soup.select_one('[class*="author"]')
    if el:
        text = el.get_text(" ", strip=True)
        # Strip common prefixes like "By:"
        text = re.sub(r"^[Bb]y\s*:?\s*", "", text).strip()
        return text if text else None
    return None


def _extract_date(soup: BeautifulSoup, jsonld: Dict[str, Any], field: str) -> Optional[str]:
    parsed = parse_iso_datetime(jsonld.get(field))
    if parsed:
        return parsed
    # Fallback: <time> tag for datePublished only
    if field == "datePublished":
        time_tag = soup.find("time", datetime=True)
        if time_tag:
            return _safe_strip(time_tag.get("datetime"))
    return None


def _extract_keywords(jsonld: Dict[str, Any]) -> List[str]:
    return extract_keywords(jsonld)


def _extract_section(jsonld: Dict[str, Any]) -> Optional[str]:
    sec = jsonld.get("articleSection")
    if isinstance(sec, list):
        return sec[0] if sec else None
    return _safe_strip(sec)


def _extract_description(soup: BeautifulSoup) -> Optional[str]:
    for sel in [
        'meta[name="description"]',
        'meta[property="og:description"]',
    ]:
        meta = soup.select_one(sel)
        if meta:
            return _safe_strip(meta.get("content"))
    return None


def _extract_canonical(soup: BeautifulSoup, jsonld: Dict[str, Any]) -> Optional[str]:
    # JSON-LD mainEntityOfPage
    mep = jsonld.get("mainEntityOfPage")
    if isinstance(mep, dict):
        return _safe_strip(mep.get("@id"))
    if isinstance(mep, str):
        return _safe_strip(mep)
    # <link rel="canonical">
    link = soup.select_one('link[rel="canonical"]')
    return _safe_strip(link.get("href")) if link else None


# ---------- Orchestrator ----------

def parse_article(url: str, timeout: int = 30) -> FederalistArticle:
    soup = fetch_soup(
        url,
        headers=_HEADERS,
        timeout=timeout,
        require_html_content_type=True,
    )
    jsonld = extract_jsonld_article(soup)

    return FederalistArticle(
        title=_safe_strip(soup.title.string) if soup.title else None,
        body=_extract_body(soup),
        description=_extract_description(soup),
        headline=_safe_strip(jsonld.get("headline")),
        author=_extract_author(soup, jsonld),
        keywords=_extract_keywords(jsonld),
        article_section=_extract_section(jsonld),
        in_language=_safe_strip(jsonld.get("inLanguage")),
        canonical_url=_extract_canonical(soup, jsonld),
        publisher_name="The Federalist",
        date_published=_extract_date(soup, jsonld, "datePublished"),
        date_modified=_extract_date(soup, jsonld, "dateModified"),
        date_accessed=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        word_count=jsonld.get("wordCount"),
    )


# ---------- Homepage link discovery ----------

def fetch_article_links(limit: int = 20, timeout: int = 15) -> List[str]:
    """Fetch recent article URLs from The Federalist homepage."""
    soup = fetch_soup("https://thefederalist.com/", timeout=timeout)
    seen: set = set()
    links: List[str] = []
    for a in soup.select("a[href]"):
        href = a["href"]
        if re.match(r"https://thefederalist\.com/\d{4}/\d{2}/\d{2}/", href):
            if href not in seen:
                seen.add(href)
                links.append(href)
                if len(links) >= limit:
                    break
    return links


# ---------- Persistence ----------

def scrape_federalist(
    urls: List[str],
    out_dir: Path,
    timeout: int = 30,
) -> None:
    """Fetch Federalist articles and save each as JSON in *out_dir*.

    Skips articles whose JSON file already exists.
    """
    scrape_urls(urls, out_dir, "the_federalist", parse_article, timeout=timeout)


# ---------- CLI entry point ----------

def main() -> None:
    """Scrape recent articles from The Federalist homepage and save to data/webdata/the_federalist/."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape The Federalist articles into JSON files."
    )
    parser.add_argument(
        "--urls-file",
        type=Path,
        default=None,
        help="Plain-text file with one URL per line. "
             "If omitted, discovers links from the homepage.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "webdata" / "the_federalist",
        help="Directory to write JSON files into.",
    )
    parser.add_argument("--limit", type=int, default=20, help="Max articles to fetch from homepage.")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    if args.urls_file:
        urls = args.urls_file.read_text().splitlines()
    else:
        print("Fetching article links from homepage…")
        urls = fetch_article_links(limit=args.limit, timeout=args.timeout)
        print(f"Found {len(urls)} article links.")

    scrape_federalist(urls, args.out_dir, timeout=args.timeout)


if __name__ == "__main__":
    main()
