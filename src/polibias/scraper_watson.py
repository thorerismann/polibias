"""Fetch and parse Watson FR articles into structured JSON."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
from polibias.filenames import stable_article_filename


@dataclass
class WatsonArticle:
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
    source: str = "watson"


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "fr-CH,fr;q=0.9,en;q=0.8",
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
        "article p",
        "main article p",
        "div.article p",
        "div.content p",
    ]
    parts: List[str] = []
    for sel in selectors:
        items = [el.get_text(" ", strip=True) for el in soup.select(sel)]
        items = [x for x in items if x]
        if items:
            parts = items
            break
    if not parts:
        return None
    return "\n\n".join(parts)


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


def parse_article(url: str, timeout: int = 30) -> WatsonArticle:
    soup = fetch_soup(url, timeout=timeout)
    jsonld = _extract_jsonld_article(soup)

    title = (
        _safe_strip(jsonld.get("headline"))
        or _safe_strip((soup.title.string if soup.title else None))
    )
    canonical = (
        _safe_strip(jsonld.get("mainEntityOfPage"))
        or _safe_strip((soup.select_one('link[rel="canonical"]') or {}).get("href"))
        or url
    )

    return WatsonArticle(
        title=title,
        body=_extract_body(soup),
        description=(
            _safe_strip(jsonld.get("description"))
            or _safe_strip((soup.select_one('meta[name="description"]') or {}).get("content"))
        ),
        headline=_safe_strip(jsonld.get("headline")) or title,
        author=_extract_author(jsonld),
        keywords=_extract_keywords(jsonld),
        article_section=_safe_strip(jsonld.get("articleSection")),
        in_language=_safe_strip(jsonld.get("inLanguage")) or "fr",
        canonical_url=canonical,
        publisher_name="Watson",
        date_published=_extract_date(jsonld.get("datePublished")),
        date_modified=_extract_date(jsonld.get("dateModified")),
        date_accessed=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    )


def fetch_article_links(limit: int = 20, timeout: int = 20) -> List[str]:
    """Discover Watson FR article URLs from homepage-like sections."""
    seeds = [
        "https://www.watson.ch/fr",
        "https://www.watson.ch/fr/suisse",
        "https://www.watson.ch/fr/international",
    ]
    seen: set[str] = set()
    links: List[str] = []

    for seed in seeds:
        soup = fetch_soup(seed, timeout=timeout)
        for a in soup.select("a[href]"):
            href = a.get("href", "")
            full = urljoin(seed, href)
            if "watson.ch/fr" not in full:
                continue
            if not re.search(r"/\d{7,}[-/]", full):
                continue
            full = full.split("?", 1)[0].rstrip("/")
            if full in seen:
                continue
            seen.add(full)
            links.append(full)
            if len(links) >= limit:
                return links
    return links


def _make_filename(article: dict) -> str:
    return stable_article_filename(article, "watson")


def scrape_watson(urls: List[str], out_dir: Path, timeout: int = 30) -> None:
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
            save_path = out_dir / fname
            if save_path.exists():
                print(f"  [skip] {fname}")
                continue
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"  [ok]   {fname}")
        except Exception as e:  # noqa: BLE001
            print(f"  [err]  {url}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape Watson FR articles into JSON files.")
    parser.add_argument("--limit", type=int, default=20, help="Max articles to discover.")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "webdata_watson",
        help="Directory to write JSON files into.",
    )
    parser.add_argument(
        "--urls-file",
        type=Path,
        default=None,
        help="Plain-text file with one URL per line. If omitted, discover from Watson FR sections.",
    )
    args = parser.parse_args()

    if args.urls_file:
        urls = args.urls_file.read_text().splitlines()
    else:
        print("Fetching article links from Watson FR pages...")
        urls = fetch_article_links(limit=args.limit, timeout=args.timeout)
        print(f"Found {len(urls)} article links.")

    scrape_watson(urls, args.out_dir, timeout=args.timeout)
