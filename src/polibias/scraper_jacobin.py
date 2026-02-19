"""Fetch and parse Jacobin articles into structured JSON.

Jacobin (jacobin.com) is a left-wing socialist political commentary magazine.
This scraper mirrors the interface of scraper_federalist.py.

HTML class conventions used by Jacobin (custom CSS, no JSON-LD):
  Body:       section.po-cn__section  (main article paragraphs)
  Lead:       section.po-cn__intro    (intro/dek paragraphs)
  Author:     dd.po-hr-cn__author     (article-level, not sidebar)
  Date:       time.po-hr-fl__date     (text format: MM.DD.YYYY)
  Categories: ul.po-hr-fl__taxonomies li
  Title:      meta[property="og:title"]
  Canonical:  meta[property="og:url"]   (no <link rel="canonical">)
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup


# ---------- Schema ----------

@dataclass
class JacobinArticle:
    title: Optional[str]
    lead: Optional[str]
    body: Optional[str]
    description: Optional[str]
    author: Optional[str]
    keywords: List[str]
    article_section: Optional[str]
    in_language: str
    canonical_url: Optional[str]
    publisher_name: str
    date_published: Optional[str]
    date_accessed: str


# ---------- Fetch ----------

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_soup(url: str, timeout: int = 30) -> BeautifulSoup:
    req = Request(url, headers=_HEADERS)
    with urlopen(req, timeout=timeout) as r:
        ctype = (r.headers.get("Content-Type") or "").lower()
        if "text/html" not in ctype:
            raise ValueError(f"Not HTML: {ctype}")
        html = r.read().decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    if not soup.html or not soup.body:
        raise ValueError("Malformed or non-document HTML")
    return soup


# ---------- Field extractors ----------

def _safe_strip(s: Any) -> Optional[str]:
    if not s:
        return None
    import html as _html
    s2 = _html.unescape(str(s)).strip()
    return s2 if s2 else None


def _extract_title(soup: BeautifulSoup) -> Optional[str]:
    for sel, attr in [
        ('meta[property="og:title"]', "content"),
        ('meta[name="twitter:title"]', "content"),
    ]:
        el = soup.select_one(sel)
        if el:
            return _safe_strip(el.get(attr))
    return _safe_strip(soup.title.string) if soup.title else None


def _extract_description(soup: BeautifulSoup) -> Optional[str]:
    for sel in ['meta[name="description"]', 'meta[property="og:description"]']:
        el = soup.select_one(sel)
        if el:
            return _safe_strip(el.get("content"))
    return None


def _extract_lead(soup: BeautifulSoup) -> Optional[str]:
    """Article intro/dek section (section.po-cn__intro)."""
    container = soup.select_one("section.po-cn__intro, section.po-wp__intro")
    if not container:
        return None
    parts = [p.get_text(" ", strip=True) for p in container.select("p")]
    return "\n\n".join(p for p in parts if p) or None


def _extract_body(soup: BeautifulSoup) -> Optional[str]:
    """Main article body (section.po-cn__section).

    Falls back to the intro section for short articles that have no separate
    body container (all content is in po-cn__intro in those cases).
    """
    container = soup.select_one("section.po-cn__section, section.po-wp__section")
    if not container:
        # Short-form articles keep everything in the intro section
        container = soup.select_one("section.po-cn__intro, section.po-wp__intro")
    if not container:
        return None
    parts: List[str] = []
    for el in container.select("p, h2, h3, blockquote"):
        txt = el.get_text(" ", strip=True)
        if txt:
            parts.append(txt)
    return "\n\n".join(parts) if parts else None


def _extract_author(soup: BeautifulSoup) -> Optional[str]:
    """Article-level author (dd.po-hr-cn__author), not sidebar authors."""
    el = soup.select_one("dd.po-hr-cn__author")
    if el:
        return _safe_strip(el.get_text(" ", strip=True))
    # Broader fallback: the first author anchor in the article header
    el = soup.select_one("dl.po-hr-cn__authors a")
    if el:
        return _safe_strip(el.get_text(strip=True))
    return None


def _parse_jacobin_date(text: str) -> Optional[str]:
    """Parse Jacobin's MM.DD.YYYY date format into YYYY-MM-DD 00:00:00."""
    text = text.strip()
    m = re.match(r"(\d{2})\.(\d{2})\.(\d{4})", text)
    if m:
        month, day, year = m.groups()
        try:
            dt = datetime(int(year), int(month), int(day))
            return dt.strftime("%Y-%m-%d 00:00:00")
        except ValueError:
            pass
    return text if text else None


def _extract_date_published(soup: BeautifulSoup) -> Optional[str]:
    el = soup.select_one("time.po-hr-fl__date")
    if el:
        return _parse_jacobin_date(el.get_text(strip=True))
    # Fallback: any <time> with datetime attribute
    time_tag = soup.find("time", datetime=True)
    if time_tag:
        raw = time_tag["datetime"]
        try:
            return (
                datetime.fromisoformat(raw.replace("Z", "+00:00"))
                .strftime("%Y-%m-%d %H:%M:%S")
            )
        except ValueError:
            return raw
    return None


def _extract_keywords(soup: BeautifulSoup) -> List[str]:
    """Article categories from ul.po-hr-fl__taxonomies (first occurrence = article tags)."""
    ul = soup.select_one("ul.po-hr-fl__taxonomies")
    if ul:
        return [li.get_text(strip=True) for li in ul.select("li") if li.get_text(strip=True)]
    return []


def _extract_section(soup: BeautifulSoup) -> Optional[str]:
    """First category tag as the primary section."""
    kws = _extract_keywords(soup)
    return kws[0] if kws else None


def _extract_canonical(soup: BeautifulSoup) -> Optional[str]:
    # No <link rel="canonical"> on Jacobin; use og:url
    el = soup.select_one('meta[property="og:url"]')
    if el:
        return _safe_strip(el.get("content"))
    return None


# ---------- Orchestrator ----------

def parse_article(url: str, timeout: int = 30) -> JacobinArticle:
    soup = fetch_soup(url, timeout=timeout)
    return JacobinArticle(
        title=_extract_title(soup),
        lead=_extract_lead(soup),
        body=_extract_body(soup),
        description=_extract_description(soup),
        author=_extract_author(soup),
        keywords=_extract_keywords(soup),
        article_section=_extract_section(soup),
        in_language="en",
        canonical_url=_extract_canonical(soup),
        publisher_name="Jacobin",
        date_published=_extract_date_published(soup),
        date_accessed=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    )


# ---------- Homepage link discovery ----------

_ARTICLE_RE = re.compile(r"^/\d{4}/\d{2}/[^/]+/?$")
_SKIP_FRAGMENTS = {"subscribe", "account", "podcast", "video", "store", "donate", "about", "contact"}


def fetch_article_links(limit: int = 20, timeout: int = 15) -> List[str]:
    """Discover recent article URLs from the Jacobin homepage."""
    soup = fetch_soup("https://jacobin.com", timeout=timeout)
    seen: set = set()
    links: List[str] = []
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        if not _ARTICLE_RE.match(href):
            continue
        if any(skip in href for skip in _SKIP_FRAGMENTS):
            continue
        full = "https://jacobin.com" + href.rstrip("/")
        if full not in seen:
            seen.add(full)
            links.append(full)
            if len(links) >= limit:
                break
    return links


# ---------- Persistence ----------

def _make_filename(article: dict) -> str:
    url = article.get("canonical_url") or ""
    m = re.search(r"jacobin\.com/(\d{4}/\d{2}/[^/?#]+)", url)
    if m:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", m.group(1)).strip("_")
        return f"{slug[:60]}.json"
    title = article.get("title") or "article"
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", title.lower()).strip("_")
    return f"{slug[:40]}.json"


def scrape_jacobin(
    urls: List[str],
    out_dir: Path,
    timeout: int = 30,
) -> None:
    """Fetch Jacobin articles and save each as JSON in *out_dir*.

    Skips articles whose JSON file already exists.
    """
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
        except Exception as e:
            print(f"  [err]  {url}: {e}")


# ---------- CLI entry point ----------

def main() -> None:
    """Scrape recent articles from Jacobin and save to data/webdata_jacobin/."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape Jacobin articles into JSON files."
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
        default=Path(__file__).resolve().parents[2] / "data" / "webdata_jacobin",
        help="Directory to write JSON files into.",
    )
    parser.add_argument("--limit", type=int, default=20, help="Max articles from homepage.")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    if args.urls_file:
        urls = args.urls_file.read_text().splitlines()
    else:
        print("Fetching article links from homepage…")
        urls = fetch_article_links(limit=args.limit, timeout=args.timeout)
        print(f"Found {len(urls)} article links.")

    scrape_jacobin(urls, args.out_dir, timeout=args.timeout)


if __name__ == "__main__":
    main()
