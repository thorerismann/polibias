"""Fetch and parse RTS news articles into structured JSON."""

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
class RTSArticle:
    title: Optional[str]
    lead: Optional[str]
    body: Optional[str]
    headline: Optional[str]
    alternative_headline: Optional[str]
    description: Optional[str]
    keywords: List[str]
    article_section: Optional[str]
    in_language: Optional[str]
    canonical_url: Optional[str]
    publisher_name: Optional[str]
    date_published: Optional[str]
    date_accessed: str
    sources: List[str]
    credit: List[str]
    source: str = "rts"


# ---------- Fetch ----------

def fetch_rts_soup(url: str, timeout: int = 30) -> BeautifulSoup:
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "fr-CH,fr;q=0.9,en;q=0.8,de;q=0.7",
        },
    )
    with urlopen(req, timeout=timeout) as r:
        ctype = (r.headers.get("Content-Type") or "").lower()
        if "text/html" not in ctype:
            raise ValueError(f"Not HTML: {ctype}")
        html = r.read().decode("utf-8", errors="ignore")

    soup = BeautifulSoup(html, "html.parser")
    if not soup.html or not soup.body:
        raise ValueError("Malformed or non-document HTML")
    return soup


# ---------- JSON-LD ----------

def _pick_newsarticle(obj: Any) -> Optional[Dict[str, Any]]:
    if isinstance(obj, dict):
        if obj.get("@type") in ("NewsArticle", "Article"):
            return obj
        return None
    if isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict) and it.get("@type") in ("NewsArticle", "Article"):
                return it
    return None


def extract_jsonld_newsarticle(soup: BeautifulSoup) -> Dict[str, Any]:
    if not soup:
        return {}
    for sc in soup.select('script[type="application/ld+json"]'):
        raw = sc.string
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        article = _pick_newsarticle(data)
        if isinstance(article, dict):
            if (
                article.get("@type") in ("NewsArticle", "Article")
                or "headline" in article
                or "datePublished" in article
            ):
                return article
    return {}


# ---------- Safe helpers ----------

def _safe_text(node, selector, default=None, many=False):
    if not node:
        return default
    if many:
        els = node.select(selector)
        return [el.get_text(" ", strip=True) for el in els] if els else default
    el = node.select_one(selector)
    return el.get_text(" ", strip=True) if el else default


def _safe_strip(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s2 = s.strip()
    return s2 if s2 else None


# ---------- Field extractors ----------

def _extract_title(soup) -> Optional[str]:
    return soup.title.string.strip() if soup.title and soup.title.string else None


def _extract_lead(soup) -> Optional[str]:
    return _safe_text(soup, "div.article-part.article-lead")


def _extract_body(body) -> Optional[str]:
    if not body:
        return None
    parts: List[str] = []
    for el in body.select("p, h2, h3"):
        classes = el.get("class", []) or []
        if "sources" in classes or "credit" in classes:
            break
        if classes:
            continue
        txt = el.get_text(" ", strip=True)
        if txt:
            parts.append(txt)
    return "\n\n".join(parts) if parts else None


def _extract_sources(body) -> List[str]:
    return _safe_text(body, "p.sources", default=[], many=True)


def _extract_credits(body) -> List[str]:
    return _safe_text(body, "p.credit", default=[], many=True)


def _extract_date_published(soup) -> Optional[str]:
    time_tag = soup.find("time", datetime=True)
    if not time_tag:
        return None
    raw = time_tag["datetime"]
    try:
        return (
            datetime.fromisoformat(raw.replace("Z", "+00:00"))
            .strftime("%Y-%m-%d %H:%M:%S")
        )
    except ValueError:
        return raw


def _extract_description(soup):
    meta = soup.select_one('meta[name="dcterms.description"]')
    return meta.get("content") if meta else None


# JSON-LD field helpers

def _kw_from_jsonld(j: Dict[str, Any]) -> List[str]:
    kw = j.get("keywords")
    if isinstance(kw, list):
        return [k for k in (_safe_strip(str(x)) for x in kw) if k]
    if isinstance(kw, str):
        return [p.strip() for p in kw.split(",") if p.strip()]
    return []


def _str_field(j: Dict[str, Any], key: str) -> Optional[str]:
    v = j.get(key)
    return _safe_strip(v) if v else None


def _publisher_name(j: Dict[str, Any]) -> Optional[str]:
    pub = j.get("publisher")
    if isinstance(pub, dict):
        return _safe_strip(pub.get("name"))
    return None


# ---------- Orchestrator ----------

def parse_html(url: str, timeout: int = 30) -> RTSArticle:
    soup = fetch_rts_soup(url, timeout=timeout)
    jsonld = extract_jsonld_newsarticle(soup)
    return RTSArticle(
        title=_extract_title(soup),
        lead=_extract_lead(soup),
        body=_extract_body(soup.body),
        sources=_extract_sources(soup.body),
        credit=_extract_credits(soup.body),
        date_published=_extract_date_published(soup),
        date_accessed=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        keywords=_kw_from_jsonld(jsonld),
        publisher_name=_publisher_name(jsonld),
        in_language=_str_field(jsonld, "inLanguage"),
        article_section=_str_field(jsonld, "articleSection"),
        headline=_str_field(jsonld, "headline"),
        alternative_headline=_str_field(jsonld, "alternativeHeadline"),
        canonical_url=_str_field(jsonld, "mainEntityOfPage"),
        description=_extract_description(soup),
    )


# ---------- Persistence ----------

def _make_filename(article: dict) -> str:
    url = article.get("canonical_url") or article.get("url")
    if isinstance(url, str):
        m = re.search(r"-(\d+)\.html$", url)
        if m:
            return f"{m.group(1)}.json"
    title = article.get("title") or article.get("headline") or "article"
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", title.lower()).strip("_")
    return f"{slug[:30]}.json"


def scrape_articles(settings) -> None:
    """Fetch articles from the input URL list and save as JSON.

    Skips articles whose JSON already exists in the rts webdata subfolder.
    """
    webdata_dir = settings.source_webdata_dir("rts")
    webdata_dir.mkdir(parents=True, exist_ok=True)

    with open(settings.input_file, "r") as f:
        urls = f.readlines()

    for url in urls:
        url = url.strip()
        if not url.startswith("http"):
            continue
        try:
            data = parse_html(url, timeout=settings.scrape_timeout)
            if is_dataclass(data):
                data = asdict(data)
            fname = _make_filename(data)
            save_path = webdata_dir / fname
            if save_path.exists():
                print(f"  [skip] {fname} already exists")
                continue
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"  [ok]   {fname}")
        except Exception as e:
            print(f"  [err]  {url}: {e}")
