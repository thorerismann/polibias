#!/usr/bin/env python3
"""Generate source link input files from live site discovery.

Usage:
  python scripts_seed_links.py --source watson --limit 10
  python scripts_seed_links.py --source lib_inst --limit 10
  python scripts_seed_links.py --source all --limit 10
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _get_fetcher(source: str):
    if source == "jacobin":
        from polibias.scraper_jacobin import fetch_article_links
        return fetch_article_links
    if source == "the_federalist":
        from polibias.scraper_federalist import fetch_article_links
        return fetch_article_links
    if source == "watson":
        from polibias.scraper_watson import fetch_article_links
        return fetch_article_links
    if source == "lib_inst":
        from polibias.scraper_lib_inst import fetch_article_links
        return fetch_article_links
    if source == "protestinfo":
        from polibias.scraper_protestinfo import fetch_article_links
        return fetch_article_links
    if source == "cathinfo":
        from polibias.scraper_cathinfo import fetch_article_links
        return fetch_article_links
    if source == "srf":
        from polibias.scraper_srf import fetch_article_links
        return fetch_article_links
    if source == "20minutes":
        from polibias.scraper_20minutes import fetch_article_links
        return fetch_article_links
    raise ValueError(f"Unsupported source: {source}")


def _write_links(source: str, links: list[str], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source}_links.txt"
    out_path.write_text("\n".join(links) + ("\n" if links else ""), encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate source input link files")
    parser.add_argument(
        "--source",
        default="all",
        choices=[
            "all",
            "jacobin",
            "the_federalist",
            "watson",
            "lib_inst",
            "protestinfo",
            "cathinfo",
            "srf",
            "20minutes",
        ],
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    sources = (
        ["jacobin", "the_federalist", "watson", "lib_inst", "protestinfo", "cathinfo", "srf", "20minutes"]
        if args.source == "all"
        else [args.source]
    )

    input_dir = args.root / "data" / "input_files"

    for source in sources:
        fetcher = _get_fetcher(source)
        print(f"\nFetching {source} links (limit={args.limit}) ...")
        links = fetcher(limit=args.limit, timeout=args.timeout)
        links = [u.strip() for u in links if u.strip()]
        dedup = []
        seen = set()
        for u in links:
            if u not in seen:
                seen.add(u)
                dedup.append(u)
        out = _write_links(source, dedup[: args.limit], input_dir)
        print(f"  wrote {len(dedup[: args.limit])} links -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
