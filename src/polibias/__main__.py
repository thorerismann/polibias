"""CLI entry point for polibias.

Usage::

    python -m polibias              # run full pipeline
    python -m polibias scrape       # only scrape articles
    python -m polibias score        # only run model scoring
    python -m polibias analyse      # only build CSVs from results
    python -m polibias check        # verify expected output files
    python -m polibias viz          # launch Streamlit dashboard
"""

from __future__ import annotations

import argparse
import sys


def _run_scrape(settings):
    from polibias.scraper import scrape_articles

    print("\n[1/3] Scraping articles ...")
    if any(settings.webdata_dir.glob("*.json")):
        n = len(list(settings.webdata_dir.glob("*.json")))
        print(f"  Found {n} existing article(s) — only missing ones will be fetched.")
    scrape_articles(settings)
    print("  Scraping complete.")


def _run_score(settings):
    from polibias.scoring import score_all

    print("\n[2/3] Scoring articles with Ollama models ...")
    if not any(settings.webdata_dir.glob("*.json")):
        print("  ERROR: No articles found. Run 'scrape' first.")
        sys.exit(1)
    score_all(settings)
    print("  Scoring complete.")


def _run_analyse(settings):
    from polibias.analysis import build_bias_frame, build_webdata_frame

    print("\n[3/3] Building analysis CSVs ...")
    bias_csv = settings.bias_csv_path
    web_csv = settings.web_csv_path

    bias_df = build_bias_frame(settings)
    if not bias_df.empty:
        bias_df.to_csv(bias_csv, index=False)
        print(f"  Wrote {bias_csv}  ({len(bias_df)} rows)")
    else:
        print("  WARNING: No bias results found. Run 'score' first.")

    web_df = build_webdata_frame(settings)
    if not web_df.empty:
        web_df.to_csv(web_csv, index=False)
        print(f"  Wrote {web_csv}  ({len(web_df)} rows)")

    print("  Analysis complete.")


def _run_check(settings):
    print("\nPipeline save-path check")
    web_count = len(list(settings.webdata_dir.glob("*.json")))
    result_count = len(list(settings.results_dir.glob("*/*/*.json")))
    print(f"  webdata JSONs: {web_count}")
    print(f"  result JSONs: {result_count}")
    print(
        "  aggregate CSVs: "
        f"bias_data.csv={'yes' if settings.bias_csv_path.exists() else 'no'}, "
        f"web_data.csv={'yes' if settings.web_csv_path.exists() else 'no'}"
    )

    legacy = []
    if settings.legacy_app_bias_csv_path.exists():
        legacy.append(str(settings.legacy_app_bias_csv_path))
    if settings.legacy_app_web_csv_path.exists():
        legacy.append(str(settings.legacy_app_web_csv_path))
    if settings.legacy_results_bias_csv_path.exists():
        legacy.append(str(settings.legacy_results_bias_csv_path))
    if legacy:
        print("  WARNING: Legacy CSV(s) found outside canonical data outputs:")
        for p in legacy:
            print(f"    - {p}")
        print("  Canonical outputs are only under data/:")
        print(f"    - {settings.bias_csv_path}")
        print(f"    - {settings.web_csv_path}")


def _run_viz(settings):
    import subprocess

    dashboard = str(settings.root / "src" / "polibias" / "dashboard.py")
    print("\nLaunching Streamlit dashboard ...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard], check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="polibias",
        description="Political bias scoring of news articles using local LLMs.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=["all", "scrape", "score", "analyse", "check", "viz"],
        help="Pipeline step to run (default: all)",
    )
    args = parser.parse_args(argv)

    from polibias.config import Settings

    settings = Settings()

    if args.command == "viz":
        _run_viz(settings)
        return

    steps = {
        "scrape": _run_scrape,
        "score": _run_score,
        "analyse": _run_analyse,
    }

    if args.command == "all":
        print("polibias — running full pipeline")
        for step in steps.values():
            step(settings)
    elif args.command == "check":
        _run_check(settings)
    else:
        steps[args.command](settings)

    print("\nDone.")


if __name__ == "__main__":
    main()
