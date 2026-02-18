"""CLI entry point for polibias.

Usage::

    python -m polibias              # run full pipeline
    python -m polibias scrape       # only scrape articles
    python -m polibias score        # only run model scoring
    python -m polibias analyse      # only build CSVs from results
    python -m polibias check        # verify expected output files
    python -m polibias validate     # pre-flight checks (Ollama, models, data)
    python -m polibias stats        # compute statistical analysis
    python -m polibias export       # generate HTML report, LaTeX, summaries
    python -m polibias viz          # generate HTML report and open it
    python -m polibias upload       # upload run results to GCS
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from polibias.config import Settings, load_settings


def _run_validate(settings: Settings) -> None:
    from polibias.validation import validate

    ok = validate(settings)
    if not ok:
        sys.exit(1)


def _run_scrape(settings: Settings) -> None:
    from polibias.scraper import scrape_articles

    print("\n[1/3] Scraping articles ...")
    if any(settings.webdata_dir.glob("*.json")):
        n = len(list(settings.webdata_dir.glob("*.json")))
        print(f"  Found {n} existing article(s) — only missing ones will be fetched.")
    scrape_articles(settings)
    print("  Scraping complete.")


def _run_score(settings: Settings) -> None:
    from polibias.scoring import score_all

    print("\n[2/3] Scoring articles with Ollama models ...")
    if not any(settings.webdata_dir.glob("*.json")):
        print("  ERROR: No articles found. Run 'scrape' first.")
        sys.exit(1)
    score_all(settings)
    print("  Scoring complete.")


def _run_analyse(settings: Settings) -> None:
    from polibias.analysis import build_bias_frame, build_webdata_frame

    print("\n[3/3] Building analysis CSVs ...")
    settings.run_dir.mkdir(parents=True, exist_ok=True)
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


def _run_stats(settings: Settings) -> None:
    from polibias.stats import run_stats

    print("\nComputing statistics ...")
    run_stats(settings)
    print("  Statistics complete.")


def _run_export(settings: Settings) -> None:
    from polibias.export import run_export

    print("\nGenerating exports ...")
    run_export(settings)
    print("  Export complete.")


def _run_viz(settings: Settings) -> None:
    import webbrowser

    from polibias.export import run_export

    print("\nGenerating HTML report ...")
    run_export(settings)

    report_path = settings.report_html_path
    if report_path.exists():
        url = report_path.as_uri()
        print(f"\n  Report ready: {report_path}")
        print(f"  Opening in browser ...")
        webbrowser.open(url)
    else:
        print(f"  ERROR: Report was not generated. Check for errors above.")
        sys.exit(1)


def _run_check(settings: Settings) -> None:
    print("\nPipeline save-path check")
    print(f"  run dir: {settings.run_dir}")
    web_count = len(list(settings.webdata_dir.glob("*.json")))
    result_count = len(list(settings.results_dir.rglob("*.json")))
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
        print("  WARNING: Legacy CSV(s) found outside run-dir outputs:")
        for p in legacy:
            print(f"    - {p}")
        print("  Canonical outputs for this run are:")
        print(f"    - {settings.bias_csv_path}")
        print(f"    - {settings.web_csv_path}")


def _run_upload(settings: Settings, bucket: str) -> None:
    try:
        from google.cloud import storage  # type: ignore[import-untyped]
    except ImportError:
        print("  ERROR: google-cloud-storage not installed.")
        print("  Install with: pip install 'polibias[cloud]'")
        sys.exit(1)

    run_dir = settings.run_dir
    if not run_dir.exists():
        print(f"  ERROR: Run directory not found: {run_dir}")
        sys.exit(1)

    prefix = settings.run_name
    client = storage.Client()
    gcs_bucket = client.bucket(bucket)

    uploaded = 0
    for path in sorted(run_dir.rglob("*")):
        if path.is_dir():
            continue
        blob_name = f"{prefix}/{path.relative_to(run_dir)}"
        blob = gcs_bucket.blob(blob_name)
        blob.upload_from_filename(str(path))
        uploaded += 1

    print(f"  Uploaded {uploaded} file(s) to gs://{bucket}/{prefix}/")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="polibias",
        description="Political bias scoring of news articles using local LLMs.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=[
            "all", "scrape", "score", "analyse",
            "check", "validate", "stats", "export", "viz", "upload",
        ],
        help="Pipeline step to run (default: all)",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Run folder name under data/runs/ (default: run_results).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a TOML config file (keys map to Settings fields).",
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help="GCS bucket name for the 'upload' command.",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config) if args.config else None
    overrides: dict[str, Any] = {}
    if args.run_dir is not None:
        overrides["run_name"] = args.run_dir
    settings = load_settings(config_path, **overrides)

    if args.command == "validate":
        _run_validate(settings)
        return

    if args.command == "stats":
        _run_stats(settings)
        return

    if args.command == "export":
        _run_export(settings)
        return

    if args.command == "viz":
        _run_viz(settings)
        return

    if args.command == "upload":
        bucket = args.bucket or os.environ.get("GCS_BUCKET")
        if not bucket:
            print("  ERROR: --bucket or GCS_BUCKET env var is required.")
            sys.exit(1)
        print(f"\nUploading results to gs://{bucket}/ ...")
        _run_upload(settings, bucket)
        return

    pipeline_steps = {
        "scrape": _run_scrape,
        "score": _run_score,
        "analyse": _run_analyse,
    }

    if args.command == "all":
        print("polibias — running full pipeline")
        print(f"Run outputs: {settings.run_dir}")
        _run_validate(settings)
        for step in pipeline_steps.values():
            step(settings)
    elif args.command == "check":
        _run_check(settings)
    else:
        pipeline_steps[args.command](settings)

    print("\nDone.")


if __name__ == "__main__":
    main()
