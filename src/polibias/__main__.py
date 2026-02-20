"""CLI entry point for polibias.

Usage::

    python -m polibias              # run full pipeline
    python -m polibias scrape       # only scrape articles
    python -m polibias score        # run model scoring for all sources
    python -m polibias score-rts    # score RTS only
    python -m polibias score-federalist  # score Federalist only
    python -m polibias score-jacobin  # score Jacobin only
    python -m polibias analyse      # only build CSVs from results
    python -m polibias check        # verify expected output files
    python -m polibias validate     # pre-flight checks (Ollama, models, data)
    python -m polibias stats        # compute statistical analysis
    python -m polibias export       # generate HTML report, LaTeX, summaries
    python -m polibias bambi-analyse  # Bayesian fit + holdout prediction analysis
    python -m polibias bambi-viz      # build HTML report from Bayesian outputs
    python -m polibias viz          # generate HTML report and open it
    python -m polibias viz-rts      # generate RTS-only HTML report
    python -m polibias viz-fed      # generate Federalist-only HTML report
    python -m polibias viz-jacobin  # generate Jacobin-only HTML report
    python -m polibias viz-all      # generate cross-source HTML report
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

    rts_dir = settings.source_webdata_dir("rts")
    print("\n[1/3] Scraping RTS articles ...")
    if any(rts_dir.glob("*.json")):
        n = len(list(rts_dir.glob("*.json")))
        print(f"  Found {n} existing article(s) — only missing ones will be fetched.")
    scrape_articles(settings)
    print("  Scraping complete.")


def _run_scrape_federalist(settings: Settings, limit: int, urls_file: Path | None) -> None:
    from polibias.scraper_federalist import fetch_article_links, scrape_federalist

    out_dir = settings.source_webdata_dir("the_federalist")
    if urls_file:
        urls = Path(urls_file).read_text().splitlines()
        print(f"\nScraping {len(urls)} Federalist URLs from {urls_file} → {out_dir}")
    else:
        print(f"\nFetching Federalist article links (limit={limit}) ...")
        urls = fetch_article_links(limit=limit, timeout=settings.scrape_timeout)
        print(f"  Found {len(urls)} article links.")
    scrape_federalist(urls, out_dir, timeout=settings.scrape_timeout)
    print("  Federalist scraping complete.")


def _run_scrape_jacobin(settings: Settings, limit: int, urls_file: Path | None) -> None:
    from polibias.scraper_jacobin import fetch_article_links, scrape_jacobin

    out_dir = settings.source_webdata_dir("jacobin")
    if urls_file:
        urls = Path(urls_file).read_text().splitlines()
        print(f"\nScraping {len(urls)} Jacobin URLs from {urls_file} → {out_dir}")
    else:
        print(f"\nFetching Jacobin article links (limit={limit}) ...")
        urls = fetch_article_links(limit=limit, timeout=settings.scrape_timeout)
        print(f"  Found {len(urls)} article links.")
    scrape_jacobin(urls, out_dir, timeout=settings.scrape_timeout)
    print("  Jacobin scraping complete.")


def _run_score(settings: Settings) -> None:
    from polibias.scoring import score_all

    print("\n[2/3] Scoring articles with Ollama models ...")
    if not settings.all_source_webdata_dirs:
        print("  ERROR: No articles found. Run 'scrape' first.")
        sys.exit(1)
    score_all(settings)
    print("  Scoring complete.")


def _run_score_source(settings: Settings, source: str) -> None:
    from polibias.scoring import score_all

    src_dir = settings.source_webdata_dir(source)
    if not src_dir.exists():
        print(f"  ERROR: No scraped articles found for source '{source}' in {src_dir}.")
        print("  Run the corresponding scrape command first.")
        sys.exit(1)

    print(f"\nScoring source '{source}' with Ollama models ...")
    score_all(settings, sources=[source])
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


def _run_bambi_analyse(settings: Settings, args: argparse.Namespace) -> None:
    from polibias.bambi_audit import BambiAuditOptions, run_bambi_analyse

    print("\nRunning Bayesian scoring-behaviour audit ...")
    opts = BambiAuditOptions(
        draws=args.bayes_draws,
        tune=args.bayes_tune,
        chains=args.bayes_chains,
        cores=args.bayes_cores,
        target_accept=args.bayes_target_accept,
        random_seed=args.bayes_seed,
        collapse_runs=args.bayes_collapse_runs,
        complete_articles_only=args.bayes_complete_articles_only,
        no_imputation=args.bayes_no_imputation,
        test_fraction=args.bayes_test_fraction,
    )
    run_bambi_analyse(settings, opts)
    print("  Bayesian audit complete.")


def _run_bambi_viz(settings: Settings) -> None:
    from polibias.bambi_audit import run_bambi_viz

    print("\nGenerating Bayesian HTML report ...")
    run_bambi_viz(settings)
    print("  Bayesian report complete.")


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


def _run_viz_source(settings: Settings, source: str, *, output_name: str) -> None:
    import webbrowser

    from polibias.export import run_export

    print(f"\nGenerating HTML report for source '{source}' ...")
    run_export(
        settings,
        source=source,
        output_filename=output_name,
        include_tables=False,
        write_artifacts=False,
    )

    report_path = settings.run_dir / output_name
    if report_path.exists():
        print(f"\n  Report ready: {report_path}")
        print("  Opening in browser ...")
        webbrowser.open(report_path.as_uri())
    else:
        print("  ERROR: Report was not generated. Check for errors above.")
        sys.exit(1)


def _run_viz_all(settings: Settings) -> None:
    import webbrowser

    from polibias.export import run_export, run_export_cross_source

    targets = [
        ("rts", "report_rts.html"),
        ("the_federalist", "report_fed.html"),
        ("jacobin", "report_jacobin.html"),
    ]
    generated: dict[str, str] = {}
    for source, output_name in targets:
        if settings.source_webdata_dir(source).exists():
            run_export(
                settings,
                source=source,
                output_filename=output_name,
                include_tables=False,
                write_artifacts=False,
            )
            generated[source] = output_name

    if not generated:
        print("  ERROR: No source webdata directories found. Run scrape first.")
        sys.exit(1)

    print("\nGenerating cross-source report ...")
    run_export_cross_source(settings, output_filename="report_all.html", source_reports=generated)

    report_path = settings.run_dir / "report_all.html"
    if report_path.exists():
        print(f"\n  Cross-source report ready: {report_path}")
        print("  Opening in browser ...")
        webbrowser.open(report_path.as_uri())
    else:
        print("  ERROR: Cross-source report was not generated. Check for errors above.")
        sys.exit(1)


def _run_check(settings: Settings) -> None:
    print("\nPipeline save-path check")
    print(f"  run dir: {settings.run_dir}")
    web_count = sum(len(list(d.glob("*.json"))) for d in settings.all_source_webdata_dirs)
    result_count = len(list(settings.results_dir.rglob("*.json")))
    for src_results in settings.run_dir.glob("*_results"):
        result_count += len(list(src_results.rglob("*.json")))
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
            "all", "scrape", "scrape-federalist", "scrape-jacobin",
            "score", "score-rts", "score-federalist", "score-jacobin", "analyse",
            "check", "validate", "stats", "export", "bambi", "bambi-analyse", "bambi-viz",
            "viz", "viz-rts", "viz-fed", "viz-jacobin", "viz-all", "upload",
        ],
        help="Pipeline step to run (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max articles to fetch from homepage (scrape-federalist / scrape-jacobin).",
    )
    parser.add_argument(
        "--urls-file",
        type=Path,
        default=None,
        help="Plain-text file with one URL per line (scrape-federalist / scrape-jacobin).",
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
    parser.add_argument("--bayes-draws", type=int, default=1500, help="Bambi posterior draws.")
    parser.add_argument("--bayes-tune", type=int, default=1500, help="Bambi warmup/tune steps.")
    parser.add_argument("--bayes-chains", type=int, default=4, help="Bambi chains.")
    parser.add_argument("--bayes-cores", type=int, default=2, help="Bambi parallel cores.")
    parser.add_argument(
        "--bayes-target-accept",
        type=float,
        default=0.9,
        help="Bambi target_accept for NUTS.",
    )
    parser.add_argument("--bayes-seed", type=int, default=42, help="Bambi random seed.")
    parser.add_argument(
        "--bayes-collapse-runs",
        action="store_true",
        help="Collapse repeated runs to model/article averages before score model.",
    )
    parser.add_argument(
        "--bayes-complete-articles-only",
        action="store_true",
        help="Only include articles where every model has at least one successful score.",
    )
    parser.add_argument(
        "--bayes-no-imputation",
        action="store_true",
        help="Disable predictor imputation; failure model drops confidence/length predictors.",
    )
    parser.add_argument(
        "--bayes-test-fraction",
        type=float,
        default=0.5,
        help="Holdout test fraction for Bayesian prediction evaluation.",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config) if args.config else None
    overrides: dict[str, Any] = {}
    if args.run_dir is not None:
        overrides["run_name"] = args.run_dir
    settings = load_settings(config_path, **overrides)

    if args.command == "scrape-federalist":
        _run_scrape_federalist(settings, limit=args.limit, urls_file=args.urls_file)
        return

    if args.command == "scrape-jacobin":
        _run_scrape_jacobin(settings, limit=args.limit, urls_file=args.urls_file)
        return

    if args.command == "validate":
        _run_validate(settings)
        return

    if args.command == "stats":
        _run_stats(settings)
        return

    if args.command == "export":
        _run_export(settings)
        return

    if args.command in {"bambi", "bambi-analyse"}:
        _run_bambi_analyse(settings, args)
        return

    if args.command == "bambi-viz":
        _run_bambi_viz(settings)
        return

    if args.command == "viz":
        _run_viz(settings)
        return

    if args.command == "viz-rts":
        _run_viz_source(settings, "rts", output_name="report_rts.html")
        return

    if args.command == "viz-fed":
        _run_viz_source(settings, "the_federalist", output_name="report_fed.html")
        return

    if args.command == "viz-jacobin":
        _run_viz_source(settings, "jacobin", output_name="report_jacobin.html")
        return

    if args.command == "viz-all":
        _run_viz_all(settings)
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
    elif args.command == "score-rts":
        _run_score_source(settings, "rts")
    elif args.command == "score-federalist":
        _run_score_source(settings, "the_federalist")
    elif args.command == "score-jacobin":
        _run_score_source(settings, "jacobin")
    else:
        pipeline_steps[args.command](settings)

    print("\nDone.")


if __name__ == "__main__":
    main()
