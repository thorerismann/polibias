"""CLI entry point for polibias.

Grouped command usage::

    python -m polibias run                        # run full pipeline
    python -m polibias scrape --source rts
    python -m polibias scrape --source the_federalist --limit 20
    python -m polibias score --source all
    python -m polibias score --source jacobin
    python -m polibias analyze
    python -m polibias validate
    python -m polibias stats
    python -m polibias export
    python -m polibias bambi analyze
    python -m polibias bambi viz
    python -m polibias viz                        # main report.html
    python -m polibias viz --source rts
    python -m polibias viz --source all           # cross-source report_all.html
    python -m polibias upload --bucket my-bucket

Legacy single-token commands are still accepted for backward compatibility.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from polibias.config import Settings, load_settings


SOURCE_CHOICES = ["rts", "the_federalist", "jacobin", "all"]
SOURCE_ALIAS = {
    "fed": "the_federalist",
    "federalist": "the_federalist",
}


def _normalize_source(source: str) -> str:
    return SOURCE_ALIAS.get(source, source)


def _translate_legacy_argv(argv: list[str]) -> list[str]:
    """Map legacy flat commands to grouped subcommands."""
    if not argv:
        return ["run"]

    cmd = argv[0]
    rest = argv[1:]

    mapping: dict[str, list[str]] = {
        "all": ["run"],
        "analyse": ["analyze"],
        "scrape-federalist": ["scrape", "--source", "the_federalist"],
        "scrape-jacobin": ["scrape", "--source", "jacobin"],
        "score-rts": ["score", "--source", "rts"],
        "score-federalist": ["score", "--source", "the_federalist"],
        "score-jacobin": ["score", "--source", "jacobin"],
        "viz-rts": ["viz", "--source", "rts"],
        "viz-fed": ["viz", "--source", "the_federalist"],
        "viz-jacobin": ["viz", "--source", "jacobin"],
        "viz-all": ["viz", "--source", "all"],
        "bambi": ["bambi", "analyze"],
        "bambi-analyse": ["bambi", "analyze"],
        "bambi-viz": ["bambi", "viz"],
    }

    if cmd in mapping:
        return mapping[cmd] + rest
    return argv


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
        print("  Opening in browser ...")
        webbrowser.open(url)
    else:
        print("  ERROR: Report was not generated. Check for errors above.")
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polibias",
        description="Political bias scoring of news articles using local LLMs.",
    )

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--run-dir",
        default=None,
        help="Run folder name under data/runs/ (default: run_results).",
    )
    common.add_argument(
        "--config",
        default=None,
        help="Path to a TOML config file (keys map to Settings fields).",
    )

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("run", aliases=["all"], parents=[common], help="Run full pipeline")

    p_scrape = subparsers.add_parser("scrape", parents=[common], help="Scrape article sources")
    p_scrape.add_argument(
        "--source",
        default="rts",
        choices=SOURCE_CHOICES,
        help="Source to scrape (default: rts)",
    )
    p_scrape.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max articles to fetch for homepage-backed scrapers.",
    )
    p_scrape.add_argument(
        "--urls-file",
        type=Path,
        default=None,
        help="Plain-text file with one URL per line (federalist/jacobin scrape).",
    )

    p_score = subparsers.add_parser("score", parents=[common], help="Run model scoring")
    p_score.add_argument(
        "--source",
        default="all",
        choices=SOURCE_CHOICES,
        help="Source to score (default: all)",
    )

    subparsers.add_parser(
        "analyze",
        aliases=["analyse"],
        parents=[common],
        help="Build analysis CSVs",
    )
    subparsers.add_parser("check", parents=[common], help="Check expected outputs")
    subparsers.add_parser("validate", parents=[common], help="Run pre-flight validation")
    subparsers.add_parser("stats", parents=[common], help="Compute statistics")
    subparsers.add_parser("export", parents=[common], help="Generate report artifacts")

    p_viz = subparsers.add_parser("viz", parents=[common], help="Generate/open HTML reports")
    p_viz.add_argument(
        "--source",
        default="default",
        choices=["default", *SOURCE_CHOICES],
        help="default=report.html, all=report_all.html, otherwise source-specific report",
    )

    p_upload = subparsers.add_parser("upload", parents=[common], help="Upload run outputs to GCS")
    p_upload.add_argument(
        "--bucket",
        default=None,
        help="GCS bucket name for upload (or use GCS_BUCKET env var).",
    )

    p_bambi = subparsers.add_parser("bambi", parents=[common], help="Bayesian audit tools")
    p_bambi_sub = p_bambi.add_subparsers(dest="bambi_command")
    p_bambi_sub.add_parser("viz", help="Build Bayesian HTML report")
    p_bambi_sub.add_parser("analyze", aliases=["analyse"], help="Run Bayesian fit + holdout")
    p_bambi.set_defaults(bambi_command="analyze")
    p_bambi.add_argument("--bayes-draws", type=int, default=1500, help="Bambi posterior draws.")
    p_bambi.add_argument("--bayes-tune", type=int, default=1500, help="Bambi warmup/tune steps.")
    p_bambi.add_argument("--bayes-chains", type=int, default=4, help="Bambi chains.")
    p_bambi.add_argument("--bayes-cores", type=int, default=2, help="Bambi parallel cores.")
    p_bambi.add_argument(
        "--bayes-target-accept",
        type=float,
        default=0.9,
        help="Bambi target_accept for NUTS.",
    )
    p_bambi.add_argument("--bayes-seed", type=int, default=42, help="Bambi random seed.")
    p_bambi.add_argument(
        "--bayes-collapse-runs",
        action="store_true",
        help="Collapse repeated runs to model/article averages before score model.",
    )
    p_bambi.add_argument(
        "--bayes-complete-articles-only",
        action="store_true",
        help="Only include articles where every model has at least one successful score.",
    )
    p_bambi.add_argument(
        "--bayes-no-imputation",
        action="store_true",
        help="Disable predictor imputation; failure model drops confidence/length predictors.",
    )
    p_bambi.add_argument(
        "--bayes-test-fraction",
        type=float,
        default=0.5,
        help="Holdout test fraction for Bayesian prediction evaluation.",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    raw = list(argv) if argv is not None else sys.argv[1:]
    normalized = _translate_legacy_argv(raw)

    parser = _build_parser()
    args = parser.parse_args(normalized)

    if args.command is None:
        args = parser.parse_args(["run"])

    config_path = Path(args.config) if args.config else None
    overrides: dict[str, Any] = {}
    if args.run_dir is not None:
        overrides["run_name"] = args.run_dir
    settings = load_settings(config_path, **overrides)

    if args.command == "run":
        print("polibias — running full pipeline")
        print(f"Run outputs: {settings.run_dir}")
        _run_validate(settings)
        _run_scrape(settings)
        _run_score(settings)
        _run_analyse(settings)
        print("\nDone.")
        return

    if args.command == "scrape":
        source = _normalize_source(args.source)
        if source == "rts":
            _run_scrape(settings)
        elif source == "the_federalist":
            _run_scrape_federalist(settings, limit=args.limit, urls_file=args.urls_file)
        elif source == "jacobin":
            _run_scrape_jacobin(settings, limit=args.limit, urls_file=args.urls_file)
        else:
            _run_scrape(settings)
            _run_scrape_federalist(settings, limit=args.limit, urls_file=args.urls_file)
            _run_scrape_jacobin(settings, limit=args.limit, urls_file=args.urls_file)
        print("\nDone.")
        return

    if args.command == "score":
        source = _normalize_source(args.source)
        if source == "all":
            _run_score(settings)
        else:
            _run_score_source(settings, source)
        print("\nDone.")
        return

    if args.command in {"analyze", "analyse"}:
        _run_analyse(settings)
        print("\nDone.")
        return

    if args.command == "check":
        _run_check(settings)
        print("\nDone.")
        return

    if args.command == "validate":
        _run_validate(settings)
        print("\nDone.")
        return

    if args.command == "stats":
        _run_stats(settings)
        print("\nDone.")
        return

    if args.command == "export":
        _run_export(settings)
        print("\nDone.")
        return

    if args.command == "viz":
        source = _normalize_source(args.source)
        if source == "default":
            _run_viz(settings)
        elif source == "all":
            _run_viz_all(settings)
        elif source == "rts":
            _run_viz_source(settings, "rts", output_name="report_rts.html")
        elif source == "the_federalist":
            _run_viz_source(settings, "the_federalist", output_name="report_fed.html")
        elif source == "jacobin":
            _run_viz_source(settings, "jacobin", output_name="report_jacobin.html")
        print("\nDone.")
        return

    if args.command == "upload":
        bucket = args.bucket or os.environ.get("GCS_BUCKET")
        if not bucket:
            print("  ERROR: --bucket or GCS_BUCKET env var is required.")
            sys.exit(1)
        print(f"\nUploading results to gs://{bucket}/ ...")
        _run_upload(settings, bucket)
        print("\nDone.")
        return

    if args.command == "bambi":
        if args.bambi_command in {"analyze", "analyse", None}:
            _run_bambi_analyse(settings, args)
        elif args.bambi_command == "viz":
            _run_bambi_viz(settings)
        print("\nDone.")
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
