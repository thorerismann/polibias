from __future__ import annotations

import json
from pathlib import Path

from polibias import __main__ as cli
from polibias.config import Settings
from polibias.export import run_export, run_export_cross_source


def _mk_settings(tmp_path: Path) -> Settings:
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    return Settings(root=tmp_path, run_name="exp_src", models=["m1"], runs=1)


def _write_score(settings: Settings, source: str, article_id: str, value: float) -> None:
    out_dir = settings.source_results_dir(source) / "m1" / "1"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{article_id}.json").write_text(
        json.dumps(
            {
                "subject_bias": value,
                "framing_bias": value,
                "treatment_bias": value,
                "guests_bias": value,
                "confidence": 0.8,
                "comment": f"{source}-{article_id}",
                "status": "ok",
            }
        ),
        encoding="utf-8",
    )


def test_run_export_source_and_cross_source(tmp_path: Path) -> None:
    settings = _mk_settings(tmp_path)
    _write_score(settings, "rts", "a1", 0.2)
    _write_score(settings, "jacobin", "a2", -0.3)

    run_export(
        settings,
        source="rts",
        output_filename="report_rts.html",
        include_tables=False,
        write_artifacts=False,
    )
    rts_report = settings.run_dir / "report_rts.html"
    assert rts_report.exists()
    assert "rts Bias Report" in rts_report.read_text(encoding="utf-8")
    assert not settings.article_summaries_csv_path.exists()

    run_export_cross_source(
        settings,
        output_filename="report_all.html",
        source_reports={"rts": "report_rts.html", "jacobin": "report_jacobin.html"},
    )
    all_report = settings.run_dir / "report_all.html"
    html = all_report.read_text(encoding="utf-8")
    assert all_report.exists()
    assert "Cross-source report" in html
    assert "Article cross-tab (mean overall bias)" in html
    assert "report_rts.html" in html


def test_cli_routes_source_commands(monkeypatch, tmp_path: Path) -> None:
    settings = _mk_settings(tmp_path)
    seen: dict[str, object] = {}

    monkeypatch.setattr(cli, "load_settings", lambda config_path, **overrides: settings)
    monkeypatch.setattr(cli, "_run_score_source", lambda s, source: seen.setdefault("score", source))
    monkeypatch.setattr(
        cli,
        "_run_scrape_federalist",
        lambda s, limit, urls_file: seen.setdefault("scrape", ("the_federalist", limit, urls_file)),
    )
    monkeypatch.setattr(
        cli,
        "_run_scrape_watson",
        lambda s, limit, urls_file: seen.setdefault("scrape", ("watson", limit, urls_file)),
    )
    monkeypatch.setattr(
        cli,
        "_run_scrape_protestinfo",
        lambda s, limit, urls_file: seen.setdefault("scrape", ("protestinfo", limit, urls_file)),
    )
    monkeypatch.setattr(
        cli,
        "_run_viz_source",
        lambda s, source, output_name: seen.setdefault("viz", (source, output_name)),
    )

    cli.main(["score-federalist"])
    assert seen["score"] == "the_federalist"

    cli.main(["viz-fed"])
    assert seen["viz"] == ("the_federalist", "report_fed.html")

    seen.clear()
    cli.main(["score", "--source", "the_federalist"])
    assert seen["score"] == "the_federalist"

    seen.clear()
    cli.main(["scrape", "--source", "the_federalist", "--limit", "11"])
    assert seen["scrape"] == ("the_federalist", 11, None)

    seen.clear()
    cli.main(["viz", "--source", "the_federalist"])
    assert seen["viz"] == ("the_federalist", "report_fed.html")

    seen.clear()
    cli.main(["scrape", "--source", "watson", "--limit", "9"])
    assert seen["scrape"] == ("watson", 9, None)

    seen.clear()
    cli.main(["score", "--source", "lib_inst"])
    assert seen["score"] == "lib_inst"

    seen.clear()
    cli.main(["viz", "--source", "lib_inst"])
    assert seen["viz"] == ("lib_inst", "report_lib_inst.html")

    seen.clear()
    cli.main(["scrape", "--source", "protestinfo", "--limit", "7"])
    assert seen["scrape"] == ("protestinfo", 7, None)

    seen.clear()
    cli.main(["score", "--source", "cathinfo"])
    assert seen["score"] == "cathinfo"

    seen.clear()
    cli.main(["viz", "--source", "cathinfo"])
    assert seen["viz"] == ("cathinfo", "report_cathinfo.html")


def test_cli_passes_runs_override(monkeypatch, tmp_path: Path) -> None:
    settings = _mk_settings(tmp_path)
    seen: dict[str, object] = {}

    def _load_settings(config_path, **overrides):  # noqa: ANN001
        seen["overrides"] = overrides
        return settings

    monkeypatch.setattr(cli, "load_settings", _load_settings)
    monkeypatch.setattr(cli, "_run_score_source", lambda s, source: None)

    cli.main(["score", "--source", "jacobin", "--runs", "2"])
    assert seen["overrides"]["runs"] == 2


def test_cli_help_prints_quick_help(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda *args, **kwargs: None)
    cli.main(["help"])
    out = capsys.readouterr().out
    assert "polibias quick help" in out
    assert "There is no --source-dir flag." in out
    assert "polibias viz --run-dir temp0.4" in out


def test_cli_default_writes_run_log(monkeypatch, tmp_path: Path, capsys) -> None:
    settings = _mk_settings(tmp_path)
    monkeypatch.setattr(cli, "load_settings", lambda config_path, **overrides: settings)
    monkeypatch.setattr(cli, "_run_score_source", lambda s, source: None)

    cli.main(["score", "--source", "jacobin"])

    out = capsys.readouterr().out
    assert "Detailed logs:" in out
    log_path = settings.run_dir / "polibias.log"
    assert log_path.exists()
    assert "command: score --source jacobin" in log_path.read_text(encoding="utf-8")


def test_cli_verbose_streams_output(monkeypatch, tmp_path: Path, capsys) -> None:
    settings = _mk_settings(tmp_path)
    monkeypatch.setattr(cli, "load_settings", lambda config_path, **overrides: settings)
    monkeypatch.setattr(cli, "_run_score_source", lambda s, source: print(f"score={source}"))

    cli.main(["score", "--source", "jacobin", "--verbose"])

    out = capsys.readouterr().out
    assert "score=jacobin" in out
    assert "Detailed logs:" not in out
