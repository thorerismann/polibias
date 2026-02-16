from __future__ import annotations

import json
from pathlib import Path

from polibias.analysis import build_bias_frame
from polibias.config import Settings


def _mk_settings(tmp_path: Path, *, run_name: str) -> Settings:
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    return Settings(root=tmp_path, run_name=run_name, models=["m1"], runs=1)


def test_settings_resolve_run_scoped_paths(tmp_path: Path) -> None:
    settings = _mk_settings(tmp_path, run_name="exp_temp_02")

    assert settings.run_dir == tmp_path / "data" / "runs" / "exp_temp_02"
    assert settings.results_dir == settings.run_dir / "results"
    assert settings.errors_dir == settings.run_dir / "errors"
    assert settings.bias_csv_path == settings.run_dir / "bias_data.csv"
    assert settings.web_csv_path == settings.run_dir / "web_data.csv"
    assert settings.report_html_path == settings.run_dir / "report.html"


def test_bias_frame_reads_only_selected_run_dir(tmp_path: Path) -> None:
    s1 = _mk_settings(tmp_path, run_name="exp_one")
    s2 = _mk_settings(tmp_path, run_name="exp_two")

    model_dir = s1.results_dir / "m1" / "1"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "a1.json").write_text(
        json.dumps(
            {
                "subject_bias": 0.1,
                "framing_bias": 0.2,
                "treatment_bias": 0.3,
                "guests_bias": 0.4,
                "confidence": 0.9,
                "comment": "ok",
                "status": "ok",
            }
        ),
        encoding="utf-8",
    )

    df1 = build_bias_frame(s1)
    df2 = build_bias_frame(s2)

    assert len(df1) == 1
    assert df1.iloc[0]["article_id"] == "a1"
    assert df2.empty
