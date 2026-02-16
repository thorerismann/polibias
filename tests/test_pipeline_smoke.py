from __future__ import annotations

import json
from pathlib import Path

from polibias.__main__ import _run_analyse
from polibias.config import Settings


def test_analyse_writes_csvs_to_run_dir(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    settings = Settings(root=tmp_path, run_name="exp_smoke", models=["m1"], runs=1)

    # Minimal score result for analysis stage
    result_dir = settings.results_dir / "m1" / "1"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "a1.json").write_text(
        json.dumps(
            {
                "subject_bias": 0.2,
                "framing_bias": 0.1,
                "treatment_bias": 0.0,
                "guests_bias": -0.1,
                "confidence": 0.9,
                "comment": "ok",
                "status": "ok",
            }
        ),
        encoding="utf-8",
    )

    # Minimal webdata file
    settings.webdata_dir.mkdir(parents=True, exist_ok=True)
    (settings.webdata_dir / "a1.json").write_text(
        json.dumps(
            {
                "title": "T",
                "lead": "L",
                "body": "B",
                "date_accessed": "2026-02-16 00:00:00",
                "keywords": [],
                "sources": [],
                "credit": [],
            }
        ),
        encoding="utf-8",
    )

    _run_analyse(settings)

    assert settings.bias_csv_path.exists()
    assert settings.web_csv_path.exists()
    assert settings.bias_csv_path.parent == settings.run_dir
    assert settings.web_csv_path.parent == settings.run_dir
