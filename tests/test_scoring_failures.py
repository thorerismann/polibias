from __future__ import annotations

import json
from pathlib import Path

from polibias.config import Settings
from polibias.scoring import score_one_article


class _Resp:
    def __init__(self, response: str):
        self.response = response


class _Client:
    def __init__(self, responses: list[str]):
        self._responses = responses

    def generate(self, **kwargs):  # noqa: ANN003
        if not self._responses:
            return _Resp("")
        return _Resp(self._responses.pop(0))


def _mk_settings(tmp_path: Path, *, parse_retries: int = 0) -> Settings:
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    return Settings(
        root=tmp_path,
        run_name="exp_errs",
        parse_retries=parse_retries,
        models=["m1"],
        runs=1,
    )


def _mk_article(tmp_path: Path) -> Path:
    p = tmp_path / "article.json"
    p.write_text(json.dumps({"body": "hello world"}), encoding="utf-8")
    return p


def test_parse_failure_logs_raw_output_file(tmp_path: Path) -> None:
    settings = _mk_settings(tmp_path, parse_retries=0)
    article = _mk_article(tmp_path)
    client = _Client(["not-json"])

    result = score_one_article(client, article, "m1", 1, settings)

    assert result["status"] == "fallback"

    err_log = settings.errors_dir / "errors.jsonl"
    assert err_log.exists()
    entries = [json.loads(line) for line in err_log.read_text(encoding="utf-8").splitlines()]
    parse_entry = next(e for e in entries if e["stage"] == "json_parse_failed")

    raw_file = Path(parse_entry["raw_file"])
    assert raw_file.exists()
    assert raw_file.read_text(encoding="utf-8") == "not-json"


def test_retry_recovers_after_invalid_json(tmp_path: Path) -> None:
    settings = _mk_settings(tmp_path, parse_retries=1)
    article = _mk_article(tmp_path)
    client = _Client(
        [
            "{bad-json",
            json.dumps(
                {
                    "subject_bias": -0.1,
                    "framing_bias": 0.0,
                    "treatment_bias": 0.2,
                    "guests_bias": 0.1,
                    "confidence": 0.7,
                    "comment": "recovered",
                }
            ),
        ]
    )

    result = score_one_article(client, article, "m1", 1, settings)

    assert result["status"] == "recovered"
    assert result["comment"] == "recovered"
