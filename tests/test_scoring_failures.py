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


def _mk_article_with_body(tmp_path: Path, body) -> Path:  # noqa: ANN001
    p = tmp_path / "article_body.json"
    p.write_text(json.dumps({"body": body}), encoding="utf-8")
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

    assert result["status"] == "ok"
    assert result["comment"] == "recovered"


def test_missing_body_returns_fallback_without_raising(tmp_path: Path) -> None:
    settings = _mk_settings(tmp_path, parse_retries=0)
    article = _mk_article_with_body(tmp_path, None)
    client = _Client([json.dumps({"subject_bias": 0.1})])

    result = score_one_article(client, article, "m1", 1, settings)

    assert result["status"] == "fallback"
    err_log = settings.errors_dir / "errors.jsonl"
    assert err_log.exists()
    entries = [json.loads(line) for line in err_log.read_text(encoding="utf-8").splitlines()]
    assert any(e["stage"] == "missing_article_body" for e in entries)


def test_bias_scores_are_clamped_to_unit_interval(tmp_path: Path) -> None:
    settings = _mk_settings(tmp_path, parse_retries=0)
    article = _mk_article(tmp_path)
    client = _Client(
        [
            json.dumps(
                {
                    "subject_bias": 4.2,
                    "framing_bias": -9.9,
                    "treatment_bias": "0.4",
                    "guests_bias": "-0.25",
                    "confidence": "0.8",
                    "comment": "ok",
                }
            )
        ]
    )

    result = score_one_article(client, article, "m1", 1, settings)

    assert result["status"] == "ok"
    assert result["subject_bias"] == 1.0
    assert result["framing_bias"] == -1.0
    assert result["treatment_bias"] == 0.4
    assert result["guests_bias"] == -0.25


def test_malformed_comment_tail_is_recovered_locally(tmp_path: Path) -> None:
    settings = _mk_settings(tmp_path, parse_retries=0)
    article = _mk_article(tmp_path)
    broken = (
        '{'
        '"subject_bias": -0.1,'
        '"framing_bias": 0.3,'
        '"treatment_bias": 0.2,'
        '"guests_bias": 0.0,'
        '"confidence": 0.8,'
        '"comment": "first comment",'
        '"first comment": "trailing garbage that should be ignored'
    )
    client = _Client([broken])

    result = score_one_article(client, article, "m1", 1, settings)

    assert result["status"] == "recovered"
    assert result["comment"] == "first comment"
    err_log = settings.errors_dir / "errors.jsonl"
    entries = [json.loads(line) for line in err_log.read_text(encoding="utf-8").splitlines()]
    assert any(e["stage"] == "json_recovered" for e in entries)


def test_missing_required_keys_triggers_retry(tmp_path: Path) -> None:
    settings = _mk_settings(tmp_path, parse_retries=1)
    article = _mk_article(tmp_path)
    client = _Client(
        [
            json.dumps({"subject_bias": 0.1}),
            json.dumps(
                {
                    "subject_bias": 0.1,
                    "framing_bias": 0.2,
                    "treatment_bias": 0.3,
                    "guests_bias": 0.4,
                    "confidence": 0.9,
                    "comment": "after retry",
                }
            ),
        ]
    )

    result = score_one_article(client, article, "m1", 1, settings)

    assert result["status"] == "ok"
    assert result["comment"] == "after retry"
