import concurrent.futures
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any


MODEL_NAME = "codex"
RUN_IDS = (1, 2)
MAX_WORKERS = 2


def _prompt_text() -> str:
    return Path("src/polibias/prompt.md").read_text(encoding="utf-8").strip()


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


def _schema_path() -> Path:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "subject_bias",
            "framing_bias",
            "treatment_bias",
            "guests_bias",
            "confidence",
            "comment",
        ],
        "properties": {
            "subject_bias": {"type": ["number", "null"]},
            "framing_bias": {"type": ["number", "null"]},
            "treatment_bias": {"type": ["number", "null"]},
            "guests_bias": {"type": ["number", "null"]},
            "confidence": {"type": ["number", "null"]},
            "comment": {"type": ["string", "null"]},
        },
    }
    fd, tmp = tempfile.mkstemp(prefix="codex_schema_", suffix=".json")
    path = Path(tmp)
    path.write_text(json.dumps(schema), encoding="utf-8")
    return path


def _clamp(v: Any) -> float | None:
    if v is None:
        return None
    try:
        n = float(v)
    except (TypeError, ValueError):
        return None
    if n < -1.0:
        return -1.0
    if n > 1.0:
        return 1.0
    return n


def _fallback(article: str, prompt_hash: str) -> dict[str, Any]:
    return {
        "subject_bias": None,
        "framing_bias": None,
        "treatment_bias": None,
        "guests_bias": None,
        "confidence": None,
        "comment": None,
        "status": "fallback",
        "model": MODEL_NAME,
        "article": article,
        "prompt_hash": prompt_hash,
    }


def _invoke_codex(prompt: str, body: str, schema_path: Path) -> dict[str, Any]:
    full_prompt = (
        f"{prompt}\n\nArticle:\n<<<ARTICLE_START>>>\n{body}\n<<<ARTICLE_END>>>\n"
    )
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=True) as tmp:
        cmd = [
            "codex",
            "exec",
            "-",
            "--skip-git-repo-check",
            "--output-schema",
            str(schema_path),
            "-o",
            tmp.name,
            "--color",
            "never",
        ]
        proc = subprocess.run(
            cmd,
            input=full_prompt,
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "codex exec failed")
        raw = Path(tmp.name).read_text(encoding="utf-8").strip()
        if not raw:
            raise RuntimeError("codex returned empty output")
        return json.loads(raw)


def _score_file(
    source: str,
    web_file: Path,
    run_id: int,
    prompt: str,
    phash: str,
    schema_path: Path,
) -> str:
    out_dir = Path(f"data/runs/comparisons/{source}_results/{MODEL_NAME}/{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / web_file.name
    if out_path.exists():
        return f"SKIP {source}/{web_file.name} run={run_id}"

    payload = json.loads(web_file.read_text(encoding="utf-8"))
    body = payload.get("body")
    if not isinstance(body, str) or not body.strip():
        result = _fallback(web_file.name, phash)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return f"FALLBACK {source}/{web_file.name} run={run_id} (missing body)"

    try:
        obj = _invoke_codex(prompt, body, schema_path)
        result = {
            "subject_bias": _clamp(obj.get("subject_bias")),
            "framing_bias": _clamp(obj.get("framing_bias")),
            "treatment_bias": _clamp(obj.get("treatment_bias")),
            "guests_bias": _clamp(obj.get("guests_bias")),
            "confidence": _clamp(obj.get("confidence")),
            "comment": None if obj.get("comment") is None else str(obj.get("comment")),
            "status": "ok",
            "model": MODEL_NAME,
            "article": web_file.name,
            "prompt_hash": phash,
        }
    except Exception:
        result = _fallback(web_file.name, phash)

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"SAVED {source}/{web_file.name} run={run_id} status={result['status']}"


def main() -> None:
    prompt = _prompt_text()
    phash = _prompt_hash(prompt)
    schema_path = _schema_path()
    tasks: list[tuple[str, Path, int]] = []
    for source_dir in sorted(Path("data/webdata").iterdir()):
        if not source_dir.is_dir():
            continue
        source = source_dir.name
        for file_path in sorted(source_dir.glob("*.json")):
            for run_id in RUN_IDS:
                tasks.append((source, file_path, run_id))

    print(f"Prompt hash: {phash}")
    print(f"Total tasks: {len(tasks)}")
    saved = 0
    skipped = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [
            pool.submit(_score_file, s, p, r, prompt, phash, schema_path)
            for s, p, r in tasks
        ]
        for i, fut in enumerate(concurrent.futures.as_completed(futures), start=1):
            msg = fut.result()
            print(f"[{i}/{len(tasks)}] {msg}")
            if msg.startswith("SKIP"):
                skipped += 1
            else:
                saved += 1
    print(f"Done. saved={saved} skipped={skipped}")
    try:
        schema_path.unlink(missing_ok=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
