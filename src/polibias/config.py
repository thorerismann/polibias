"""Centralised configuration for the polibias pipeline.

All pipeline functions receive a ``Settings`` instance.
Configuration values are accessed via attributes or properties::

    settings.webdata_dir
    settings.models
    settings.ollama_options
    settings.prompt_template
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

ModelOptions = Dict[str, Any]


def _default_root() -> Path:
    """Return the repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class Settings:
    root: Path = field(default_factory=_default_root)

    # ---- runtime params ----
    runs: int = 4
    timeout: int = 200
    parse_retries: int = 2
    max_workers: int = 4
    max_article_chars: int = 6000
    ollama_host: str = "http://127.0.0.1:11434"

    models: List[str] = field(default_factory=lambda: [
        "llama3.2:latest",
        "gemma2:latest",
        "phi3:mini",
        "qwen2.5:3b-instruct",
        "gemma3:4b",
    ])

    ollama_options: ModelOptions = field(default_factory=lambda: {
        "temperature": 0.8,
        "num_predict": 250,
        "num_ctx": 2048,
    })

    # ---- resolved paths ----
    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def input_file(self) -> Path:
        return self.data_dir / "input_files" / "some_rts_links.csv"

    @property
    def webdata_dir(self) -> Path:
        return self.data_dir / "webdata"

    @property
    def results_dir(self) -> Path:
        return self.data_dir / "results"

    @property
    def errors_dir(self) -> Path:
        return self.data_dir / "errors"

    @property
    def stats_csv_path(self) -> Path:
        return self.data_dir / "stats_report.csv"

    @property
    def report_html_path(self) -> Path:
        return self.data_dir / "report.html"

    @property
    def bias_csv_path(self) -> Path:
        return self.data_dir / "bias_data.csv"

    @property
    def web_csv_path(self) -> Path:
        return self.data_dir / "web_data.csv"

    @property
    def legacy_app_bias_csv_path(self) -> Path:
        return self.root / "app" / "bias_data.csv"

    @property
    def legacy_app_web_csv_path(self) -> Path:
        return self.root / "app" / "web_data.csv"

    @property
    def legacy_results_bias_csv_path(self) -> Path:
        return self.results_dir / "bias_data.csv"

    @property
    def prompt_template_path(self) -> Path:
        return Path(__file__).resolve().parent / "prompt.md"

    @property
    def prompt_template(self) -> str:
        return self.prompt_template_path.read_text(encoding="utf-8")

    @property
    def prompt_hash(self) -> str:
        """Short SHA-256 hex of the prompt template for reproducibility."""
        return hashlib.sha256(self.prompt_template.encode()).hexdigest()[:12]

    def __post_init__(self) -> None:
        if not self.data_dir.is_dir():
            raise RuntimeError(
                f"Data directory not found: {self.data_dir}\n"
                f"Run from the project root or set root= explicitly."
            )
