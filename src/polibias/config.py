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
    run_name: str = "run_results"

    # ---- runtime params ----
    runs: int = 4
    timeout: int = 400
    scrape_timeout: int = 30
    parse_retries: int = 2
    max_workers: int = 2
    max_article_chars: int = 6000
    ollama_host: str = "http://127.0.0.1:11434"
    keep_alive: str = "5m"
    kappa_bins: int = 5

    models: List[str] = field(default_factory=lambda: [
        "llama3.2:latest",
        "gemma2:latest",
        "phi3:mini",
        "qwen2.5:3b-instruct",
        "gemma3:4b",
        "MichelRosselli/apertus:latest",
        "mistral:latest",
        "llama3.1:8b",
        "olmo-3:7b",
        "phi3.5:latest",
        
    ])

    ollama_options: ModelOptions = field(default_factory=lambda: {
        "temperature": 0.4,
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
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def run_dir(self) -> Path:
        return self.runs_dir / self.run_name

    @property
    def results_dir(self) -> Path:
        return self.run_dir / "results"

    @property
    def errors_dir(self) -> Path:
        return self.run_dir / "errors"

    @property
    def stats_csv_path(self) -> Path:
        return self.run_dir / "stats_report.csv"

    @property
    def report_html_path(self) -> Path:
        return self.run_dir / "report.html"

    @property
    def bias_csv_path(self) -> Path:
        return self.run_dir / "bias_data.csv"

    @property
    def web_csv_path(self) -> Path:
        return self.run_dir / "web_data.csv"

    @property
    def article_summaries_csv_path(self) -> Path:
        return self.run_dir / "article_summaries.csv"

    @property
    def latex_table_path(self) -> Path:
        return self.run_dir / "bias_table.tex"

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
        # Cached on first access to avoid re-reading disk per article.
        # Uses object.__setattr__ because the dataclass is frozen.
        try:
            return self._prompt_template_cache  # type: ignore[attr-defined]
        except AttributeError:
            text = self.prompt_template_path.read_text(encoding="utf-8")
            object.__setattr__(self, "_prompt_template_cache", text)
            return text

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
        run_name = self.run_name.strip()
        if not run_name:
            raise RuntimeError("run_name cannot be empty.")
        if run_name in {".", ".."} or "/" in run_name or "\\" in run_name:
            raise RuntimeError("run_name must be a simple folder name (no path separators).")


def load_settings(config_path: Path | None = None, **overrides: Any) -> Settings:
    """Create Settings from an optional TOML config file and CLI overrides.

    TOML keys map directly to Settings field names.  CLI overrides
    (e.g. ``run_name="foo"``) take precedence over the file.
    """
    import tomllib

    values: Dict[str, Any] = {}
    if config_path is not None:
        with open(config_path, "rb") as f:
            raw = tomllib.load(f)
        # Flatten [polibias] section if present, otherwise use top-level keys.
        values = raw.get("polibias", raw)
    values.update({k: v for k, v in overrides.items() if v is not None})

    # Coerce types that TOML may read differently than the dataclass expects.
    if "root" in values:
        values["root"] = Path(values["root"])

    return Settings(**values)
