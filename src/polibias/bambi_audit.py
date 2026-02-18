"""Bayesian audit of LLM scoring behaviour using Bambi.

This module runs a two-part analysis:
1) Failure model: probability that a score is present (non-NaN overall bias)
2) Score model: distribution of score values conditional on success
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px

from polibias.analysis import build_bias_frame, build_webdata_frame


@dataclass(frozen=True)
class BambiAuditOptions:
    draws: int = 1000
    tune: int = 1000
    chains: int = 2
    cores: int = 2
    target_accept: float = 0.9
    random_seed: int = 42
    collapse_runs: bool = False


def _safe_z(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd


def _quantize_preflight(df: pd.DataFrame) -> pd.DataFrame:
    vals = df["overall_bias"].dropna()
    rounded = vals.round(3)
    vc = rounded.value_counts().rename_axis("overall_bias_rounded").reset_index(name="n")
    vc["share"] = vc["n"] / len(vals) if len(vals) else np.nan
    return vc


def _build_features(settings, raw_df: pd.DataFrame) -> pd.DataFrame:
    web_df = build_webdata_frame(settings)
    if web_df.empty:
        raw_df = raw_df.copy()
        raw_df["text_words_total"] = np.nan
        raw_df["log_words"] = np.nan
        raw_df["section"] = "unknown"
        return raw_df

    use_cols = ["article_id"]
    for c in ["text_words_total", "article_section"]:
        if c in web_df.columns:
            use_cols.append(c)

    merged = raw_df.merge(web_df[use_cols], on="article_id", how="left")
    merged["text_words_total"] = pd.to_numeric(merged.get("text_words_total"), errors="coerce")
    merged["log_words"] = np.log1p(merged["text_words_total"])
    merged["section"] = merged.get("article_section", pd.Series(index=merged.index)).fillna("unknown")
    merged["confidence"] = pd.to_numeric(merged["confidence"], errors="coerce")
    merged["confidence_z"] = _safe_z(merged["confidence"])
    merged["log_words_z"] = _safe_z(merged["log_words"])
    merged["success"] = merged["overall_bias"].notna().astype(int)
    return merged


def _collapse_for_score(df: pd.DataFrame) -> pd.DataFrame:
    # Collapses repeated runs per model/article to one row while preserving failure rate.
    agg = (
        df.groupby(["model", "article_id"], as_index=False)
        .agg(
            overall_bias=("overall_bias", "mean"),
            confidence=("confidence", "mean"),
            confidence_z=("confidence_z", "mean"),
            log_words=("log_words", "mean"),
            log_words_z=("log_words_z", "mean"),
            success_rate=("success", "mean"),
            n_runs=("run", "nunique"),
            n_rows=("run", "size"),
        )
    )
    return agg


def _write_preflight_outputs(df: pd.DataFrame, out_dir: Path) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)

    failure_by_model = (
        df.groupby("model", as_index=False)
        .agg(
            n_rows=("model", "size"),
            n_success=("success", "sum"),
        )
        .assign(success_rate=lambda d: d["n_success"] / d["n_rows"])
        .sort_values("success_rate", ascending=False)
    )

    score_by_model = (
        df[df["success"] == 1]
        .groupby("model", as_index=False)
        .agg(
            n=("overall_bias", "size"),
            mean_overall_bias=("overall_bias", "mean"),
            sd_overall_bias=("overall_bias", "std"),
            p10=("overall_bias", lambda x: np.quantile(x, 0.10)),
            p50=("overall_bias", lambda x: np.quantile(x, 0.50)),
            p90=("overall_bias", lambda x: np.quantile(x, 0.90)),
        )
        .sort_values("mean_overall_bias")
    )

    quant = _quantize_preflight(df)

    failure_by_model.to_csv(out_dir / "failure_by_model.csv", index=False)
    score_by_model.to_csv(out_dir / "score_by_model.csv", index=False)
    quant.to_csv(out_dir / "overall_bias_quantization.csv", index=False)

    fig_fail = px.bar(
        failure_by_model,
        x="model",
        y="success_rate",
        title="Success rate by model (non-NaN overall_bias)",
    )
    fig_fail.update_yaxes(range=[0, 1])
    fig_fail.write_html(out_dir / "failure_by_model.html", include_plotlyjs="cdn")

    fig_score = px.box(
        df[df["success"] == 1],
        x="model",
        y="overall_bias",
        points="all",
        color="model",
        title="Observed overall_bias by model (success rows only)",
    )
    fig_score.update_layout(showlegend=False)
    fig_score.write_html(out_dir / "score_by_model.html", include_plotlyjs="cdn")

    return {
        "failure_by_model": failure_by_model,
        "score_by_model": score_by_model,
        "quant": quant,
    }


def _bambi_available() -> bool:
    try:
        import bambi  # noqa: F401
        import arviz  # noqa: F401
    except Exception:
        return False
    return True


def _fit_bambi(df: pd.DataFrame, out_dir: Path, options: BambiAuditOptions) -> dict[str, Any]:
    import arviz as az
    import bambi as bmb

    result: dict[str, Any] = {}

    # Part 1: failure model (Bernoulli)
    d_fail = df.copy()
    fail_formula = "success ~ model + confidence_z + log_words_z + (1|article_id)"
    fail_model = bmb.Model(fail_formula, data=d_fail, family="bernoulli")
    idata_fail = fail_model.fit(
        draws=options.draws,
        tune=options.tune,
        chains=options.chains,
        cores=options.cores,
        random_seed=options.random_seed,
        target_accept=options.target_accept,
    )
    az.summary(idata_fail).to_csv(out_dir / "bambi_failure_summary.csv")
    result["failure_formula"] = fail_formula

    # Part 2: score model (Beta) on successful rows
    d_score = df[df["success"] == 1].copy()
    eps = 1e-3
    d_score["y01"] = ((d_score["overall_bias"] + 1.0) / 2.0).clip(eps, 1 - eps)

    if options.collapse_runs:
        d_score = _collapse_for_score(d_score)
        d_score = d_score.dropna(subset=["overall_bias"]).copy()
        d_score["y01"] = ((d_score["overall_bias"] + 1.0) / 2.0).clip(eps, 1 - eps)

    score_formula = "y01 ~ model + confidence_z + log_words_z + (1|article_id)"
    score_model = bmb.Model(score_formula, data=d_score, family="beta")
    idata_score = score_model.fit(
        draws=options.draws,
        tune=options.tune,
        chains=options.chains,
        cores=options.cores,
        random_seed=options.random_seed,
        target_accept=options.target_accept,
    )
    az.summary(idata_score).to_csv(out_dir / "bambi_score_summary.csv")
    result["score_formula"] = score_formula

    # Posterior predictive checks
    ppc_fail = fail_model.predict(idata_fail, kind="pps", inplace=False)
    ppc_score = score_model.predict(idata_score, kind="pps", inplace=False)
    az.summary(ppc_fail).to_csv(out_dir / "bambi_failure_ppc_summary.csv")
    az.summary(ppc_score).to_csv(out_dir / "bambi_score_ppc_summary.csv")

    result["n_failure_rows"] = len(d_fail)
    result["n_score_rows"] = len(d_score)
    return result


def run_bambi_audit(settings, options: BambiAuditOptions | None = None) -> None:
    """Run two-part Bayesian audit and export quant artifacts under run_dir/bayes."""
    options = options or BambiAuditOptions()

    out_dir = settings.run_dir / "bayes"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = build_bias_frame(settings)
    if raw.empty:
        print("  No bias data found. Run 'score' and 'analyse' first.")
        return

    df = _build_features(settings, raw)
    preflight = _write_preflight_outputs(df, out_dir)

    lines = [
        "# Bayesian LLM Audit",
        "",
        "This report audits scoring behavior, not real-world bias truth.",
        "",
        "## Data",
        f"- Rows: {len(df)}",
        f"- Models: {df['model'].nunique()}",
        f"- Articles: {df['article_id'].nunique()}",
        f"- Runs: {df['run'].nunique()}",
        "",
        "## Preflight",
        "- `failure_by_model.csv`: non-NaN success rates",
        "- `score_by_model.csv`: empirical score distribution summary (success rows)",
        "- `overall_bias_quantization.csv`: repeated rounded values (attractor check)",
        "",
    ]

    topq = preflight["quant"].head(8)
    if not topq.empty:
        lines.append("Top repeated rounded overall_bias values:")
        for _, r in topq.iterrows():
            lines.append(f"- {r['overall_bias_rounded']:.3f}: n={int(r['n'])}, share={r['share']:.3f}")
        lines.append("")

    if not _bambi_available():
        lines.extend([
            "## Bambi fit",
            "Bambi is not installed in this environment.",
            "",
            "Install and rerun:",
            "```bash",
            "pip install -e '.[bayes]'",
            f"PYTHONPATH=src python -m polibias bambi --run-dir {settings.run_name}",
            "```",
        ])
        (out_dir / "bambi_audit.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"  Wrote Bayesian preflight outputs to: {out_dir}")
        print("  Bambi not installed; wrote instructions to bambi_audit.md")
        return

    try:
        result = _fit_bambi(df, out_dir, options)
        lines.extend([
            "## Bambi fit",
            "- Status: success",
            f"- Failure model: `{result['failure_formula']}`",
            f"- Score model: `{result['score_formula']}`",
            f"- Failure rows: {result['n_failure_rows']}",
            f"- Score rows: {result['n_score_rows']}",
            "",
            "Outputs:",
            "- `bambi_failure_summary.csv`",
            "- `bambi_score_summary.csv`",
            "- `bambi_failure_ppc_summary.csv`",
            "- `bambi_score_ppc_summary.csv`",
        ])
        (out_dir / "bambi_audit.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"  Wrote Bayesian outputs to: {out_dir}")
    except Exception as e:
        lines.extend([
            "## Bambi fit",
            "- Status: failed",
            f"- Error: `{e}`",
        ])
        (out_dir / "bambi_audit.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"  Bambi fit failed; see {out_dir / 'bambi_audit.md'}")
