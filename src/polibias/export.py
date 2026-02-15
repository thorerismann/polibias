"""Export: standalone HTML report, per-article summaries, LaTeX tables."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _bias_scatter_by_model(bdf: pd.DataFrame) -> go.Figure:
    return px.scatter(
        bdf, y="overall_bias", x="model", color="confidence",
        title="Overall bias by model",
    )


def _bias_box_by_model(bdf: pd.DataFrame) -> go.Figure:
    fig = px.box(bdf, x="model", y="overall_bias", points="all", color="model",
                 title="Bias variance by model")
    fig.update_layout(showlegend=False)
    return fig


def _bias_heatmap(bdf: pd.DataFrame) -> go.Figure:
    bdf = bdf.copy()
    bdf["article_str"] = bdf["article_id"].astype(str)
    pivot = bdf.pivot_table(
        index="article_str", columns="model", values="overall_bias", aggfunc="mean",
    )
    height = max(400, 30 * len(pivot))
    fig = px.imshow(
        pivot, color_continuous_scale="RdBu", zmin=-1, zmax=1,
        aspect="auto", title="Mean bias: articles x models",
    )
    fig.update_layout(height=height, margin=dict(l=200, r=20, t=60, b=40))
    return fig


def _confidence_vs_bias(bdf: pd.DataFrame) -> go.Figure:
    bdf = bdf.copy()
    bdf["abs_bias"] = bdf["overall_bias"].abs()
    return px.scatter(
        bdf, x="confidence", y="abs_bias", color="model",
        title="Confidence vs |overall bias|",
    )


def build_html_report(bias_df: pd.DataFrame, output_path: str) -> None:
    """Generate a standalone HTML report with embedded Plotly charts."""
    figs = [
        _bias_scatter_by_model(bias_df),
        _bias_box_by_model(bias_df),
        _bias_heatmap(bias_df),
        _confidence_vs_bias(bias_df),
    ]

    html_parts = [
        "<!DOCTYPE html><html><head>",
        '<meta charset="utf-8">',
        "<title>polibias report</title>",
        "<style>body{font-family:sans-serif;max-width:1200px;margin:auto;padding:20px}"
        "h1{color:#333}table{border-collapse:collapse;width:100%;margin:20px 0}"
        "th,td{border:1px solid #ddd;padding:8px;text-align:right}"
        "th{background:#f5f5f5}</style>",
        "</head><body>",
        "<h1>polibias — Bias Analysis Report</h1>",
    ]

    # Summary table
    summary = bias_df.groupby("model").agg(
        n=("overall_bias", "count"),
        mean_bias=("overall_bias", "mean"),
        std_bias=("overall_bias", "std"),
        mean_confidence=("confidence", "mean"),
        ok_pct=("status", lambda x: f"{(x == 'ok').mean():.0%}"),
    ).reset_index()
    html_parts.append("<h2>Model summary</h2>")
    html_parts.append(summary.to_html(index=False, float_format="%.3f"))

    # Charts
    for fig in figs:
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))

    html_parts.append("</body></html>")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))


def build_article_summaries(bias_df: pd.DataFrame) -> pd.DataFrame:
    """Per-article summary: mean bias across all models, model agreement."""
    if bias_df.empty:
        return pd.DataFrame()

    rows = []
    for article_id, grp in bias_df.groupby("article_id"):
        row = {"article_id": article_id}
        for col in ["subject_bias", "framing_bias", "treatment_bias", "guests_bias", "overall_bias"]:
            vals = grp[col].dropna()
            if not vals.empty:
                row[f"{col}_mean"] = float(vals.mean())
                row[f"{col}_std"] = float(vals.std(ddof=1))
        row["n_scores"] = len(grp)
        row["n_models"] = grp["model"].nunique()
        rows.append(row)
    return pd.DataFrame(rows)


def build_latex_table(bias_df: pd.DataFrame) -> str:
    """Generate a LaTeX table of mean bias per model."""
    summary = bias_df.groupby("model").agg(
        n=("overall_bias", "count"),
        subject=("subject_bias", "mean"),
        framing=("framing_bias", "mean"),
        treatment=("treatment_bias", "mean"),
        guests=("guests_bias", "mean"),
        overall=("overall_bias", "mean"),
        confidence=("confidence", "mean"),
    ).reset_index()

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Mean bias scores per model}",
        r"\label{tab:bias_scores}",
        r"\begin{tabular}{l r r r r r r r}",
        r"\hline",
        r"Model & $n$ & Subject & Framing & Treatment & Guests & Overall & Confidence \\",
        r"\hline",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"{r['model']} & {r['n']:.0f} & {r['subject']:.3f} & {r['framing']:.3f} "
            f"& {r['treatment']:.3f} & {r['guests']:.3f} & {r['overall']:.3f} "
            f"& {r['confidence']:.3f} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def run_export(settings) -> None:
    """Load bias data and generate all export outputs."""
    from polibias.analysis import build_bias_frame

    bias_df = build_bias_frame(settings)
    if bias_df.empty:
        print("  No bias data found. Run 'score' and 'analyse' first.")
        return

    # HTML report
    html_path = str(settings.report_html_path)
    build_html_report(bias_df, html_path)
    print(f"  Wrote HTML report: {html_path}")

    # Article summaries
    summaries = build_article_summaries(bias_df)
    summary_path = settings.data_dir / "article_summaries.csv"
    summaries.to_csv(summary_path, index=False)
    print(f"  Wrote article summaries: {summary_path}")

    # LaTeX
    latex = build_latex_table(bias_df)
    latex_path = settings.data_dir / "bias_table.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"  Wrote LaTeX table: {latex_path}")
