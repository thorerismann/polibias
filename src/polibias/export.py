"""Export: standalone HTML report, per-article summaries, LaTeX tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ---------- Chart builders ----------

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


def _ci_forest_plot(ci_df: pd.DataFrame) -> go.Figure:
    """Horizontal forest plot of mean overall_bias with 95% CIs per model."""
    if ci_df.empty:
        return go.Figure()
    fig = go.Figure()
    for _, row in ci_df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["ci_low"], row["ci_high"]],
            y=[row["model"], row["model"]],
            mode="lines",
            line=dict(width=3),
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[row["mean_overall_bias"]],
            y=[row["model"]],
            mode="markers",
            marker=dict(size=10, symbol="diamond"),
            showlegend=False,
        ))
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Mean overall bias with 95% CI per model",
        xaxis_title="overall_bias",
        yaxis_title="",
        xaxis=dict(range=[-0.8, 0.4]),
    )
    return fig


# ---------- HTML helpers ----------

_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1200px; margin: auto; padding: 20px; color: #222; }
h1 { color: #333; border-bottom: 2px solid #ddd; padding-bottom: 10px; }
h2 { color: #555; margin-top: 40px; }
table { border-collapse: collapse; width: 100%; margin: 20px 0; }
th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: right; }
th { background: #f5f5f5; font-weight: 600; }
td:first-child, th:first-child { text-align: left; }
tr:nth-child(even) { background: #fafafa; }
.kappa-box { background: #f0f4ff; border: 1px solid #c0d0f0; border-radius: 8px;
             padding: 16px 24px; margin: 20px 0; display: inline-block; }
.kappa-box .value { font-size: 2em; font-weight: bold; }
.kappa-box .label { color: #666; }
.section-note { color: #777; font-size: 0.9em; margin-top: -10px; }
"""


def _df_to_html(df: pd.DataFrame, float_fmt: str = "%.3f") -> str:
    return df.to_html(index=False, float_format=float_fmt, na_rep="—")


# ---------- Main report builder ----------

def build_html_report(bias_df: pd.DataFrame, output_path: str) -> None:
    """Generate a standalone HTML report with charts and stats tables."""
    from polibias.stats import build_stats_report

    report = build_stats_report(bias_df)

    figs = [
        _bias_scatter_by_model(bias_df),
        _bias_box_by_model(bias_df),
        _ci_forest_plot(report["model_ci"]),
        _bias_heatmap(bias_df),
        _confidence_vs_bias(bias_df),
    ]

    html_parts = [
        "<!DOCTYPE html><html><head>",
        '<meta charset="utf-8">',
        "<title>polibias report</title>",
        f"<style>{_CSS}</style>",
        "</head><body>",
        "<h1>polibias — Bias Analysis Report</h1>",
    ]

    # --- Model summary table ---
    html_parts.append("<h2>Model summary</h2>")
    html_parts.append('<p class="section-note">Mean bias scores per dimension. '
                      'Scale: -1.0 (left) to +1.0 (right).</p>')
    if not report["model_summary"].empty:
        summary = report["model_summary"].copy()
        display_cols = ["model"]
        for c in ["overall_bias_mean", "subject_bias_mean", "framing_bias_mean",
                   "treatment_bias_mean", "guests_bias_mean",
                   "overall_bias_std", "n_ok", "n_recovered", "n_fallback"]:
            if c in summary.columns:
                display_cols.append(c)
        html_parts.append(_df_to_html(summary[display_cols]))

    # --- Confidence intervals table ---
    html_parts.append("<h2>Confidence intervals (95%)</h2>")
    html_parts.append('<p class="section-note">If the CI crosses zero, the model\'s '
                      'bias is not statistically distinguishable from neutral.</p>')
    if not report["model_ci"].empty:
        html_parts.append(_df_to_html(report["model_ci"]))

    # --- ICC table ---
    html_parts.append("<h2>Within-model consistency (ICC)</h2>")
    html_parts.append('<p class="section-note">'
                      'ICC(1,1) measures how consistently a model scores the same article '
                      'across runs. 0 = random, 1 = perfect. Below 0.4 is poor.</p>')
    if not report["icc_per_model"].empty:
        html_parts.append(_df_to_html(report["icc_per_model"]))

    # --- Fleiss' kappa ---
    kappa = report["fleiss_kappa"]
    kappa_str = f"{kappa:.3f}" if not np.isnan(kappa) else "N/A"
    if np.isnan(kappa):
        kappa_interp = "insufficient data"
    elif kappa < 0:
        kappa_interp = "less than chance"
    elif kappa < 0.21:
        kappa_interp = "slight agreement"
    elif kappa < 0.41:
        kappa_interp = "fair agreement"
    elif kappa < 0.61:
        kappa_interp = "moderate agreement"
    elif kappa < 0.81:
        kappa_interp = "substantial agreement"
    else:
        kappa_interp = "near-perfect agreement"

    html_parts.append("<h2>Inter-model agreement (Fleiss' kappa)</h2>")
    html_parts.append('<p class="section-note">'
                      "Measures how much the models agree when scoring the same articles. "
                      "Scale: &lt;0.20 slight, 0.21-0.40 fair, 0.41-0.60 moderate, "
                      "0.61-0.80 substantial, &gt;0.80 near-perfect.</p>")
    html_parts.append(
        f'<div class="kappa-box">'
        f'<span class="value">{kappa_str}</span> '
        f'<span class="label">— {kappa_interp}</span>'
        f'</div>'
    )

    # --- Charts ---
    html_parts.append("<h2>Charts</h2>")
    for fig in figs:
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))

    html_parts.append("</body></html>")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))


# ---------- Article summaries ----------

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


# ---------- LaTeX ----------

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


# ---------- Entry point ----------

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
