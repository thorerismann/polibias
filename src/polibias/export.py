"""Export: standalone HTML report, per-article summaries, LaTeX tables."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ---------- Chart builders ----------

_SOURCE_LABELS = {
    "rts": "RTS",
    "jacobin": "Jacobin",
    "the_federalist": "The Federalist",
    "watson": "Watson",
    "lib_inst": "Liberal Institute",
    "protestinfo": "Protestinfo",
    "cathinfo": "Cathinfo",
}


def _label_source(source: str) -> str:
    return _SOURCE_LABELS.get(source, str(source))


def _style_hover(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="#111827",
            bordercolor="#374151",
            font=dict(color="#F9FAFB", size=12),
            align="left",
        )
    )
    return fig


def _wrap_for_hover(value: str, width: int = 85) -> str:
    text = (value or "").strip()
    if not text:
        return "(no comment)"
    parts = []
    for line in text.splitlines():
        wrapped = textwrap.wrap(line, width=width, break_long_words=False, replace_whitespace=False)
        parts.extend(wrapped or [""])
    return "<br>".join(parts)


def _bias_scatter_by_model(bdf: pd.DataFrame) -> go.Figure:
    df = bdf.copy()
    df["source_label"] = df["source"].astype(str).map(_label_source)
    df["comment_hover"] = df["comment"].fillna("").astype(str).map(_wrap_for_hover)
    fig = px.scatter(
        df,
        x="model",
        y="overall_bias",
        color="source_label",
        symbol="run",
        custom_data=[
            "run",
            "article_id",
            "confidence",
            "subject_bias",
            "framing_bias",
            "treatment_bias",
            "guests_bias",
            "comment_hover",
        ],
        hover_data={
            "source_label": False,
            "source": False,
            "model": True,
            "run": True,
            "article_id": True,
            "overall_bias": ":.3f",
            "subject_bias": ":.3f",
            "framing_bias": ":.3f",
            "treatment_bias": ":.3f",
            "guests_bias": ":.3f",
            "confidence": ":.3f",
            "comment": True,
        },
        labels={
            "model": "Model",
            "overall_bias": "Overall bias",
            "source_label": "Article source",
        },
        title="Overall bias by model (article source color)",
    )
    fig.update_yaxes(range=[-1, 1])
    fig.update_xaxes(title="Model")
    fig.update_layout(legend_title_text="Article source")
    fig.update_traces(
        hovertemplate=(
            "<b>Model:</b> %{x}<br>"
            "<b>Bias:</b> %{y:.3f}<br>"
            "<b>Article source:</b> %{fullData.name}<br>"
            "<b>Run:</b> %{customdata[0]}<br>"
            "<b>Article:</b> %{customdata[1]}<br>"
            "<b>Confidence:</b> %{customdata[2]:.3f}<br>"
            "<b>Sub-biases:</b> subject=%{customdata[3]:.3f}, framing=%{customdata[4]:.3f}, "
            "treatment=%{customdata[5]:.3f}, guests=%{customdata[6]:.3f}<br>"
            "<b>Comment:</b><br>%{customdata[7]}<extra></extra>"
        )
    )
    return _style_hover(fig)


def _bias_box_by_model(bdf: pd.DataFrame) -> go.Figure:
    fig = px.box(bdf, x="model", y="overall_bias", points="all", color="model",
                 title="Bias variance by model")
    fig.update_layout(showlegend=False)
    return _style_hover(fig)


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
    return _style_hover(fig)


def _subbias_dot_by_article(bdf: pd.DataFrame) -> go.Figure:
    """Dot chart across the 4 sub-bias dimensions (article x bias, model as color)."""
    dims = ["subject_bias", "framing_bias", "treatment_bias", "guests_bias"]
    long_df = bdf.melt(
        id_vars=["model", "article_id", "run", "comment", "source"],
        value_vars=dims,
        var_name="dimension",
        value_name="bias",
    )
    long_df["article_str"] = long_df["article_id"].astype(str)

    fig = px.scatter(
        long_df,
        x="article_str",
        y="bias",
        color="model",
        facet_col="dimension",
        facet_col_wrap=2,
        hover_data={
            "article_str": False,
            "article_id": True,
            "model": True,
            "run": True,
            "source": True,
            "comment": True,
        },
        title="Sub-bias dots by article (4 dimensions)",
    )
    fig.update_yaxes(range=[-1, 1])
    fig.update_xaxes(type="category", tickangle=45)
    fig.update_layout(height=850)
    return _style_hover(fig)


def _overall_bias_dot_by_article(bdf: pd.DataFrame) -> go.Figure:
    """Dot chart with same underlying article-model-run data as the heatmap family."""
    df = bdf.copy()
    df["article_str"] = df["article_id"].astype(str)
    df["comment"] = df["comment"].fillna("").astype(str)
    df["comment_hover"] = df["comment"].map(_wrap_for_hover)
    df["source_label"] = df["source"].astype(str).map(_label_source)

    fig = px.scatter(
        df,
        x="article_str",
        y="overall_bias",
        color="model",
        symbol="run",
        facet_col="source_label",
        facet_col_wrap=2,
        custom_data=[
            "model",
            "run",
            "source_label",
            "subject_bias",
            "framing_bias",
            "treatment_bias",
            "guests_bias",
            "comment_hover",
            "confidence",
        ],
        hover_data={
            "article_str": False,
            "article_id": True,
            "model": True,
            "run": True,
            "source_label": True,
            "comment": True,
            "subject_bias": ":.3f",
            "framing_bias": ":.3f",
            "treatment_bias": ":.3f",
            "guests_bias": ":.3f",
            "overall_bias": ":.3f",
            "confidence": ":.3f",
        },
        title="Overall bias dots by article and source (model color, run symbol)",
    )
    fig.update_yaxes(range=[-1, 1], title="overall_bias")
    fig.update_xaxes(type="category", tickangle=45, title="article_id")
    fig.update_layout(height=900)
    fig.update_traces(
        hovertemplate=(
            "<b>Article:</b> %{x}<br>"
            "<b>Bias:</b> %{y:.3f}<br>"
            "<b>Model:</b> %{customdata[0]}<br>"
            "<b>Run:</b> %{customdata[1]}<br>"
            "<b>Article source:</b> %{customdata[2]}<br>"
            "<b>Confidence:</b> %{customdata[8]:.3f}<br>"
            "<b>Sub-biases:</b> subject=%{customdata[3]:.3f}, framing=%{customdata[4]:.3f}, "
            "treatment=%{customdata[5]:.3f}, guests=%{customdata[6]:.3f}<br>"
            "<b>Comment:</b><br>%{customdata[7]}<extra></extra>"
        )
    )
    return _style_hover(fig)


def _confidence_vs_bias(bdf: pd.DataFrame) -> go.Figure:
    bdf = bdf.copy()
    bdf["abs_bias"] = bdf["overall_bias"].abs()
    fig = px.scatter(
        bdf, x="confidence", y="abs_bias", color="model",
        title="Confidence vs |overall bias|",
    )
    return _style_hover(fig)


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
    return _style_hover(fig)


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
.comment-controls { display: grid; grid-template-columns: repeat(3, minmax(180px, 1fr)); gap: 10px;
                    margin: 14px 0; align-items: end; }
.comment-controls label { display: block; font-size: 0.9em; color: #555; margin-bottom: 4px; }
.comment-controls select { width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 6px; }
.comment-box { border: 1px solid #ddd; border-radius: 8px; background: #fafafa;
               min-height: 70px; padding: 12px; white-space: pre-wrap; }
"""


def _df_to_html(df: pd.DataFrame, float_fmt: str = "%.3f") -> str:
    return df.to_html(index=False, float_format=float_fmt, na_rep="—")


def _comment_explorer_html(bdf: pd.DataFrame) -> str:
    cols = [
        "model", "run", "source", "article_id", "comment", "subject_bias", "framing_bias",
        "treatment_bias", "guests_bias", "overall_bias", "confidence", "status",
    ]
    data = bdf[cols].copy().fillna("")
    records = data.to_dict(orient="records")
    records_json = json.dumps(records, ensure_ascii=False)

    return f"""
<div id=\"comment-explorer\">
  <div class=\"comment-controls\">
    <div>
      <label for=\"model-select\">Model</label>
      <select id=\"model-select\"></select>
    </div>
    <div>
      <label for=\"run-select\">Run</label>
      <select id=\"run-select\"></select>
    </div>
    <div>
      <label for=\"article-select\">Article</label>
      <select id=\"article-select\"></select>
    </div>
  </div>
  <div id=\"comment-box\" class=\"comment-box\">Select model, run, and article.</div>
</div>
<script>
(function() {{
  const rows = {records_json};
  const modelSel = document.getElementById('model-select');
  const runSel = document.getElementById('run-select');
  const articleSel = document.getElementById('article-select');
  const box = document.getElementById('comment-box');

  function uniq(values) {{
    return Array.from(new Set(values)).filter(v => v !== '').sort();
  }}

  function setOptions(selectEl, values) {{
    selectEl.innerHTML = '';
    const all = document.createElement('option');
    all.value = '__ALL__';
    all.textContent = 'All';
    selectEl.appendChild(all);
    for (const v of values) {{
      const opt = document.createElement('option');
      opt.value = String(v);
      opt.textContent = String(v);
      selectEl.appendChild(opt);
    }}
  }}

  function selectedValue(sel) {{
    return sel.value === '__ALL__' ? null : sel.value;
  }}

  function filterRows() {{
    const m = selectedValue(modelSel);
    const r = selectedValue(runSel);
    const a = selectedValue(articleSel);
    return rows.filter(x =>
      (!m || String(x.model) === m) &&
      (!r || String(x.run) === r) &&
      (!a || String(x.article_id) === a)
    );
  }}

  function refreshRunOptions() {{
    const m = selectedValue(modelSel);
    const filtered = rows.filter(x => !m || String(x.model) === m);
    const current = runSel.value;
    setOptions(runSel, uniq(filtered.map(x => String(x.run))));
    if (Array.from(runSel.options).some(o => o.value === current)) runSel.value = current;
  }}

  function refreshArticleOptions() {{
    const m = selectedValue(modelSel);
    const r = selectedValue(runSel);
    const filtered = rows.filter(x =>
      (!m || String(x.model) === m) &&
      (!r || String(x.run) === r)
    );
    const current = articleSel.value;
    setOptions(articleSel, uniq(filtered.map(x => String(x.article_id))));
    if (Array.from(articleSel.options).some(o => o.value === current)) articleSel.value = current;
  }}

  function updateCommentBox() {{
    const filtered = filterRows();
    if (!filtered.length) {{
      box.textContent = 'No matching records for this selection.';
      return;
    }}

    const lines = filtered.slice(0, 10).map(x => {{
      const c = x.comment && String(x.comment).trim() ? String(x.comment) : '(no comment)';
      return `model=${{x.model}} | run=${{x.run}} | source=${{x.source}} | article=${{x.article_id}}\n` +
             `status=${{x.status}} | conf=${{x.confidence}} | overall=${{x.overall_bias}}\n` +
             `subject=${{x.subject_bias}}, framing=${{x.framing_bias}}, treatment=${{x.treatment_bias}}, guests=${{x.guests_bias}}\n` +
             `comment: ${{c}}`;
    }});

    const suffix = filtered.length > 10 ? `\n\nShowing 10 of ${{filtered.length}} matches.` : '';
    box.textContent = lines.join('\n\n---\n\n') + suffix;
  }}

  setOptions(modelSel, uniq(rows.map(x => String(x.model))));
  refreshRunOptions();
  refreshArticleOptions();
  updateCommentBox();

  modelSel.addEventListener('change', () => {{
    refreshRunOptions();
    refreshArticleOptions();
    updateCommentBox();
  }});
  runSel.addEventListener('change', () => {{
    refreshArticleOptions();
    updateCommentBox();
  }});
  articleSel.addEventListener('change', updateCommentBox);
}})();
</script>
"""


# ---------- Main report builder ----------

def build_html_report(
    bias_df: pd.DataFrame,
    output_path: str,
    kappa_bins: int = 5,
    *,
    title: str = "polibias — Bias Analysis Report",
    include_tables: bool = True,
) -> None:
    """Generate a standalone HTML report."""
    from polibias.stats import build_stats_report

    report = build_stats_report(bias_df, kappa_bins=kappa_bins)

    figs = [
        _bias_scatter_by_model(bias_df),
        _bias_box_by_model(bias_df),
        _ci_forest_plot(report["model_ci"]),
        _bias_heatmap(bias_df),
        _overall_bias_dot_by_article(bias_df),
        _subbias_dot_by_article(bias_df),
        _confidence_vs_bias(bias_df),
    ]

    html_parts = [
        "<!DOCTYPE html><html><head>",
        '<meta charset="utf-8">',
        "<title>polibias report</title>",
        f"<style>{_CSS}</style>",
        "</head><body>",
        f"<h1>{title}</h1>",
    ]

    if include_tables:
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
    else:
        html_parts.append('<p class="section-note">Compact source report (charts + comment explorer).</p>')

    # --- Charts ---
    html_parts.append("<h2>Charts</h2>")
    for fig in figs:
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))

    # --- Comment explorer ---
    html_parts.append("<h2>Comment explorer</h2>")
    html_parts.append(
        '<p class="section-note">Use model/run/article selectors to inspect original JSON comments and scores.</p>'
    )
    html_parts.append(_comment_explorer_html(bias_df))

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

def run_export(
    settings,
    *,
    source: str | None = None,
    output_filename: str | None = None,
    include_tables: bool = True,
    write_artifacts: bool = True,
) -> None:
    """Load bias data and generate export outputs."""
    from polibias.analysis import build_bias_frame

    settings.run_dir.mkdir(parents=True, exist_ok=True)
    bias_df = build_bias_frame(settings)
    if source is not None:
        bias_df = bias_df[bias_df["source"] == source].copy()
    if bias_df.empty:
        print("  No bias data found. Run 'score' and 'analyse' first.")
        return

    # HTML report
    report_path = settings.run_dir / output_filename if output_filename else settings.report_html_path
    html_path = str(report_path)
    title = "polibias — Bias Analysis Report" if source is None else f"polibias — {source} Bias Report"
    build_html_report(
        bias_df,
        html_path,
        kappa_bins=settings.kappa_bins,
        title=title,
        include_tables=include_tables,
    )
    print(f"  Wrote HTML report: {html_path}")

    if write_artifacts:
        # Article summaries
        summaries = build_article_summaries(bias_df)
        summary_path = settings.article_summaries_csv_path
        summaries.to_csv(summary_path, index=False)
        print(f"  Wrote article summaries: {summary_path}")

        # LaTeX
        latex = build_latex_table(bias_df)
        latex_path = settings.latex_table_path
        with open(latex_path, "w", encoding="utf-8") as f:
            f.write(latex)
        print(f"  Wrote LaTeX table: {latex_path}")


def run_export_cross_source(
    settings,
    *,
    output_filename: str = "report_all.html",
    source_reports: Mapping[str, str] | None = None,
) -> None:
    from polibias.analysis import build_bias_frame

    bias_df = build_bias_frame(settings)
    if bias_df.empty:
        print("  No bias data found. Run 'score' and 'analyse' first.")
        return

    by_source_model = (
        bias_df.groupby(["source", "model"], as_index=False)
        .agg(
            overall_bias_mean=("overall_bias", "mean"),
            overall_bias_std=("overall_bias", "std"),
            confidence_mean=("confidence", "mean"),
            n_scores=("overall_bias", "count"),
            n_articles=("article_id", "nunique"),
        )
        .sort_values(["source", "overall_bias_mean"], ascending=[True, False])
    )
    by_source_dim = (
        bias_df.groupby("source", as_index=False)
        .agg(
            subject_bias_mean=("subject_bias", "mean"),
            framing_bias_mean=("framing_bias", "mean"),
            treatment_bias_mean=("treatment_bias", "mean"),
            guests_bias_mean=("guests_bias", "mean"),
            overall_bias_mean=("overall_bias", "mean"),
            confidence_mean=("confidence", "mean"),
            n_scores=("overall_bias", "count"),
            n_articles=("article_id", "nunique"),
        )
        .sort_values("source")
    )
    cross_tab = (
        pd.pivot_table(
            bias_df,
            index="article_id",
            columns="source",
            values="overall_bias",
            aggfunc="mean",
        )
        .reset_index()
        .sort_values("article_id")
    )
    source_model_heat = (
        bias_df.groupby(["source", "model"], as_index=False)["overall_bias"]
        .mean()
        .pivot(index="source", columns="model", values="overall_bias")
    )
    fig = px.imshow(
        source_model_heat,
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Mean overall bias by source and model",
    )

    links_html = ""
    if source_reports:
        links = [
            f'<li><a href="{Path(name).name}">{source}</a></li>'
            for source, name in source_reports.items()
        ]
        links_html = "<h2>Per-source reports</h2><ul>" + "".join(links) + "</ul>"

    html_parts = [
        "<!DOCTYPE html><html><head>",
        '<meta charset="utf-8">',
        "<title>polibias cross-source report</title>",
        f"<style>{_CSS}</style>",
        "</head><body>",
        "<h1>polibias — Cross-source report</h1>",
        '<p class="section-note">Cross-reference view across all sources in this run.</p>',
        links_html,
        "<h2>Source summary</h2>",
        _df_to_html(by_source_dim),
        "<h2>Source x model summary</h2>",
        _df_to_html(by_source_model),
        "<h2>Article cross-tab (mean overall bias)</h2>",
        _df_to_html(cross_tab),
        "<h2>Heatmap</h2>",
        fig.to_html(full_html=False, include_plotlyjs="cdn"),
        "</body></html>",
    ]

    out_path = settings.run_dir / output_filename
    out_path.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"  Wrote cross-source HTML report: {out_path}")
