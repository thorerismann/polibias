"""Export: standalone HTML report, per-article summaries, LaTeX tables."""

from __future__ import annotations

import html
import json
import textwrap
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from polibias.analysis import model_cohort, model_display_name


# ---------- Chart builders ----------

_SOURCE_LABELS = {
    "rts": "RTS",
    "jacobin": "Jacobin",
    "the_federalist": "The Federalist",
    "watson": "Watson",
    "protestinfo": "Protestinfo",
    "cathinfo": "Cathinfo",
}
_COMMENT_STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "have", "has", "was", "were", "are",
    "but", "not", "its", "their", "they", "them", "into", "about", "than", "also", "only",
    "very", "more", "most", "much", "many", "such", "can", "could", "would", "should", "there",
    "here", "when", "where", "what", "which", "while", "without", "between", "dans", "avec",
    "pour", "une", "des", "les", "est", "sur", "pas", "plus", "comme", "mais", "cette", "cet",
    "ces", "aux", "par", "qui", "que", "sans", "tout", "tous", "elle", "elles", "nous", "vous",
    "ils", "leur", "leurs", "sein", "dont", "afin", "der", "die", "das", "und", "ist", "mit",
    "nicht", "eine", "einer", "einem", "einen", "den", "dem", "des", "ein", "auch", "als",
    "auf", "für", "von", "bei", "aus", "über", "durch", "wird", "sind", "war", "waren", "dass",
    "oder", "zum", "zur", "im", "am", "article", "articles", "comment", "comments", "report",
    "reports", "reporting", "bias", "biased", "political", "politically", "politique",
    "politiques", "model", "models", "source", "sources", "left", "right", "leaning", "neutral",
    "identifiable", "clear", "clearly", "appears", "overall", "focus", "focuses", "focused",
    "cover", "covers", "covered", "coverage", "discusses", "describes", "described", "presents",
    "presented", "larticle", "factuel", "factual", "factuelle", "factually",
}
_COMMENT_DESCRIPTIVE_SUFFIXES = (
    "able", "ible", "al", "ant", "ary", "ative", "ed", "ent", "ful", "ial", "ic", "ical", "ish",
    "ive", "less", "ory", "ous", "y", "aire", "euse", "eux", "if", "ifs", "ives", "ique", "iques",
)
_COMMENT_DESCRIPTIVE_WHITELIST = {
    "alarmist", "balanced", "balance", "critical", "cautious", "controversial", "diplomatic",
    "ethical", "favorable", "favourable", "hostile", "humanitarian", "negative", "positive",
    "optimistic", "pessimistic", "sympathetic", "technical", "tense", "urgent", "neutre",
    "neutres", "negatif", "positif", "positives", "critique", "critiques", "equilibre",
    "equilibree", "equilibrees", "prudent", "prudente", "prudentes", "favorables",
    "humanitaire", "humanitaires", "alarmiste", "alarmistes", "technique", "techniques",
    "tendu", "tendue",
}


def _label_source(source: str) -> str:
    return _SOURCE_LABELS.get(source, str(source))


def _with_model_metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["model_cohort"] = out["model"].map(model_cohort)
    out["model_display"] = out["model"].map(model_display_name)
    return out


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
    df = _with_model_metadata(bdf)
    df["source_label"] = df["source"].astype(str).map(_label_source)
    df["comment_hover"] = df["comment"].fillna("").astype(str).map(_wrap_for_hover)
    fig = px.scatter(
        df,
        x="model_display",
        y="overall_bias",
        color="source_label",
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
            "model_display": True,
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
            "model_display": "Model",
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
    df = _with_model_metadata(bdf)
    fig = px.box(df, x="model_display", y="overall_bias", points="all", color="model_display",
                 title="Bias variance by model")
    fig.update_layout(showlegend=False)
    return _style_hover(fig)


def _bias_heatmap(bdf: pd.DataFrame) -> go.Figure:
    bdf = _with_model_metadata(bdf)
    bdf["article_str"] = bdf["article_id"].astype(str)
    pivot = bdf.pivot_table(
        index="article_str", columns="model_display", values="overall_bias", aggfunc="mean",
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
    long_df = _with_model_metadata(bdf).melt(
        id_vars=["model", "model_display", "article_id", "run", "comment", "source"],
        value_vars=dims,
        var_name="dimension",
        value_name="bias",
    )
    long_df["article_str"] = long_df["article_id"].astype(str)

    fig = px.scatter(
        long_df,
        x="article_str",
        y="bias",
        color="model_display",
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
    df = _with_model_metadata(bdf)
    df["article_str"] = df["article_id"].astype(str)
    df["comment"] = df["comment"].fillna("").astype(str)
    df["comment_hover"] = df["comment"].map(_wrap_for_hover)
    df["source_label"] = df["source"].astype(str).map(_label_source)
    source_order = (
        df[["source", "source_label"]]
        .drop_duplicates()
        .sort_values("source_label")
        .reset_index(drop=True)
    )
    model_order = sorted(df["model_display"].dropna().astype(str).unique())
    palette = px.colors.qualitative.Plotly
    color_map = {m: palette[i % len(palette)] for i, m in enumerate(model_order)}

    fig = make_subplots(
        rows=len(source_order),
        cols=1,
        shared_xaxes=False,
        shared_yaxes=True,
        vertical_spacing=0.06,
        subplot_titles=source_order["source_label"].tolist(),
    )

    for row_idx, source in enumerate(source_order["source"], start=1):
        src_df = df[df["source"] == source].copy()
        article_order = src_df["article_str"].dropna().astype(str).unique().tolist()

        for model in model_order:
            model_df = src_df[src_df["model_display"].astype(str) == model]
            if model_df.empty:
                continue

            custom_data = np.column_stack(
                [
                    model_df["model_display"].astype(str),
                    model_df["run"].astype(str),
                    model_df["source_label"].astype(str),
                    model_df["subject_bias"],
                    model_df["framing_bias"],
                    model_df["treatment_bias"],
                    model_df["guests_bias"],
                    model_df["comment_hover"].astype(str),
                    model_df["confidence"],
                ]
            )
            fig.add_trace(
                go.Scatter(
                    x=model_df["article_str"].astype(str),
                    y=model_df["overall_bias"],
                    mode="markers",
                    name=model,
                    legendgroup=model,
                    showlegend=(row_idx == 1),
                    marker=dict(color=color_map[model], size=8),
                    customdata=custom_data,
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
                    ),
                ),
                row=row_idx,
                col=1,
            )
        fig.update_xaxes(
            type="category",
            tickangle=45,
            title_text="article_id",
            categoryorder="array",
            categoryarray=article_order,
            row=row_idx,
            col=1,
        )
        fig.update_yaxes(range=[-1, 1], title_text="overall_bias", row=row_idx, col=1)

    fig.update_layout(
        title="Overall bias dots by article and source (model color)",
        height=max(500, 330 * len(source_order)),
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
.comment-word-controls { display: grid; grid-template-columns: minmax(180px, 260px) minmax(220px, 1fr);
                         gap: 10px; margin: 8px 0 10px; align-items: end; }
.comment-word-controls label { display: block; font-size: 0.9em; color: #555; margin-bottom: 4px; }
.comment-word-controls select { width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 6px; }
.comment-cloud { border: 1px solid #ddd; border-radius: 8px; background: #f8fafc;
                 min-height: 84px; padding: 12px; line-height: 2.1; }
.comment-cloud-empty { color: #6b7280; font-size: 0.95em; }
.comment-box { border: 1px solid #ddd; border-radius: 8px; background: #fafafa;
               min-height: 70px; padding: 12px; white-space: pre-wrap; }
.comment-status { font-size: 0.9em; color: #4b5563; margin: 6px 0 10px; }
"""


def _df_to_html(df: pd.DataFrame, float_fmt: str = "%.3f") -> str:
    return df.to_html(index=False, float_format=float_fmt, na_rep="—")


def _comment_explorer_html(bdf: pd.DataFrame) -> str:
    cols = [
        "model_display", "run", "source", "article_id", "comment", "subject_bias", "framing_bias",
        "treatment_bias", "guests_bias", "overall_bias", "confidence", "status",
    ]
    data = _with_model_metadata(bdf)
    for col in cols:
        if col not in data.columns:
            data[col] = ""
    data = data[cols].copy().fillna("")
    records = data.to_dict(orient="records")
    records_json = json.dumps(records, ensure_ascii=True).replace("</", "<\\/")

    return f"""
<div id=\"comment-explorer\">
  <div class=\"comment-controls\">
    <div>
      <label for=\"model-select\">Model</label>
      <select id=\"model-select\"></select>
    </div>
    <div>
      <label for=\"source-select\">Source</label>
      <select id=\"source-select\"></select>
    </div>
    <div>
      <label for=\"article-select\">Article</label>
      <select id=\"article-select\"></select>
    </div>
    <div>
      <label for=\"run-select\">Run (optional)</label>
      <select id=\"run-select\"></select>
    </div>
  </div>
  <div class=\"comment-word-controls\">
    <div>
      <label for=\"cloud-mode-select\">Word cloud grouping</label>
      <select id=\"cloud-mode-select\">
        <option value=\"source\">Per source</option>
        <option value=\"model\">Per model</option>
        <option value=\"model_source\">Per model + source</option>
      </select>
    </div>
    <div>
      <label for=\"cloud-group-select\">Word cloud selection</label>
      <select id=\"cloud-group-select\"></select>
    </div>
  </div>
  <div id=\"comment-cloud\" class=\"comment-cloud\"><span class=\"comment-cloud-empty\">Loading word cloud...</span></div>
  <div id=\"comment-status\" class=\"comment-status\">Loading explorer...</div>
  <div id=\"comment-box\" class=\"comment-box\">Select any filter(s): model, source, article, run.</div>
</div>
<script id=\"comment-data\" type=\"application/json\">{records_json}</script>
<script>
(function() {{
  const dataEl = document.getElementById('comment-data');
  const rows = JSON.parse((dataEl && dataEl.textContent) ? dataEl.textContent : '[]');
  const modelSel = document.getElementById('model-select');
  const sourceSel = document.getElementById('source-select');
  const runSel = document.getElementById('run-select');
  const articleSel = document.getElementById('article-select');
  const cloudModeSel = document.getElementById('cloud-mode-select');
  const cloudGroupSel = document.getElementById('cloud-group-select');
  const cloudEl = document.getElementById('comment-cloud');
  const statusEl = document.getElementById('comment-status');
  const box = document.getElementById('comment-box');
  const stopwords = new Set({json.dumps(sorted(_COMMENT_STOPWORDS), ensure_ascii=False)});
  const descriptiveSuffixes = {json.dumps(sorted(_COMMENT_DESCRIPTIVE_SUFFIXES), ensure_ascii=False)};
  const descriptiveWhitelist = new Set({json.dumps(sorted(_COMMENT_DESCRIPTIVE_WHITELIST), ensure_ascii=False)});
  const tokenSplitRe = /[^A-Za-zÀ-ÖØ-öø-ÿ]+/;
  const cloudColors = ['#0f766e','#1d4ed8','#b45309','#be123c','#4338ca','#166534'];

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

  function labelSource(v) {{
    const map = {{
      rts: 'RTS',
      jacobin: 'Jacobin',
      the_federalist: 'The Federalist',
      watson: 'Watson',
      protestinfo: 'Protestinfo',
      cathinfo: 'Cathinfo',
      srf: 'SRF',
      '20minutes': '20 Minutes'
    }};
    return map[v] || v;
  }}

  function tokenize(text) {{
    return String(text || '')
      .toLowerCase()
      .split(tokenSplitRe)
      .filter(t =>
        t.length >= 4 &&
        !stopwords.has(t) &&
        !/^\\d+$/.test(t) &&
        !t.startsWith('http') &&
        (descriptiveWhitelist.has(t) || descriptiveSuffixes.some(s => t.endsWith(s)))
      );
  }}

  function cloudKey(row, mode) {{
    if (mode === 'model') return String(row.model_display);
    if (mode === 'model_source') return String(row.model_display) + ' | ' + labelSource(String(row.source));
    return labelSource(String(row.source));
  }}

  function filterRows() {{
    const m = selectedValue(modelSel);
    const s = selectedValue(sourceSel);
    const r = selectedValue(runSel);
    const a = selectedValue(articleSel);
    return rows.filter(x =>
      (!m || String(x.model_display) === m) &&
      (!s || String(x.source) === s) &&
      (!r || String(x.run) === r) &&
      (!a || String(x.article_id) === a)
    );
  }}

  function refreshRunOptionsFromBase() {{
    const m = selectedValue(modelSel);
    const s = selectedValue(sourceSel);
    const filtered = rows.filter(x =>
      (!m || String(x.model_display) === m) &&
      (!s || String(x.source) === s)
    );
    const current = runSel.value;
    setOptions(runSel, uniq(filtered.map(x => String(x.run))));
    if (Array.from(runSel.options).some(o => o.value === current)) runSel.value = current;
  }}

  function refreshArticleOptions() {{
    const m = selectedValue(modelSel);
    const s = selectedValue(sourceSel);
    const r = selectedValue(runSel);
    const filtered = rows.filter(x =>
      (!m || String(x.model_display) === m) &&
      (!s || String(x.source) === s) &&
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
      return `model=${{x.model_display}} | run=${{x.run}} | source=${{x.source}} | article=${{x.article_id}}\n` +
             `status=${{x.status}} | conf=${{x.confidence}} | overall=${{x.overall_bias}}\n` +
             `subject=${{x.subject_bias}}, framing=${{x.framing_bias}}, treatment=${{x.treatment_bias}}, guests=${{x.guests_bias}}\n` +
             `comment: ${{c}}`;
    }});

    const suffix = filtered.length > 10 ? `\n\nShowing 10 of ${{filtered.length}} matches.` : '';
    box.textContent = lines.join('\n\n---\n\n') + suffix;
  }}

  function renderCloudWords(wordCounts) {{
    cloudEl.innerHTML = '';
    const entries = Array.from(wordCounts.entries()).sort((a, b) => b[1] - a[1]).slice(0, 80);
    if (!entries.length) {{
      const empty = document.createElement('span');
      empty.className = 'comment-cloud-empty';
      empty.textContent = 'No words found for this selection.';
      cloudEl.appendChild(empty);
      return;
    }}
    const maxCount = entries[0][1];
    const minCount = entries[entries.length - 1][1];
    const spread = Math.max(1, maxCount - minCount);
    for (const [word, count] of entries) {{
      const norm = (count - minCount) / spread;
      const size = Math.round(13 + norm * 28);
      const span = document.createElement('span');
      span.textContent = word;
      span.title = 'count=' + count;
      span.style.display = 'inline-block';
      span.style.margin = '0 8px 6px 0';
      span.style.fontWeight = '600';
      span.style.fontSize = size + 'px';
      span.style.color = cloudColors[word.length % cloudColors.length];
      cloudEl.appendChild(span);
    }}
  }}

  function updateWordCloud() {{
    const filtered = filterRows();
    const mode = cloudModeSel.value;
    const keys = uniq(filtered.map(x => cloudKey(x, mode)));
    const current = cloudGroupSel.value;
    setOptions(cloudGroupSel, keys);
    if (Array.from(cloudGroupSel.options).some(o => o.value === current)) {{
      cloudGroupSel.value = current;
    }} else if (cloudGroupSel.options.length > 1) {{
      cloudGroupSel.value = cloudGroupSel.options[1].value;
    }}
    const selected = selectedValue(cloudGroupSel);
    if (!selected) {{
      renderCloudWords(new Map());
      return;
    }}
    const counts = new Map();
    for (const row of filtered) {{
      if (cloudKey(row, mode) !== selected) continue;
      for (const tok of tokenize(row.comment)) {{
        counts.set(tok, (counts.get(tok) || 0) + 1);
      }}
    }}
    renderCloudWords(counts);
  }}

  setOptions(modelSel, uniq(rows.map(x => String(x.model_display))));
  setOptions(sourceSel, uniq(rows.map(x => String(x.source))));
  refreshRunOptionsFromBase();
  refreshArticleOptions();
  updateWordCloud();
  updateCommentBox();
  statusEl.textContent = `Explorer loaded: rows=${{rows.length}}, models=${{modelSel.options.length - 1}}, sources=${{sourceSel.options.length - 1}}, articles=${{articleSel.options.length - 1}}`;

  modelSel.addEventListener('change', () => {{
    refreshRunOptionsFromBase();
    refreshArticleOptions();
    updateWordCloud();
    updateCommentBox();
  }});
  sourceSel.addEventListener('change', () => {{
    refreshRunOptionsFromBase();
    refreshArticleOptions();
    updateWordCloud();
    updateCommentBox();
  }});
  runSel.addEventListener('change', () => {{
    refreshArticleOptions();
    updateWordCloud();
    updateCommentBox();
  }});
  articleSel.addEventListener('change', () => {{
    updateWordCloud();
    updateCommentBox();
  }});
  cloudModeSel.addEventListener('change', updateWordCloud);
  cloudGroupSel.addEventListener('change', updateWordCloud);
}})();
</script>
"""


def build_comment_explorer_page(
    bias_df: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "polibias — Comment Explorer",
) -> None:
    cols = [
        "model_display", "run", "source", "article_id", "status", "confidence", "overall_bias",
        "subject_bias", "framing_bias", "treatment_bias", "guests_bias", "comment",
    ]
    data = _with_model_metadata(bias_df)
    for col in cols:
        if col not in data.columns:
            data[col] = ""
    data = data[cols].fillna("")

    def _fmt(value: object) -> str:
        s = str(value)
        return "" if s == "nan" else s

    rows_html: list[str] = []
    for row in data.to_dict(orient="records"):
        model = _fmt(row["model_display"])
        run = _fmt(row["run"])
        source = _fmt(row["source"])
        article_id = _fmt(row["article_id"])
        status = _fmt(row["status"])
        confidence = _fmt(row["confidence"])
        overall = _fmt(row["overall_bias"])
        subject = _fmt(row["subject_bias"])
        framing = _fmt(row["framing_bias"])
        treatment = _fmt(row["treatment_bias"])
        guests = _fmt(row["guests_bias"])
        comment = _fmt(row["comment"])
        rows_html.append(
            "<tr "
            f"data-model=\"{html.escape(model, quote=True)}\" "
            f"data-source=\"{html.escape(source, quote=True)}\" "
            f"data-article=\"{html.escape(article_id, quote=True)}\" "
            f"data-run=\"{html.escape(run, quote=True)}\" "
            f"data-status=\"{html.escape(status, quote=True)}\""
            ">"
            f"<td>{html.escape(model)}</td>"
            f"<td>{html.escape(source)}</td>"
            f"<td>{html.escape(article_id)}</td>"
            f"<td>{html.escape(run)}</td>"
            f"<td>{html.escape(status)}</td>"
            f"<td>{html.escape(confidence)}</td>"
            f"<td>{html.escape(overall)}</td>"
            f"<td>{html.escape(subject)}</td>"
            f"<td>{html.escape(framing)}</td>"
            f"<td>{html.escape(treatment)}</td>"
            f"<td>{html.escape(guests)}</td>"
            f"<td>{html.escape(comment)}</td>"
            "</tr>"
        )

    page = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 1600px; margin: auto; padding: 20px; color: #222; }}
h1 {{ margin-bottom: 8px; }}
.note {{ color: #555; margin: 0 0 16px; }}
.controls {{ display: grid; grid-template-columns: repeat(6, minmax(140px, 1fr)); gap: 10px; margin: 12px 0 16px; }}
.controls label {{ display: block; font-size: 0.9em; color: #444; margin-bottom: 3px; }}
.controls select, .controls input {{ width: 100%; padding: 7px; border: 1px solid #ccc; border-radius: 6px; }}
.stats {{ margin: 8px 0 12px; color: #333; }}
.cloud-controls {{ display: grid; grid-template-columns: minmax(180px, 260px) minmax(220px, 1fr); gap: 10px; margin: 8px 0 10px; }}
.cloud-controls label {{ display: block; font-size: 0.9em; color: #444; margin-bottom: 3px; }}
.cloud-controls select {{ width: 100%; padding: 7px; border: 1px solid #ccc; border-radius: 6px; }}
.word-cloud {{ border: 1px solid #ddd; border-radius: 8px; background: #f8fafc; min-height: 84px; padding: 12px; line-height: 2.1; margin-bottom: 12px; }}
.word-cloud-empty {{ color: #6b7280; font-size: 0.95em; }}
table {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
th, td {{ border: 1px solid #ddd; padding: 7px; vertical-align: top; font-size: 13px; }}
th {{ background: #f7f7f7; position: sticky; top: 0; z-index: 1; }}
td:nth-child(12), th:nth-child(12) {{ width: 38%; }}
td:nth-child(12) {{ white-space: pre-wrap; word-break: break-word; }}
tbody tr:nth-child(even) {{ background: #fafafa; }}
</style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p class="note">Standalone comment explorer. Filter by any combination of model/source/article/run/status and text search.</p>
  <div class="controls">
    <div><label for="f-model">Model</label><select id="f-model"></select></div>
    <div><label for="f-source">Source</label><select id="f-source"></select></div>
    <div><label for="f-article">Article</label><select id="f-article"></select></div>
    <div><label for="f-run">Run</label><select id="f-run"></select></div>
    <div><label for="f-status">Status</label><select id="f-status"></select></div>
    <div><label for="f-text">Search comment</label><input id="f-text" type="text" placeholder="contains text"></div>
  </div>
  <div id="stats" class="stats"></div>
  <div class="cloud-controls">
    <div><label for="f-cloud-mode">Word cloud grouping</label><select id="f-cloud-mode"><option value="source">Per source</option><option value="model">Per model</option><option value="model_source">Per model + source</option></select></div>
    <div><label for="f-cloud-group">Word cloud selection</label><select id="f-cloud-group"></select></div>
  </div>
  <div id="word-cloud" class="word-cloud"><span class="word-cloud-empty">Loading word cloud...</span></div>
  <table>
    <thead>
      <tr>
        <th>model</th><th>source</th><th>article_id</th><th>run</th><th>status</th>
        <th>confidence</th><th>overall</th><th>subject</th><th>framing</th><th>treatment</th><th>guests</th><th>comment</th>
      </tr>
    </thead>
    <tbody id="rows">
      {''.join(rows_html)}
    </tbody>
  </table>
<script>
(function() {{
  const rows = Array.from(document.querySelectorAll('#rows tr'));
  const stats = document.getElementById('stats');
  const model = document.getElementById('f-model');
  const source = document.getElementById('f-source');
  const article = document.getElementById('f-article');
  const run = document.getElementById('f-run');
  const status = document.getElementById('f-status');
  const text = document.getElementById('f-text');
  const cloudMode = document.getElementById('f-cloud-mode');
  const cloudGroup = document.getElementById('f-cloud-group');
  const cloudEl = document.getElementById('word-cloud');
  const stopwords = new Set({json.dumps(sorted(_COMMENT_STOPWORDS), ensure_ascii=False)});
  const descriptiveSuffixes = {json.dumps(sorted(_COMMENT_DESCRIPTIVE_SUFFIXES), ensure_ascii=False)};
  const descriptiveWhitelist = new Set({json.dumps(sorted(_COMMENT_DESCRIPTIVE_WHITELIST), ensure_ascii=False)});
  const tokenSplitRe = /[^A-Za-zÀ-ÖØ-öø-ÿ]+/;
  const cloudColors = ['#0f766e','#1d4ed8','#b45309','#be123c','#4338ca','#166534'];

  function uniq(values) {{
    return Array.from(new Set(values.filter(v => v !== ''))).sort();
  }}
  function setOptions(sel, vals) {{
    sel.innerHTML = '';
    const all = document.createElement('option');
    all.value = '__ALL__';
    all.textContent = 'All';
    sel.appendChild(all);
    for (const v of vals) {{
      const o = document.createElement('option');
      o.value = v;
      o.textContent = v;
      sel.appendChild(o);
    }}
  }}
  function selected(sel) {{ return sel.value === '__ALL__' ? '' : sel.value; }}
  function labelSource(v) {{
    const map = {{
      rts: 'RTS',
      jacobin: 'Jacobin',
      the_federalist: 'The Federalist',
      watson: 'Watson',
      protestinfo: 'Protestinfo',
      cathinfo: 'Cathinfo',
      srf: 'SRF',
      '20minutes': '20 Minutes'
    }};
    return map[v] || v;
  }}
  function tokenize(textValue) {{
    return String(textValue || '')
      .toLowerCase()
      .split(tokenSplitRe)
      .filter(t =>
        t.length >= 4 &&
        !stopwords.has(t) &&
        !/^\\d+$/.test(t) &&
        !t.startsWith('http') &&
        (descriptiveWhitelist.has(t) || descriptiveSuffixes.some(s => t.endsWith(s)))
      );
  }}
  function cloudKey(tr, mode) {{
    if (mode === 'model') return tr.dataset.model;
    if (mode === 'model_source') return tr.dataset.model + ' | ' + labelSource(tr.dataset.source);
    return labelSource(tr.dataset.source);
  }}
  function visibleRows() {{
    return rows.filter(tr => tr.style.display !== 'none');
  }}
  function renderCloudWords(wordCounts) {{
    cloudEl.innerHTML = '';
    const entries = Array.from(wordCounts.entries()).sort((a, b) => b[1] - a[1]).slice(0, 80);
    if (!entries.length) {{
      const empty = document.createElement('span');
      empty.className = 'word-cloud-empty';
      empty.textContent = 'No words found for this selection.';
      cloudEl.appendChild(empty);
      return;
    }}
    const maxCount = entries[0][1];
    const minCount = entries[entries.length - 1][1];
    const spread = Math.max(1, maxCount - minCount);
    for (const [word, count] of entries) {{
      const norm = (count - minCount) / spread;
      const size = Math.round(13 + norm * 28);
      const span = document.createElement('span');
      span.textContent = word;
      span.title = 'count=' + count;
      span.style.display = 'inline-block';
      span.style.margin = '0 8px 6px 0';
      span.style.fontWeight = '600';
      span.style.fontSize = size + 'px';
      span.style.color = cloudColors[word.length % cloudColors.length];
      cloudEl.appendChild(span);
    }}
  }}
  function updateWordCloud() {{
    const shown = visibleRows();
    const mode = cloudMode.value;
    const keys = uniq(shown.map(tr => cloudKey(tr, mode)));
    const current = cloudGroup.value;
    setOptions(cloudGroup, keys);
    if (Array.from(cloudGroup.options).some(o => o.value === current)) {{
      cloudGroup.value = current;
    }} else if (cloudGroup.options.length > 1) {{
      cloudGroup.value = cloudGroup.options[1].value;
    }}
    const selectedGroup = selected(cloudGroup);
    if (!selectedGroup) {{
      renderCloudWords(new Map());
      return;
    }}
    const counts = new Map();
    for (const tr of shown) {{
      if (cloudKey(tr, mode) !== selectedGroup) continue;
      const textCell = tr.lastElementChild ? tr.lastElementChild.textContent : '';
      for (const tok of tokenize(textCell)) {{
        counts.set(tok, (counts.get(tok) || 0) + 1);
      }}
    }}
    renderCloudWords(counts);
  }}
  function refreshOptions() {{
    const m = selected(model), s = selected(source), r = selected(run), st = selected(status);
    const base = rows.filter(tr =>
      (!m || tr.dataset.model === m) &&
      (!s || tr.dataset.source === s) &&
      (!r || tr.dataset.run === r) &&
      (!st || tr.dataset.status === st)
    );
    const current = article.value;
    setOptions(article, uniq(base.map(tr => tr.dataset.article)));
    if (Array.from(article.options).some(o => o.value === current)) article.value = current;
  }}
  function apply() {{
    const m = selected(model), s = selected(source), a = selected(article), r = selected(run), st = selected(status);
    const q = text.value.trim().toLowerCase();
    let shown = 0;
    for (const tr of rows) {{
      const ok = (!m || tr.dataset.model === m) &&
                 (!s || tr.dataset.source === s) &&
                 (!a || tr.dataset.article === a) &&
                 (!r || tr.dataset.run === r) &&
                 (!st || tr.dataset.status === st) &&
                 (!q || tr.lastElementChild.textContent.toLowerCase().includes(q));
      tr.style.display = ok ? '' : 'none';
      if (ok) shown += 1;
    }}
    stats.textContent = `Showing ${{shown}} / ${{rows.length}} rows`;
    updateWordCloud();
  }}

  setOptions(model, uniq(rows.map(tr => tr.dataset.model)));
  setOptions(source, uniq(rows.map(tr => tr.dataset.source)));
  setOptions(run, uniq(rows.map(tr => tr.dataset.run)));
  setOptions(status, uniq(rows.map(tr => tr.dataset.status)));
  setOptions(article, uniq(rows.map(tr => tr.dataset.article)));
  refreshOptions();
  apply();

  for (const el of [model, source, article, run, status]) {{
    el.addEventListener('change', () => {{ refreshOptions(); apply(); }});
  }}
  text.addEventListener('input', apply);
  cloudMode.addEventListener('change', updateWordCloud);
  cloudGroup.addEventListener('change', updateWordCloud);
}})();
</script>
</body>
</html>"""
    Path(output_path).write_text(page, encoding="utf-8")


# ---------- Main report builder ----------

def build_html_report(
    bias_df: pd.DataFrame,
    output_path: str,
    kappa_bins: int = 5,
    *,
    title: str = "polibias — Bias Analysis Report",
    include_tables: bool = True,
    include_comment_explorer: bool = False,
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
            summary["cohort"] = summary["model"].map(model_cohort)
            summary["model"] = summary["model"].map(model_display_name)
            display_cols = ["model", "cohort"]
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
            model_ci = report["model_ci"].copy()
            model_ci["model"] = model_ci["model"].map(model_display_name)
            html_parts.append(_df_to_html(model_ci))

        # --- ICC table ---
        html_parts.append("<h2>Within-model consistency (ICC)</h2>")
        html_parts.append('<p class="section-note">'
                          'ICC(1,1) measures how consistently a model scores the same article '
                          'across runs. 0 = random, 1 = perfect. Below 0.4 is poor.</p>')
        if not report["icc_per_model"].empty:
            icc_per_model = report["icc_per_model"].copy()
            icc_per_model["model"] = icc_per_model["model"].map(model_display_name)
            html_parts.append(_df_to_html(icc_per_model))

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

    if include_comment_explorer:
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
    report_df = _with_model_metadata(bias_df)
    summary = report_df.groupby("model_display").agg(
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
            f"{r['model_display']} & {r['n']:.0f} & {r['subject']:.3f} & {r['framing']:.3f} "
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
    run: int | None = None,
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
    if run is not None:
        bias_df = bias_df[bias_df["run"] == run].copy()
    if bias_df.empty:
        print("  No bias data found. Run 'score' and 'analyse' first.")
        return

    # HTML report
    report_path = settings.run_dir / output_filename if output_filename else settings.report_html_path
    html_path = str(report_path)
    if source is None:
        title = "polibias — Bias Analysis Report"
    else:
        title = f"polibias — {source} Bias Report"
    if run is not None:
        title = f"{title} (run {run})"
    build_html_report(
        bias_df,
        html_path,
        kappa_bins=settings.kappa_bins,
        title=title,
        include_tables=include_tables,
        include_comment_explorer=False,
    )
    print(f"  Wrote HTML report: {html_path}")

    # Dedicated comment explorer (no Plotly dependency)
    report_stem = Path(output_filename).stem if output_filename else "report"
    comments_filename = f"{report_stem}_comments.html"
    comments_path = settings.run_dir / comments_filename
    build_comment_explorer_page(
        bias_df,
        comments_path,
        title=f"{title} — Comment Explorer",
    )
    print(f"  Wrote comment explorer: {comments_path}")

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
    run: int | None = None,
    output_filename: str = "report_all.html",
    source_reports: Mapping[str, str] | None = None,
) -> None:
    from polibias.analysis import build_bias_frame

    bias_df = build_bias_frame(settings)
    if run is not None:
        bias_df = bias_df[bias_df["run"] == run].copy()
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
    by_source_model["cohort"] = by_source_model["model"].map(model_cohort)
    by_source_model["model"] = by_source_model["model"].map(model_display_name)
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
    source_model_heat = (
        _with_model_metadata(bias_df).groupby(["source", "model_display"], as_index=False)["overall_bias"]
        .mean()
        .pivot(index="source", columns="model_display", values="overall_bias")
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
        "<h1>polibias — Cross-source report</h1>" if run is None else f"<h1>polibias — Cross-source report (run {run})</h1>",
        '<p class="section-note">Cross-reference view across all sources in this run.</p>',
        links_html,
        "<h2>Source summary</h2>",
        _df_to_html(by_source_dim),
        "<h2>Source x model summary</h2>",
        _df_to_html(by_source_model),
        "<h2>Heatmap</h2>",
        fig.to_html(full_html=False, include_plotlyjs="cdn"),
        "<h2>Comment explorer</h2>",
        '<p class="section-note">Use model/source/article/run selectors to inspect original JSON comments and scores.</p>',
        _comment_explorer_html(bias_df),
        "</body></html>",
    ]

    out_path = settings.run_dir / output_filename
    out_path.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"  Wrote cross-source HTML report: {out_path}")
