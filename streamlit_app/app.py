"""polibias — Bias Analysis Dashboard.

Standalone Streamlit app. Reads:
    data/runs/comparisons/bias_data.csv
    data/runs/comparisons/article_summaries.csv

No pipeline dependencies — just streamlit, pandas, plotly, numpy.
"""

from __future__ import annotations

import re
import textwrap
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
BIAS_CSV = REPO_ROOT / "data" / "runs" / "comparisons" / "bias_data.csv"
SUMMARIES_CSV = REPO_ROOT / "data" / "runs" / "comparisons" / "article_summaries.csv"
PROMPT_MD = REPO_ROOT / "src" / "polibias" / "prompt.md"
LIVE_APP_URL = "https://politicalbiaswithai.streamlit.app/"

SOURCE_LABELS: dict[str, str] = {
    "rts": "RTS",
    "jacobin": "Jacobin",
    "the_federalist": "The Federalist",
    "watson": "Watson",
    "protestinfo": "Protestinfo",
    "cathinfo": "Cathinfo",
    "srf": "SRF",
    "20minutes": "20 Minutes",
}

BIAS_DIMS = ["subject_bias", "framing_bias", "treatment_bias", "guests_bias"]
DIM_LABELS = {
    "subject_bias": "Subject",
    "framing_bias": "Framing",
    "treatment_bias": "Treatment",
    "guests_bias": "Guests",
}

PALETTE = px.colors.qualitative.Plotly
_HOVER_STYLE = dict(
    bgcolor="#111827",
    bordercolor="#374151",
    font=dict(color="#F9FAFB", size=12),
    align="left",
)

TOKEN_SPLIT_RE = re.compile(r"[^A-Za-zÀ-ÖØ-öø-ÿ]+")
ONE_OFF_MODELS = {"claude-sonnet-4-6", "codex", "gemini3"}
COMMENT_STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "have", "has", "was", "were", "are",
    "but", "not", "its", "their", "they", "them", "into", "about", "than", "also", "only",
    "very", "more", "most", "much", "many", "such", "can", "could", "would", "should", "there",
    "here", "when", "where", "what", "which", "while", "then", "than", "without", "between",
    "dans", "avec", "pour", "une", "des", "les", "est", "sur", "pas", "plus", "comme", "mais",
    "cette", "cet", "ces", "aux", "par", "qui", "que", "dans", "sans", "tout", "tous", "elle",
    "elles", "nous", "vous", "ils", "leur", "leurs", "sein", "dont", "afin",
    "der", "die", "das", "und", "ist", "mit", "nicht", "eine", "einer", "einem", "einen",
    "den", "dem", "des", "ein", "auch", "als", "auf", "für", "von", "bei", "aus", "über",
    "durch", "wird", "sind", "war", "waren", "dass", "oder", "zum", "zur", "im", "am",
    "article", "articles", "comment", "comments", "report", "reports", "reporting", "bias",
    "biased", "political", "politically", "politique", "politiques", "model", "models",
    "source", "sources", "left", "right", "leaning", "neutral", "identifiable", "clear",
    "clearly", "appears", "overall", "focus", "focuses", "focused", "cover", "covers",
    "covered", "coverage", "discusses", "describes", "described", "presents", "presented",
    "article", "larticle", "factuel", "factual", "factuelle", "factually",
}
DESCRIPTIVE_SUFFIXES = (
    "able", "ible", "al", "ant", "ary", "ative", "ed", "ent", "ful", "ial", "ible", "ic",
    "ical", "ish", "ive", "less", "ory", "ous", "y", "aire", "ant", "euse", "eux", "if",
    "ifs", "ive", "ives", "ique", "iques",
)
DESCRIPTIVE_WHITELIST = {
    "alarmist", "balanced", "balance", "critical", "cautious", "controversial", "diplomatic",
    "ethical", "favorable", "favourable", "hostile", "humanitarian", "negative", "positive",
    "optimistic", "pessimistic", "sympathetic", "technical", "tense", "urgent", "neutre",
    "neutres", "negative", "negatif", "negative", "positif", "positive", "positives",
    "critique", "critiques", "equilibre", "equilibree", "equilibrees", "prudent",
    "prudente", "prudentes", "favorable", "favorables", "hostile", "humanitaire",
    "humanitaires", "alarmiste", "alarmistes", "technique", "techniques", "tendu", "tendue",
}


# ── data loading ───────────────────────────────────────────────────────────────

@st.cache_data
def load_bias() -> pd.DataFrame:
    df = pd.read_csv(BIAS_CSV)
    if "model_cohort" not in df.columns:
        df["model_cohort"] = df["model"].map(lambda m: "one-off" if str(m) in ONE_OFF_MODELS else "baseline")
    if "model_display" not in df.columns:
        df["model_display"] = df["model"].map(
            lambda m: f"{m} [one-off]" if str(m) in ONE_OFF_MODELS else str(m)
        )
    df["source_label"] = df["source"].map(SOURCE_LABELS).fillna(df["source"])
    df["comment"] = df["comment"].fillna("").astype(str)
    for col in [*BIAS_DIMS, "overall_bias", "confidence"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data
def load_summaries() -> pd.DataFrame:
    return pd.read_csv(SUMMARIES_CSV)


@st.cache_data
def load_prompt_template() -> str:
    return PROMPT_MD.read_text(encoding="utf-8").strip()


# ── helpers ────────────────────────────────────────────────────────────────────

def _wrap_hover(text: str, width: int = 80) -> str:
    if not text.strip():
        return "(no comment)"
    parts: list[str] = []
    for line in text.splitlines():
        wrapped = textwrap.wrap(line, width=width, break_long_words=False)
        parts.extend(wrapped or [""])
    return "<br>".join(parts)


def _style(fig: go.Figure) -> go.Figure:
    fig.update_layout(hoverlabel=_HOVER_STYLE)
    return fig


def _tokenize_comment(text: str) -> list[str]:
    tokens = [t for t in TOKEN_SPLIT_RE.split(text.lower()) if len(t) >= 4]
    filtered = [
        t for t in tokens
        if t not in COMMENT_STOPWORDS
        and not t.isdigit()
        and not t.startswith("http")
        and (t in DESCRIPTIVE_WHITELIST or t.endswith(DESCRIPTIVE_SUFFIXES))
    ]
    return filtered


def _render_word_cloud(counts: Counter[str], max_words: int = 80) -> str:
    if not counts:
        return (
            "<div style='padding:10px;border:1px solid #e5e7eb;border-radius:8px;color:#6b7280;'>"
            "No words found for this selection."
            "</div>"
        )
    top = counts.most_common(max_words)
    max_count = top[0][1]
    min_count = top[-1][1]
    spread = max(1, max_count - min_count)
    colors = ["#0f766e", "#1d4ed8", "#b45309", "#be123c", "#4338ca", "#166534"]

    parts = [
        "<div style='padding:12px;border:1px solid #e5e7eb;border-radius:8px;"
        "background:#f8fafc;line-height:2.2;'>"
    ]
    for word, count in top:
        norm = (count - min_count) / spread
        size = int(13 + (norm * 28))
        color = colors[hash(word) % len(colors)]
        parts.append(
            f"<span title='count={count}' "
            f"style='display:inline-block;margin:0 8px 6px 0;"
            f"font-size:{size}px;font-weight:600;color:{color};'>{word}</span>"
        )
    parts.append("</div>")
    return "".join(parts)


def _compute_ci(df: pd.DataFrame, col: str = "overall_bias") -> pd.DataFrame:
    """Mean ± 95% CI per model (z-approximation, fine for n ≥ 30)."""
    rows = []
    for model, grp in df.groupby("model_display"):
        vals = grp[col].dropna()
        n = len(vals)
        if n < 2:
            continue
        mean = float(vals.mean())
        sem = float(vals.std(ddof=1) / (n ** 0.5))
        rows.append({
            "model": model,
            "mean": mean,
            "ci_low": mean - 1.96 * sem,
            "ci_high": mean + 1.96 * sem,
            "n": n,
        })
    return pd.DataFrame(rows).sort_values("mean")


# ── charts ─────────────────────────────────────────────────────────────────────

def _chart_heatmap(df: pd.DataFrame) -> go.Figure:
    pivot = (
        df.groupby(["source_label", "model"])["overall_bias"]
        .mean()
        .reset_index()
        .pivot(index="source_label", columns="model", values="overall_bias")
    )
    fig = px.imshow(
        pivot,
        color_continuous_scale="RdBu",
        zmin=-1, zmax=1,
        aspect="auto",
        title="Mean overall bias — source × model",
        text_auto=".2f",
    )
    fig.update_layout(
        xaxis_title="Model",
        yaxis_title="Source",
        coloraxis_colorbar_title="Bias",
        height=380,
    )
    return _style(fig)


def _chart_ci_forest(ci_df: pd.DataFrame) -> go.Figure:
    if ci_df.empty:
        return go.Figure()
    fig = go.Figure()
    for _, row in ci_df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["ci_low"], row["ci_high"]],
            y=[row["model"], row["model"]],
            mode="lines",
            line=dict(width=4),
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[row["mean"]],
            y=[row["model"]],
            mode="markers",
            marker=dict(size=11, symbol="diamond"),
            showlegend=False,
            hovertemplate=(
                f"<b>{row['model']}</b><br>"
                f"mean = {row['mean']:.3f}<br>"
                f"95% CI [{row['ci_low']:.3f}, {row['ci_high']:.3f}]<br>"
                f"n = {row['n']}<extra></extra>"
            ),
        ))
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Mean overall bias with 95% CI per model",
        xaxis=dict(title="overall_bias", range=[-1, 1]),
        yaxis_title="",
        height=max(320, 52 * len(ci_df)),
    )
    return _style(fig)


def _chart_scatter_by_model(df: pd.DataFrame) -> go.Figure:
    d = df.copy()
    d["comment_hover"] = d["comment"].map(_wrap_hover)
    fig = px.scatter(
        d,
        x="model_display",
        y="overall_bias",
        color="source_label",
        custom_data=[
            "run", "article_id", "confidence",
            "subject_bias", "framing_bias", "treatment_bias", "guests_bias",
            "comment_hover",
        ],
        labels={"model_display": "Model", "overall_bias": "Overall bias", "source_label": "Source"},
        title="Overall bias by model",
    )
    fig.update_yaxes(range=[-1.2, 1.2])
    fig.update_traces(
        hovertemplate=(
            "<b>Model:</b> %{x}<br>"
            "<b>Bias:</b> %{y:.3f}<br>"
            "<b>Source:</b> %{fullData.name}<br>"
            "<b>Run:</b> %{customdata[0]}  |  "
            "<b>Article:</b> %{customdata[1]}<br>"
            "<b>Confidence:</b> %{customdata[2]:.3f}<br>"
            "<b>Sub:</b> S=%{customdata[3]:.2f} "
            "F=%{customdata[4]:.2f} "
            "T=%{customdata[5]:.2f} "
            "G=%{customdata[6]:.2f}<br>"
            "<b>Comment:</b><br>%{customdata[7]}<extra></extra>"
        )
    )
    fig.update_layout(legend_title_text="Source", height=520)
    return _style(fig)


def _chart_box_by_model(df: pd.DataFrame) -> go.Figure:
    fig = px.box(
        df,
        x="model_display", y="overall_bias",
        color="model_display", points="all",
        title="Bias distribution by model",
    )
    fig.update_yaxes(range=[-1.2, 1.2])
    fig.update_layout(showlegend=False, height=460)
    return _style(fig)


def _chart_subbias_bar(df: pd.DataFrame) -> go.Figure:
    rows = []
    for model, grp in df.groupby("model_display"):
        for dim in BIAS_DIMS:
            rows.append({
                "model": model,
                "dimension": DIM_LABELS[dim],
                "mean_bias": float(grp[dim].mean()),
            })
    long = pd.DataFrame(rows)
    fig = px.bar(
        long,
        x="model", y="mean_bias", color="dimension",
        barmode="group",
        title="Mean sub-bias by model and dimension",
        labels={"mean_bias": "Mean bias", "model": "Model", "dimension": "Dimension"},
    )
    fig.update_yaxes(range=[-1, 1])
    fig.update_layout(height=460)
    return _style(fig)


def _chart_source_dots(df: pd.DataFrame, source: str) -> go.Figure:
    src_df = df[df["source"] == source].copy()
    if src_df.empty:
        return go.Figure()
    src_df["comment_hover"] = src_df["comment"].map(_wrap_hover)
    model_order = sorted(src_df["model_display"].unique())
    color_map = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(model_order)}
    article_order = sorted(src_df["article_id"].unique())

    fig = go.Figure()
    for model in model_order:
        mdf = src_df[src_df["model_display"] == model]
        custom = np.column_stack([
            mdf["run"].astype(str),
            mdf["confidence"].fillna(0).astype(float),
            mdf["subject_bias"].fillna(0).astype(float),
            mdf["framing_bias"].fillna(0).astype(float),
            mdf["treatment_bias"].fillna(0).astype(float),
            mdf["guests_bias"].fillna(0).astype(float),
            mdf["comment_hover"].astype(str),
        ])
        fig.add_trace(go.Scatter(
            x=mdf["article_id"].astype(str),
            y=mdf["overall_bias"],
            mode="markers",
            name=model,
            marker=dict(color=color_map[model], size=9),
            customdata=custom,
            hovertemplate=(
                "<b>Article:</b> %{x}<br>"
                "<b>Bias:</b> %{y:.3f}<br>"
                "<b>Run:</b> %{customdata[0]}<br>"
                "<b>Confidence:</b> %{customdata[1]:.3f}<br>"
                "<b>Sub:</b> S=%{customdata[2]:.2f} "
                "F=%{customdata[3]:.2f} T=%{customdata[4]:.2f} G=%{customdata[5]:.2f}<br>"
                "<b>Comment:</b><br>%{customdata[6]}<extra></extra>"
            ),
        ))
    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=article_order,
        tickangle=45,
        title_text="article_id",
    )
    fig.update_yaxes(range=[-1.2, 1.2], title_text="overall_bias")
    fig.update_layout(
        title=f"Bias dots — {SOURCE_LABELS.get(source, source)}",
        height=500,
        legend_title_text="Model",
    )
    return _style(fig)


# ── tab renderers ──────────────────────────────────────────────────────────────

def _tab_overview(df: pd.DataFrame) -> None:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Articles", df["article_id"].nunique())
    m2.metric("Models", df["model"].nunique())
    m3.metric("Sources", df["source"].nunique())
    m4.metric("Runs", df["run"].nunique())
    m5.metric("Scores", len(df))

    st.plotly_chart(_chart_heatmap(df), use_container_width=True)
    st.plotly_chart(_chart_ci_forest(_compute_ci(df)), use_container_width=True)

    st.subheader("Model summary")
    summary = (
        df.groupby(["model_display", "model_cohort"])
        .agg(
            overall_mean=("overall_bias", "mean"),
            overall_std=("overall_bias", "std"),
            subject_mean=("subject_bias", "mean"),
            framing_mean=("framing_bias", "mean"),
            treatment_mean=("treatment_bias", "mean"),
            guests_mean=("guests_bias", "mean"),
            confidence_mean=("confidence", "mean"),
            n=("overall_bias", "count"),
        )
        .reset_index()
        .sort_values("overall_mean")
    )
    for col in summary.select_dtypes("float").columns:
        summary[col] = summary[col].round(3)
    st.dataframe(summary, use_container_width=True, hide_index=True)


def _tab_model_comparison(df: pd.DataFrame) -> None:
    cohort_options = ["All", "baseline", "one-off"]
    sel_cohort = st.segmented_control(
        "Model cohort",
        cohort_options,
        default="All",
        selection_mode="single",
    )
    model_df = df if sel_cohort in (None, "All") else df[df["model_cohort"] == sel_cohort]
    st.plotly_chart(_chart_scatter_by_model(model_df), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(_chart_box_by_model(model_df), use_container_width=True)
    with c2:
        st.plotly_chart(_chart_subbias_bar(model_df), use_container_width=True)


def _tab_source_explorer(df: pd.DataFrame) -> None:
    sources = sorted(df["source"].unique())
    sel = st.selectbox(
        "Source",
        sources,
        format_func=lambda s: SOURCE_LABELS.get(s, s),
    )
    st.plotly_chart(_chart_source_dots(df, sel), use_container_width=True)

    src_df = df[df["source"] == sel]
    st.subheader(f"Article summaries — {SOURCE_LABELS.get(sel, sel)}")
    art_summary = (
        src_df.groupby("article_id")
        .agg(
            overall_mean=("overall_bias", "mean"),
            overall_std=("overall_bias", "std"),
            confidence_mean=("confidence", "mean"),
            n_scores=("overall_bias", "count"),
            n_models=("model", "nunique"),
        )
        .reset_index()
        .sort_values("overall_mean")
    )
    for col in art_summary.select_dtypes("float").columns:
        art_summary[col] = art_summary[col].round(3)
    st.dataframe(art_summary, use_container_width=True, hide_index=True)


def _tab_comment_explorer(df: pd.DataFrame) -> None:
    st.caption("Cascading filters — each selection narrows the next.")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        sources = ["All"] + sorted(df["source"].unique())
        sel_source = st.selectbox(
            "Source",
            sources,
            format_func=lambda s: SOURCE_LABELS.get(s, s) if s != "All" else "All",
        )

    filtered = df if sel_source == "All" else df[df["source"] == sel_source]

    with c2:
        models = ["All"] + sorted(filtered["model_display"].unique())
        sel_model = st.selectbox("Model", models)

    filtered = filtered if sel_model == "All" else filtered[filtered["model_display"] == sel_model]

    with c3:
        articles = ["All"] + sorted(filtered["article_id"].unique())
        sel_article = st.selectbox("Article", articles)

    filtered = filtered if sel_article == "All" else filtered[filtered["article_id"] == sel_article]

    with c4:
        runs = ["All"] + [str(r) for r in sorted(filtered["run"].unique())]
        sel_run = st.selectbox("Run", runs)

    if sel_run != "All":
        filtered = filtered[filtered["run"] == int(sel_run)]

    search = st.text_input("Search comments", placeholder="contains text...")
    if search.strip():
        mask = filtered["comment"].str.lower().str.contains(
            search.strip().lower(), na=False, regex=False
        )
        filtered = filtered[mask]

    st.subheader("Word cloud")
    wc1, wc2 = st.columns([1, 2])
    with wc1:
        group_mode = st.selectbox(
            "Group by",
            ["source", "model", "model+source"],
            index=0,
            key="comment_cloud_group_mode",
        )

    if group_mode == "source":
        group_series = filtered["source"].map(lambda s: SOURCE_LABELS.get(str(s), str(s)))
    elif group_mode == "model":
        group_series = filtered["model_display"].astype(str)
    else:
        group_series = filtered.apply(
            lambda r: f"{r['model_display']} | {SOURCE_LABELS.get(str(r['source']), str(r['source']))}",
            axis=1,
        )

    group_values = sorted(group_series.unique().tolist())
    with wc2:
        sel_group = st.selectbox(
            "Selection",
            group_values if group_values else ["(none)"],
            key="comment_cloud_group_value",
        )

    cloud_counts: Counter[str] = Counter()
    if group_values:
        for text in filtered[group_series == sel_group]["comment"].fillna("").astype(str):
            cloud_counts.update(_tokenize_comment(text))
    st.markdown(_render_word_cloud(cloud_counts), unsafe_allow_html=True)

    st.caption(f"Showing {min(len(filtered), 25)} of {len(filtered)} matching rows")

    if filtered.empty:
        st.info("No rows match the current filters.")
        return

    for _, row in filtered.head(25).iterrows():
        comment = row["comment"].strip() or "(no comment)"
        src_label = SOURCE_LABELS.get(str(row["source"]), str(row["source"]))
        header = (
            f"**{row['model_display']}** | {src_label} | `{row['article_id']}` | "
            f"run {row['run']} | bias **{row['overall_bias']:.3f}** | "
            f"conf {row['confidence']:.2f} | `{row['status']}`"
        )
        with st.expander(header, expanded=False):
            bc1, bc2, bc3, bc4 = st.columns(4)
            bc1.metric("Subject", f"{row['subject_bias']:.2f}")
            bc2.metric("Framing", f"{row['framing_bias']:.2f}")
            bc3.metric("Treatment", f"{row['treatment_bias']:.2f}")
            bc4.metric("Guests", f"{row['guests_bias']:.2f}")
            st.markdown("**Comment:**")
            st.write(comment)

    if len(filtered) > 25:
        st.caption("Narrow your filters or use the search box to see more results.")


def _tab_explanation(df: pd.DataFrame) -> None:
    st.subheader("What this app is")
    st.markdown(
        "This dashboard visualizes political-bias scores generated from scraped articles "
        "across multiple sources and models."
    )
    st.link_button("Open public app URL", LIVE_APP_URL)

    st.subheader("What was done")
    st.markdown(
        "- Articles were scraped from configured news sources.\n"
        "- Local LLMs scored each article on four bias dimensions.\n"
        "- Each model/run/article score was stored in `bias_data.csv`.\n"
        "- This app renders read-only analysis for comparison and inspection."
    )

    st.subheader("Bias dimensions")
    st.markdown(
        "- `subject_bias`: topic selection leaning.\n"
        "- `framing_bias`: framing/tone leaning.\n"
        "- `treatment_bias`: favorability toward left vs right.\n"
        "- `guests_bias`: leaning of quoted/invited voices.\n"
        "- Score range is `[-1.0, +1.0]` where negative=left, positive=right."
    )

    st.subheader("How to read the tabs")
    st.markdown(
        "- `Overview`: dataset size, source×model heatmap, confidence intervals, model summary table.\n"
        "- `Model Comparison`: scatter, boxplots, and per-dimension means by model.\n"
        "- `Source Explorer`: article-level behavior inside one source.\n"
        "- `Comment Explorer`: inspect raw model comments with cascading filters."
    )

    st.subheader("Exact scoring prompt")
    st.caption("Source of truth: `src/polibias/prompt.md`")
    st.code(load_prompt_template(), language="markdown")

    st.subheader("Current filtered view")
    col1, col2, col3 = st.columns(3)
    col1.metric("Visible scores", len(df))
    col2.metric("Visible models", df["model_display"].nunique())
    col3.metric("Visible sources", df["source"].nunique())


# ── app entry point ────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="polibias dashboard",
        page_icon="📰",
        layout="wide",
    )
    st.title("polibias — Bias Analysis Dashboard")
    st.caption(f"Live app URL: {LIVE_APP_URL}")

    df_full = load_bias()

    # ── sidebar filters ────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Filters")

        all_sources = sorted(df_full["source"].unique())
        sel_sources = st.multiselect(
            "Sources",
            all_sources,
            default=all_sources,
            format_func=lambda s: SOURCE_LABELS.get(s, s),
        )

        all_models = sorted(df_full["model_display"].unique())
        sel_models = st.multiselect("Models", all_models, default=all_models)

        all_runs = sorted(df_full["run"].unique())
        sel_runs = st.multiselect("Runs", [str(r) for r in all_runs], default=[str(r) for r in all_runs])

        status_filter = st.radio("Status", ["All", "ok only"], index=0)

        st.divider()
        st.caption(
            "polibias — read-only visualization of pipeline results.\n\n"
            f"Data: `bias_data.csv`"
        )

    # ── apply filters ──────────────────────────────────────────────────────────
    df = df_full.copy()
    if sel_sources:
        df = df[df["source"].isin(sel_sources)]
    if sel_models:
        df = df[df["model_display"].isin(sel_models)]
    if sel_runs:
        df = df[df["run"].isin([int(r) for r in sel_runs])]
    if status_filter == "ok only":
        df = df[df["status"] == "ok"]

    if df.empty:
        st.warning("No data matches the current sidebar filters.")
        return

    # ── tabs ───────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Explanation",
        "Overview",
        "Model Comparison",
        "Source Explorer",
        "Comment Explorer",
    ])

    with tab1:
        _tab_explanation(df)
    with tab2:
        _tab_overview(df)
    with tab3:
        _tab_model_comparison(df)
    with tab4:
        _tab_source_explorer(df)
    with tab5:
        _tab_comment_explorer(df)


if __name__ == "__main__":
    main()
