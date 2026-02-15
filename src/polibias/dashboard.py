"""Streamlit dashboard for exploring bias results.

Launch with::

    streamlit run -m polibias.dashboard
    # or
    polibias viz
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from polibias.config import Settings


def _load(path: Path) -> pd.DataFrame:
    p = path
    if not p.exists():
        st.error(
            f"File not found: {p}. Run `python -m polibias analyse` first."
        )
        st.stop()
    df = pd.read_csv(p)
    return df.drop(columns=[c for c in ["Unnamed: 0"] if c in df.columns])


def _load_errors(settings: Settings) -> pd.DataFrame:
    """Load errors.jsonl if it exists."""
    log_path = settings.errors_dir / "errors.jsonl"
    if not log_path.exists():
        return pd.DataFrame()
    rows = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def main() -> None:
    settings = Settings()
    st.set_page_config(page_title="polibias — dashboard", layout="wide")
    st.title("Political Bias — model variability")

    tab_charts, tab_errors = st.tabs(["Charts", "Errors"])

    bdf = _load(settings.bias_csv_path)

    # Sidebar filters
    st.sidebar.header("Filters")
    models = st.sidebar.multiselect(
        "Models", sorted(bdf["model"].unique()), default=sorted(bdf["model"].unique())
    )
    articles = st.sidebar.multiselect(
        "Articles", bdf["article_id"].unique(), default=bdf["article_id"].unique()
    )
    runs = st.sidebar.multiselect(
        "Runs", sorted(bdf["run"].unique()), default=sorted(bdf["run"].unique())
    )
    show_data = st.sidebar.checkbox("Show raw data", value=False)

    # Filter
    bdf["article_str"] = bdf["article_id"].astype(str) + "A"
    bdf["run"] = bdf["run"].astype(str)
    mask = (
        bdf["model"].isin(models)
        & bdf["article_id"].isin(articles)
        & bdf["run"].isin([str(x) for x in runs])
    )
    dff = bdf[mask].copy()

    with tab_charts:
        if show_data:
            st.dataframe(dff, use_container_width=True)

        st.subheader("Bias by model")
        fig1 = px.scatter(dff, y="overall_bias", x="model", color="confidence")
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Bias by article")
        fig2 = px.scatter(dff, y="overall_bias", x="article_str", color="model")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Bias variance by model")
        fig3 = px.box(dff, x="model", y="overall_bias", points="all", color="model")
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Bias heatmap")
        pivot = dff.pivot_table(
            index="article_str", columns="model", values="overall_bias", aggfunc="mean"
        )
        height = max(400, 30 * len(pivot))
        fig4 = px.imshow(
            pivot, color_continuous_scale="RdBu", zmin=-1, zmax=1, aspect="auto"
        )
        fig4.update_layout(height=height, margin=dict(l=200, r=20, t=40, b=40))
        st.plotly_chart(fig4, use_container_width=True)

        st.subheader("Confidence vs |bias|")
        fig5 = px.scatter(
            dff, x="confidence", y=dff["overall_bias"].abs(), color="model"
        )
        fig5.update_yaxes(title="|overall_bias|")
        st.plotly_chart(fig5, use_container_width=True)

    with tab_errors:
        st.subheader("Error log")
        err_df = _load_errors(settings)
        if err_df.empty:
            st.success("No errors recorded.")
        else:
            st.warning(f"{len(err_df)} error(s) found")

            # Filters for error view
            if "stage" in err_df.columns:
                stages = st.multiselect(
                    "Filter by stage",
                    err_df["stage"].unique(),
                    default=err_df["stage"].unique(),
                )
                err_df = err_df[err_df["stage"].isin(stages)]

            if "model" in err_df.columns:
                err_models = st.multiselect(
                    "Filter by model (errors)",
                    err_df["model"].unique(),
                    default=err_df["model"].unique(),
                )
                err_df = err_df[err_df["model"].isin(err_models)]

            st.dataframe(err_df, use_container_width=True)

            # Error summary
            if "stage" in err_df.columns:
                st.subheader("Errors by stage")
                st.bar_chart(err_df["stage"].value_counts())

            if "model" in err_df.columns:
                st.subheader("Errors by model")
                st.bar_chart(err_df["model"].value_counts())


if __name__ == "__main__":
    main()
