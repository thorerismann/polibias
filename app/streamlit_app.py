from __future__ import annotations

import io
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from polibias import __main__ as cli  # noqa: E402
from polibias.config import Settings, load_settings  # noqa: E402


SOURCE_LABELS = {
    "RTS": "rts",
    "The Federalist": "the_federalist",
    "Jacobin": "jacobin",
    "All Sources": "all",
}


def _build_settings(run_name: str, config_path: str) -> Settings:
    cfg = Path(config_path).expanduser() if config_path.strip() else None
    return load_settings(cfg, run_name=run_name.strip() or "run_results")


def _run_action(label: str, fn, *args, **kwargs) -> None:  # noqa: ANN001
    st.subheader(label)
    buf = io.StringIO()
    with st.spinner(f"Running {label} ..."):
        try:
            with redirect_stdout(buf), redirect_stderr(buf):
                fn(*args, **kwargs)
            st.success("Completed")
        except SystemExit as e:
            st.error(f"Exited with code {e.code}")
        except Exception as e:  # noqa: BLE001
            st.exception(e)
    output = buf.getvalue().strip()
    if output:
        st.code(output, language="text")


def _show_artifacts(settings: Settings) -> None:
    st.markdown("### Run Artifacts")
    st.text(f"Run directory: {settings.run_dir}")
    for p in [
        settings.bias_csv_path,
        settings.web_csv_path,
        settings.stats_csv_path,
        settings.report_html_path,
        settings.run_dir / "report_rts.html",
        settings.run_dir / "report_fed.html",
        settings.run_dir / "report_jacobin.html",
        settings.run_dir / "report_all.html",
    ]:
        exists = "yes" if p.exists() else "no"
        st.text(f"{p} (exists: {exists})")


def main() -> None:
    st.set_page_config(page_title="polibias control panel", layout="wide")
    st.title("polibias Control Panel")
    st.caption("Run scrape/score/analyze/report tasks and keep HTML outputs.")

    with st.sidebar:
        st.header("Settings")
        run_name = st.text_input("Run name", value="run_results")
        config_path = st.text_input("Config TOML path (optional)", value="")
        source_label = st.selectbox("Source", list(SOURCE_LABELS.keys()), index=3)
        limit = st.number_input("Scrape link limit", min_value=1, max_value=500, value=20)
        urls_file = st.text_input("URLs file for scrape (optional)", value="")
        bucket = st.text_input("GCS bucket (upload)", value="")
        st.caption("Run with: streamlit run app/streamlit_app.py")

    settings = _build_settings(run_name, config_path)
    source = SOURCE_LABELS[source_label]
    urls_file_path = Path(urls_file).expanduser() if urls_file.strip() else None

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Pipeline")
        if st.button("Validate"):
            _run_action("Validate", cli._run_validate, settings)
        if st.button("Scrape"):
            if source == "rts":
                _run_action("Scrape RTS", cli._run_scrape, settings)
            elif source == "the_federalist":
                _run_action(
                    "Scrape Federalist",
                    cli._run_scrape_federalist,
                    settings,
                    int(limit),
                    urls_file_path,
                )
            elif source == "jacobin":
                _run_action(
                    "Scrape Jacobin",
                    cli._run_scrape_jacobin,
                    settings,
                    int(limit),
                    urls_file_path,
                )
            else:
                _run_action("Scrape RTS", cli._run_scrape, settings)
                _run_action(
                    "Scrape Federalist",
                    cli._run_scrape_federalist,
                    settings,
                    int(limit),
                    urls_file_path,
                )
                _run_action(
                    "Scrape Jacobin",
                    cli._run_scrape_jacobin,
                    settings,
                    int(limit),
                    urls_file_path,
                )
        if st.button("Score"):
            if source == "all":
                _run_action("Score All", cli._run_score, settings)
            else:
                _run_action("Score Source", cli._run_score_source, settings, source)
        if st.button("Analyze"):
            _run_action("Analyze", cli._run_analyse, settings)
        if st.button("Run Full Pipeline"):
            _run_action("Validate", cli._run_validate, settings)
            _run_action("Scrape RTS", cli._run_scrape, settings)
            _run_action("Score All", cli._run_score, settings)
            _run_action("Analyze", cli._run_analyse, settings)

    with col2:
        st.markdown("### Reports")
        if st.button("Stats"):
            _run_action("Stats", cli._run_stats, settings)
        if st.button("Export Artifacts"):
            _run_action("Export", cli._run_export, settings)
        if st.button("Visualize"):
            if source == "all":
                _run_action("Cross-source report", cli._run_viz_all, settings)
            elif source == "rts":
                _run_action("RTS report", cli._run_viz_source, settings, "rts", output_name="report_rts.html")
            elif source == "the_federalist":
                _run_action(
                    "Federalist report",
                    cli._run_viz_source,
                    settings,
                    "the_federalist",
                    output_name="report_fed.html",
                )
            elif source == "jacobin":
                _run_action(
                    "Jacobin report",
                    cli._run_viz_source,
                    settings,
                    "jacobin",
                    output_name="report_jacobin.html",
                )
        if st.button("Check Outputs"):
            _run_action("Check", cli._run_check, settings)

    with col3:
        st.markdown("### Bayesian")
        b_draws = st.number_input("Draws", min_value=100, value=1500, step=100)
        b_tune = st.number_input("Tune", min_value=100, value=1500, step=100)
        b_chains = st.number_input("Chains", min_value=1, max_value=8, value=4, step=1)
        b_cores = st.number_input("Cores", min_value=1, max_value=16, value=2, step=1)
        b_accept = st.slider("Target accept", min_value=0.7, max_value=0.99, value=0.9, step=0.01)
        b_seed = st.number_input("Seed", min_value=0, value=42, step=1)
        b_collapse = st.checkbox("Collapse runs", value=False)
        b_complete = st.checkbox("Complete articles only", value=False)
        b_no_imp = st.checkbox("No imputation", value=False)
        b_test_frac = st.slider("Test fraction", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
        if st.button("Run Bambi Analyze"):
            args = type(
                "Args",
                (),
                {
                    "bayes_draws": int(b_draws),
                    "bayes_tune": int(b_tune),
                    "bayes_chains": int(b_chains),
                    "bayes_cores": int(b_cores),
                    "bayes_target_accept": float(b_accept),
                    "bayes_seed": int(b_seed),
                    "bayes_collapse_runs": bool(b_collapse),
                    "bayes_complete_articles_only": bool(b_complete),
                    "bayes_no_imputation": bool(b_no_imp),
                    "bayes_test_fraction": float(b_test_frac),
                },
            )()
            _run_action("Bambi Analyze", cli._run_bambi_analyse, settings, args)
        if st.button("Run Bambi Viz"):
            _run_action("Bambi Viz", cli._run_bambi_viz, settings)
        if st.button("Upload to GCS"):
            if not bucket.strip():
                st.error("Set a bucket in the sidebar first.")
            else:
                _run_action("Upload", cli._run_upload, settings, bucket.strip())

    _show_artifacts(settings)


if __name__ == "__main__":
    main()
