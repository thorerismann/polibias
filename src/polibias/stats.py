"""Statistical analysis: inter-model agreement, within-model consistency, CIs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


BIAS_COLS = ["subject_bias", "framing_bias", "treatment_bias", "guests_bias", "overall_bias"]


# ---------- Fleiss' kappa (inter-model agreement) ----------

def _to_bins(values: pd.Series, n_bins: int = 5) -> pd.Series:
    """Discretise continuous bias scores into equal-width bins for kappa."""
    bins = np.linspace(-1, 1, n_bins + 1)
    return pd.cut(values, bins=bins, labels=False, include_lowest=True)


def fleiss_kappa(ratings_matrix: np.ndarray) -> float:
    """Compute Fleiss' kappa from an (n_subjects x n_categories) count matrix."""
    n_sub, n_cat = ratings_matrix.shape
    n_raters = ratings_matrix.sum(axis=1)[0]
    if n_raters <= 1:
        return float("nan")

    p_j = ratings_matrix.sum(axis=0) / (n_sub * n_raters)
    P_i = (ratings_matrix ** 2).sum(axis=1) - n_raters
    P_i = P_i / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()
    P_e = (p_j ** 2).sum()

    if P_e == 1.0:
        return float("nan")
    return float((P_bar - P_e) / (1 - P_e))


def compute_fleiss_kappa_per_article(bias_df: pd.DataFrame, col: str = "overall_bias") -> pd.DataFrame:
    """Compute Fleiss' kappa per article across models.

    Each model's score (averaged over runs) is treated as one rater.
    """
    rows = []
    df = bias_df.dropna(subset=[col])
    for article_id, grp in df.groupby("article_id"):
        model_means = grp.groupby("model")[col].mean()
        if len(model_means) < 2:
            continue
        binned = _to_bins(model_means)
        binned = binned.dropna()
        if binned.empty:
            continue
        n_cats = 5
        counts = np.zeros((1, n_cats), dtype=float)
        for b in binned:
            counts[0, int(b)] += 1
        k = fleiss_kappa(counts)
        rows.append({"article_id": article_id, f"fleiss_kappa_{col}": k})
    return pd.DataFrame(rows)


# ---------- ICC (within-model consistency) ----------

def icc_oneway(values: np.ndarray) -> float:
    """ICC(1,1) — one-way random, single measures.

    *values* is a 1-D array of repeated scores from the same model on the
    same article.  Returns NaN if fewer than 2 values.
    """
    vals = values[~np.isnan(values)]
    k = len(vals)
    if k < 2:
        return float("nan")
    grand_mean = vals.mean()
    ms_between = np.var(vals, ddof=1)
    if ms_between == 0:
        return 1.0
    ms_within = np.mean((vals - grand_mean) ** 2)
    if ms_within == 0:
        return 1.0
    return float((ms_between - ms_within) / (ms_between + (k - 1) * ms_within))


def compute_icc_per_model(bias_df: pd.DataFrame, col: str = "overall_bias") -> pd.DataFrame:
    """ICC(1,1) per model — measures within-model consistency across runs."""
    rows = []
    df = bias_df.dropna(subset=[col])
    for model, mgrp in df.groupby("model"):
        iccs = []
        for article_id, agrp in mgrp.groupby("article_id"):
            vals = agrp[col].values.astype(float)
            if len(vals) >= 2:
                iccs.append(icc_oneway(vals))
        if iccs:
            rows.append({
                "model": model,
                f"mean_icc_{col}": float(np.nanmean(iccs)),
                f"median_icc_{col}": float(np.nanmedian(iccs)),
                "n_articles": len(iccs),
            })
    return pd.DataFrame(rows)


# ---------- Confidence intervals ----------

def compute_model_ci(bias_df: pd.DataFrame, col: str = "overall_bias", confidence: float = 0.95) -> pd.DataFrame:
    """Mean bias and CI per model."""
    rows = []
    df = bias_df.dropna(subset=[col])
    for model, grp in df.groupby("model"):
        vals = grp[col].values.astype(float)
        n = len(vals)
        if n < 2:
            continue
        mean = float(vals.mean())
        se = float(sp_stats.sem(vals))
        t_crit = float(sp_stats.t.ppf((1 + confidence) / 2, n - 1))
        rows.append({
            "model": model,
            f"mean_{col}": mean,
            f"std_{col}": float(vals.std(ddof=1)),
            "n": n,
            "ci_low": mean - t_crit * se,
            "ci_high": mean + t_crit * se,
        })
    return pd.DataFrame(rows)


# ---------- Orchestrator ----------

def build_stats_report(bias_df: pd.DataFrame) -> dict[str, Any]:
    """Compute all stats and return as a dict of DataFrames."""
    report: dict[str, Any] = {}

    report["model_ci"] = compute_model_ci(bias_df)
    report["icc_per_model"] = compute_icc_per_model(bias_df)
    report["fleiss_per_article"] = compute_fleiss_kappa_per_article(bias_df)

    # Summary stats per model
    summary_rows = []
    for model, grp in bias_df.groupby("model"):
        row: dict[str, Any] = {"model": model}
        for c in BIAS_COLS:
            vals = grp[c].dropna()
            if not vals.empty:
                row[f"{c}_mean"] = float(vals.mean())
                row[f"{c}_std"] = float(vals.std(ddof=1))
        row["n_ok"] = int((grp["status"] == "ok").sum())
        row["n_recovered"] = int((grp["status"] == "recovered").sum())
        row["n_fallback"] = int((grp["status"] == "fallback").sum())
        summary_rows.append(row)
    report["model_summary"] = pd.DataFrame(summary_rows)

    return report


def run_stats(settings) -> None:
    """Load bias CSV, compute stats, print and save report."""
    from polibias.analysis import build_bias_frame

    bias_df = build_bias_frame(settings)
    if bias_df.empty:
        print("  No bias data found. Run 'score' and 'analyse' first.")
        return

    report = build_stats_report(bias_df)

    print("\n--- Model summary ---")
    if not report["model_summary"].empty:
        print(report["model_summary"].to_string(index=False))

    print("\n--- Confidence intervals (overall_bias, 95%) ---")
    if not report["model_ci"].empty:
        print(report["model_ci"].to_string(index=False))

    print("\n--- Within-model consistency (ICC) ---")
    if not report["icc_per_model"].empty:
        print(report["icc_per_model"].to_string(index=False))

    print("\n--- Inter-model agreement (Fleiss' kappa per article) ---")
    if not report["fleiss_per_article"].empty:
        print(report["fleiss_per_article"].to_string(index=False))

    # Save combined CSV
    combined = report["model_summary"]
    combined.to_csv(settings.stats_csv_path, index=False)
    print(f"\n  Wrote {settings.stats_csv_path}")
