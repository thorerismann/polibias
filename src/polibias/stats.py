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
    """Compute Fleiss' kappa from an (n_subjects x n_categories) count matrix.

    Each row is a subject (article).  Each column is a category (bin).
    Cell values are the number of raters (models) that assigned that
    category to that subject.
    """
    n_sub, n_cat = ratings_matrix.shape
    if n_sub < 2:
        return float("nan")
    n_raters = int(ratings_matrix.sum(axis=1)[0])
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


def compute_fleiss_kappa(bias_df: pd.DataFrame, col: str = "overall_bias") -> float:
    """Global Fleiss' kappa across all articles.

    Each model is a rater.  Each article is a subject.  Model scores are
    averaged over runs, then discretised into 5 equal-width bins on [-1, 1].
    The result is a single kappa value measuring inter-model agreement.
    """
    df = bias_df.dropna(subset=[col])
    model_article_means = df.groupby(["article_id", "model"])[col].mean().reset_index()

    pivot = model_article_means.pivot(index="article_id", columns="model", values=col)
    pivot = pivot.dropna()
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        return float("nan")

    n_cats = 5
    ratings = np.zeros((len(pivot), n_cats), dtype=float)
    for i, (_, row) in enumerate(pivot.iterrows()):
        binned = _to_bins(row, n_bins=n_cats).dropna()
        for b in binned:
            ratings[i, int(b)] += 1

    return fleiss_kappa(ratings)


# ---------- ICC (within-model consistency) ----------

def icc_oneway_matrix(data: np.ndarray) -> float:
    """ICC(1,1) — one-way random, single measures.

    *data* is an (n_subjects x k_raters) matrix.  Rows are articles,
    columns are runs.  Returns NaN if fewer than 2 subjects or 2 raters.
    """
    n, k = data.shape
    if n < 2 or k < 2:
        return float("nan")

    # Remove rows with any NaN
    mask = ~np.isnan(data).any(axis=1)
    data = data[mask]
    n = data.shape[0]
    if n < 2:
        return float("nan")

    grand_mean = data.mean()
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)

    # Sum of squares
    ss_total = ((data - grand_mean) ** 2).sum()
    ss_rows = k * ((row_means - grand_mean) ** 2).sum()
    ss_cols = n * ((col_means - grand_mean) ** 2).sum()
    ss_error = ss_total - ss_rows - ss_cols

    # Mean squares (one-way: treat raters as random)
    ms_between = ss_rows / (n - 1)
    ms_within = (ss_total - ss_rows) / (n * (k - 1))

    if ms_within == 0 and ms_between == 0:
        return float("nan")

    return float((ms_between - ms_within) / (ms_between + (k - 1) * ms_within))


def compute_icc_per_model(bias_df: pd.DataFrame, col: str = "overall_bias") -> pd.DataFrame:
    """ICC(1,1) per model across all articles.

    For each model, builds a (articles x runs) matrix and computes ICC.
    Articles are subjects, runs are raters.  This measures how consistent
    the model is across repeated scoring runs.
    """
    rows = []
    df = bias_df.dropna(subset=[col])
    for model, mgrp in df.groupby("model"):
        pivot = mgrp.pivot_table(index="article_id", columns="run", values=col)
        if pivot.shape[0] < 2 or pivot.shape[1] < 2:
            continue
        icc = icc_oneway_matrix(pivot.values)
        rows.append({
            "model": model,
            f"icc_{col}": icc,
            "n_articles": pivot.shape[0],
            "n_runs": pivot.shape[1],
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
    report["fleiss_kappa"] = compute_fleiss_kappa(bias_df)

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

    print("\n--- Within-model consistency (ICC per model) ---")
    if not report["icc_per_model"].empty:
        print(report["icc_per_model"].to_string(index=False))

    kappa = report["fleiss_kappa"]
    print(f"\n--- Inter-model agreement (Fleiss' kappa): {kappa:.3f} ---")
    if np.isnan(kappa):
        print("  (insufficient data for kappa computation)")

    # Save combined CSV
    combined = report["model_summary"]
    combined.to_csv(settings.stats_csv_path, index=False)
    print(f"\n  Wrote {settings.stats_csv_path}")
