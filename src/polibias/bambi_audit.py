"""Bayesian audit of LLM scoring behaviour using Bambi.

This module runs a two-part analysis:
1) Failure model: probability that a score is present (non-NaN overall bias)
2) Score model: distribution of score values conditional on success

It also performs holdout evaluation (default 50/50 split) and can generate
an HTML visualization report from produced artifacts.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import ks_2samp, wasserstein_distance

from polibias.analysis import build_bias_frame, build_webdata_frame


@dataclass(frozen=True)
class BambiAuditOptions:
    draws: int = 1500
    tune: int = 1500
    chains: int = 4
    cores: int = 2
    target_accept: float = 0.9
    random_seed: int = 42
    collapse_runs: bool = False
    complete_articles_only: bool = False
    no_imputation: bool = False
    test_fraction: float = 0.5
    sample_eff_repeats: int = 40


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


def _build_features(settings, raw_df: pd.DataFrame, no_imputation: bool = False) -> pd.DataFrame:
    web_df = build_webdata_frame(settings)
    if web_df.empty:
        raw_df = raw_df.copy()
        raw_df["text_words_total"] = np.nan
        raw_df["log_words"] = np.nan
        raw_df["section"] = "unknown"
        raw_df["confidence"] = pd.to_numeric(raw_df["confidence"], errors="coerce")
        raw_df["confidence_z"] = _safe_z(raw_df["confidence"])
        raw_df["log_words_z"] = np.nan
        if not no_imputation:
            raw_df["confidence_z"] = raw_df["confidence_z"].fillna(0.0)
            raw_df["log_words_z"] = raw_df["log_words_z"].fillna(0.0)
        raw_df["success"] = raw_df["overall_bias"].notna().astype(int)
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
    if not no_imputation:
        merged["confidence_z"] = merged["confidence_z"].fillna(0.0)
        merged["log_words_z"] = merged["log_words_z"].fillna(0.0)
    merged["success"] = merged["overall_bias"].notna().astype(int)
    return merged


def _filter_complete_articles(df: pd.DataFrame) -> pd.DataFrame:
    """Keep articles where every model has at least one successful score."""
    success = df[df["success"] == 1]
    if success.empty:
        return df.iloc[0:0].copy()
    model_count = success["model"].nunique()
    coverage = success.groupby("article_id")["model"].nunique()
    keep_articles = coverage[coverage == model_count].index
    return df[df["article_id"].isin(keep_articles)].copy()


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


def _split_train_test(df: pd.DataFrame, test_fraction: float, seed: int) -> pd.DataFrame:
    """Split within (article_id, model) groups to avoid dropping group levels."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["split"] = "train"

    for _, idx in out.groupby(["article_id", "model"], sort=False).groups.items():
        ids = np.array(list(idx))
        rng.shuffle(ids)
        n = len(ids)
        if n < 2:
            continue
        n_test = int(round(n * test_fraction))
        n_test = max(1, min(n - 1, n_test))
        test_ids = ids[:n_test]
        out.loc[test_ids, "split"] = "test"

    return out


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((y - p) ** 2))


def _log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-9) -> float:
    pp = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(pp) + (1 - y) * np.log(1 - pp)))


def _roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    sum_pos = ranks[y == 1].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def _extract_draw_stats(idata, var_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(idata.posterior[var_name].values)  # (chain, draw, obs)
    mean = arr.mean(axis=(0, 1))
    lo = np.quantile(arr, 0.05, axis=(0, 1))
    hi = np.quantile(arr, 0.95, axis=(0, 1))
    return mean, lo, hi


def _write_effect_plot(summary_csv: Path, output_html: Path, title: str) -> None:
    s = pd.read_csv(summary_csv, index_col=0).reset_index(names="term")
    mask = (
        s["term"].str.startswith("model[")
        | s["term"].isin(["confidence_z", "log_words_z", "Intercept"])
    )
    s = s[mask].copy()
    if s.empty:
        return
    s["effect"] = s["term"].str.replace("model[", "", regex=False).str.replace("]", "", regex=False)
    s = s.sort_values("mean")
    fig = px.scatter(
        s,
        x="mean",
        y="effect",
        error_x=(s["hdi_97%"] - s["mean"]).clip(lower=0),
        error_x_minus=(s["mean"] - s["hdi_3%"]).clip(lower=0),
        title=title,
        hover_data=["sd", "hdi_3%", "hdi_97%", "ess_bulk", "r_hat"],
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=max(420, 26 * len(s)))
    fig.write_html(output_html, include_plotlyjs="cdn")


def _compute_sample_efficiency(
    score_df: pd.DataFrame,
    out_dir: Path,
    test_fraction: float,
    seed: int,
    repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate how fast a model's score distribution is recoverable from few samples."""
    if score_df.empty:
        empty = pd.DataFrame()
        empty.to_csv(out_dir / "sample_efficiency_by_model.csv", index=False)
        empty.to_csv(out_dir / "sample_efficiency_overall.csv", index=False)
        return empty, empty

    split = _split_train_test(score_df, test_fraction=test_fraction, seed=seed + 100)
    train = split[split["split"] == "train"].copy()
    test = split[split["split"] == "test"].copy()
    if train.empty or test.empty:
        empty = pd.DataFrame()
        empty.to_csv(out_dir / "sample_efficiency_by_model.csv", index=False)
        empty.to_csv(out_dir / "sample_efficiency_overall.csv", index=False)
        return empty, empty

    rows = []
    rng = np.random.default_rng(seed + 200)
    quantiles = np.array([0.1, 0.5, 0.9])

    for model, g_train in train.groupby("model"):
        g_test = test[test["model"] == model]
        if g_test.empty:
            continue
        train_vals = g_train["overall_bias"].to_numpy(dtype=float)
        test_vals = g_test["overall_bias"].to_numpy(dtype=float)
        if len(train_vals) < 2 or len(test_vals) < 2:
            continue

        # Compact grid that still shows early-vs-late behavior.
        k_grid = sorted(set([2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, len(train_vals)]))
        k_grid = [k for k in k_grid if k <= len(train_vals)]
        test_q = np.quantile(test_vals, quantiles)

        for k in k_grid:
            wds: list[float] = []
            kss: list[float] = []
            qes: list[float] = []

            for _ in range(repeats):
                replace = k > len(train_vals)
                samp = rng.choice(train_vals, size=k, replace=replace)
                wd = float(wasserstein_distance(samp, test_vals))
                ks = float(ks_2samp(samp, test_vals, method="auto").statistic)
                sq = np.quantile(samp, quantiles)
                qe = float(np.mean(np.abs(sq - test_q)))
                wds.append(wd)
                kss.append(ks)
                qes.append(qe)

            rows.append(
                {
                    "model": model,
                    "k_samples": k,
                    "n_train_model": len(train_vals),
                    "n_test_model": len(test_vals),
                    "wasserstein_mean": float(np.mean(wds)),
                    "wasserstein_sd": float(np.std(wds, ddof=1)) if len(wds) > 1 else 0.0,
                    "ks_mean": float(np.mean(kss)),
                    "ks_sd": float(np.std(kss, ddof=1)) if len(kss) > 1 else 0.0,
                    "quantile_mae_mean": float(np.mean(qes)),
                    "quantile_mae_sd": float(np.std(qes, ddof=1)) if len(qes) > 1 else 0.0,
                }
            )

    by_model = pd.DataFrame(rows)
    if by_model.empty:
        by_model.to_csv(out_dir / "sample_efficiency_by_model.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "sample_efficiency_overall.csv", index=False)
        return by_model, pd.DataFrame()

    by_model.to_csv(out_dir / "sample_efficiency_by_model.csv", index=False)

    overall = (
        by_model.groupby("k_samples", as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "wasserstein_mean": np.average(g["wasserstein_mean"], weights=g["n_test_model"]),
                    "ks_mean": np.average(g["ks_mean"], weights=g["n_test_model"]),
                    "quantile_mae_mean": np.average(g["quantile_mae_mean"], weights=g["n_test_model"]),
                    "n_models": g["model"].nunique(),
                }
            )
        )
        .reset_index(drop=True)
        .sort_values("k_samples")
    )
    overall.to_csv(out_dir / "sample_efficiency_overall.csv", index=False)
    return by_model, overall


def _sample_efficiency_comment(overall: pd.DataFrame) -> str:
    if overall.empty or len(overall) < 2:
        return "Not enough data to infer sample-efficiency trends."
    first = overall.iloc[0]
    last = overall.iloc[-1]
    drop = (first["wasserstein_mean"] - last["wasserstein_mean"]) / max(first["wasserstein_mean"], 1e-9)
    if drop > 0.5:
        trend = "strongly improve"
    elif drop > 0.25:
        trend = "improve"
    else:
        trend = "improve only slightly"
    return (
        f"As k increases from {int(first['k_samples'])} to {int(last['k_samples'])}, "
        f"distribution reconstruction errors {trend} "
        f"(Wasserstein {first['wasserstein_mean']:.3f} → {last['wasserstein_mean']:.3f})."
    )


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


def _fit_and_evaluate(df: pd.DataFrame, out_dir: Path, options: BambiAuditOptions) -> dict[str, Any]:
    import arviz as az
    import bambi as bmb

    result: dict[str, Any] = {}
    split_df = _split_train_test(df, test_fraction=options.test_fraction, seed=options.random_seed)

    d_fail_train = split_df[split_df["split"] == "train"].copy()
    d_fail_test = split_df[split_df["split"] == "test"].copy()

    if options.no_imputation:
        fail_formula = "success ~ model + (1|article_id)"
    else:
        fail_formula = "success ~ model + confidence_z + log_words_z + (1|article_id)"

    fail_model = bmb.Model(fail_formula, data=d_fail_train, family="bernoulli")
    idata_fail = fail_model.fit(
        draws=options.draws,
        tune=options.tune,
        chains=options.chains,
        cores=options.cores,
        random_seed=options.random_seed,
        target_accept=options.target_accept,
    )
    az.summary(idata_fail).to_csv(out_dir / "bambi_failure_summary.csv")

    pred_fail = fail_model.predict(
        idata_fail,
        data=d_fail_test,
        kind="response_params",
        inplace=False,
        sample_new_groups=True,
    )
    p_mean, p_lo, p_hi = _extract_draw_stats(pred_fail, "p")

    fail_pred_df = d_fail_test[["model", "article_id", "run", "success"]].copy()
    fail_pred_df["p_success"] = p_mean
    fail_pred_df["p_success_lo"] = p_lo
    fail_pred_df["p_success_hi"] = p_hi
    fail_pred_df.to_csv(out_dir / "holdout_failure_predictions.csv", index=False)

    y_fail = fail_pred_df["success"].to_numpy(dtype=float)
    p_fail = fail_pred_df["p_success"].to_numpy(dtype=float)
    fail_metrics = {
        "brier": _brier(y_fail, p_fail),
        "log_loss": _log_loss(y_fail, p_fail),
        "accuracy_0.5": float(np.mean((p_fail >= 0.5).astype(int) == y_fail.astype(int))),
        "roc_auc": _roc_auc(y_fail, p_fail),
        "n_test": int(len(fail_pred_df)),
    }

    fail_by_model = []
    for model, g in fail_pred_df.groupby("model"):
        y = g["success"].to_numpy(dtype=float)
        p = g["p_success"].to_numpy(dtype=float)
        fail_by_model.append({
            "model": model,
            "n_test": len(g),
            "brier": _brier(y, p),
            "accuracy_0.5": float(np.mean((p >= 0.5).astype(int) == y.astype(int))),
            "roc_auc": _roc_auc(y, p),
        })
    pd.DataFrame(fail_by_model).sort_values("brier").to_csv(
        out_dir / "holdout_failure_metrics_by_model.csv", index=False
    )

    d_score = split_df[split_df["success"] == 1].copy()
    # Re-split on successful rows to keep train/test valid for the score model hierarchy.
    d_score = _split_train_test(
        d_score, test_fraction=options.test_fraction, seed=options.random_seed + 1
    )

    if options.collapse_runs:
        # Split needs to be recomputed after collapse because row-level split labels are gone.
        d_score = _collapse_for_score(d_score)
        d_score = _split_train_test(
            d_score, test_fraction=options.test_fraction, seed=options.random_seed + 1
        )

    d_score = d_score.dropna(subset=["overall_bias"]).copy()
    d_score["y01"] = ((d_score["overall_bias"] + 1.0) / 2.0).clip(1e-3, 1 - 1e-3)

    if options.no_imputation:
        d_score = d_score.dropna(subset=["confidence_z", "log_words_z"]).copy()

    d_score_train = d_score[d_score["split"] == "train"].copy()
    d_score_test = d_score[d_score["split"] == "test"].copy()

    score_formula = "y01 ~ model + confidence_z + log_words_z + (1|article_id)"
    score_model = bmb.Model(score_formula, data=d_score_train, family="beta")
    idata_score = score_model.fit(
        draws=options.draws,
        tune=options.tune,
        chains=options.chains,
        cores=options.cores,
        random_seed=options.random_seed,
        target_accept=options.target_accept,
    )
    az.summary(idata_score).to_csv(out_dir / "bambi_score_summary.csv")

    pred_score = score_model.predict(
        idata_score,
        data=d_score_test,
        kind="response_params",
        inplace=False,
        sample_new_groups=True,
    )
    mu_mean, mu_lo, mu_hi = _extract_draw_stats(pred_score, "mu")

    pred_cols = [c for c in ["model", "article_id", "run", "overall_bias", "y01"] if c in d_score_test.columns]
    score_pred_df = d_score_test[pred_cols].copy()
    score_pred_df["pred_y01"] = mu_mean
    score_pred_df["pred_y01_lo"] = mu_lo
    score_pred_df["pred_y01_hi"] = mu_hi
    score_pred_df["pred_overall_bias"] = (2 * score_pred_df["pred_y01"] - 1).clip(-1, 1)
    score_pred_df["pred_overall_bias_lo"] = (2 * score_pred_df["pred_y01_lo"] - 1).clip(-1, 1)
    score_pred_df["pred_overall_bias_hi"] = (2 * score_pred_df["pred_y01_hi"] - 1).clip(-1, 1)
    score_pred_df["residual"] = score_pred_df["overall_bias"] - score_pred_df["pred_overall_bias"]
    score_pred_df.to_csv(out_dir / "holdout_score_predictions.csv", index=False)

    y = score_pred_df["overall_bias"].to_numpy(dtype=float)
    yhat = score_pred_df["pred_overall_bias"].to_numpy(dtype=float)
    score_metrics = {
        "mae": float(np.mean(np.abs(y - yhat))),
        "rmse": _rmse(y, yhat),
        "r2": _r2(y, yhat),
        "corr": float(np.corrcoef(y, yhat)[0, 1]) if len(score_pred_df) > 1 else float("nan"),
        "n_test": int(len(score_pred_df)),
    }

    score_by_model = []
    for model, g in score_pred_df.groupby("model"):
        yy = g["overall_bias"].to_numpy(dtype=float)
        yyhat = g["pred_overall_bias"].to_numpy(dtype=float)
        score_by_model.append({
            "model": model,
            "n_test": len(g),
            "mae": float(np.mean(np.abs(yy - yyhat))),
            "rmse": _rmse(yy, yyhat),
            "r2": _r2(yy, yyhat),
            "corr": float(np.corrcoef(yy, yyhat)[0, 1]) if len(g) > 1 else float("nan"),
        })
    pd.DataFrame(score_by_model).sort_values("rmse").to_csv(
        out_dir / "holdout_score_metrics_by_model.csv", index=False
    )

    # Sample-efficiency curves: how quickly each model's score distribution is recoverable.
    eff_by_model, eff_overall = _compute_sample_efficiency(
        d_score[["model", "article_id", "overall_bias"]].copy(),
        out_dir=out_dir,
        test_fraction=options.test_fraction,
        seed=options.random_seed,
        repeats=options.sample_eff_repeats,
    )

    pd.DataFrame(
        [
            {"component": "failure", "metric": k, "value": v} for k, v in fail_metrics.items()
        ]
        + [{"component": "score", "metric": k, "value": v} for k, v in score_metrics.items()]
    ).to_csv(out_dir / "holdout_metrics.csv", index=False)

    # Posterior predictive summaries
    ppc_fail = fail_model.predict(idata_fail, kind="response", inplace=False)
    ppc_score = score_model.predict(idata_score, kind="response", inplace=False)
    az.summary(ppc_fail).to_csv(out_dir / "bambi_failure_ppc_summary.csv")
    az.summary(ppc_score).to_csv(out_dir / "bambi_score_ppc_summary.csv")

    _write_effect_plot(
        out_dir / "bambi_failure_summary.csv",
        out_dir / "bambi_failure_effects.html",
        "Failure model effects (log-odds scale)",
    )
    _write_effect_plot(
        out_dir / "bambi_score_summary.csv",
        out_dir / "bambi_score_effects.html",
        "Score model effects (logit mean scale)",
    )

    result.update(
        {
            "failure_formula": fail_formula,
            "score_formula": score_formula,
            "n_failure_train": len(d_fail_train),
            "n_failure_test": len(d_fail_test),
            "n_score_train": len(d_score_train),
            "n_score_test": len(d_score_test),
            "failure_metrics": fail_metrics,
            "score_metrics": score_metrics,
            "sample_efficiency_summary": _sample_efficiency_comment(eff_overall),
        }
    )
    return result


def _build_bambi_report_html(out_dir: Path) -> Path:
    metrics_path = out_dir / "holdout_metrics.csv"
    fail_pred_path = out_dir / "holdout_failure_predictions.csv"
    score_pred_path = out_dir / "holdout_score_predictions.csv"

    if not metrics_path.exists() or not fail_pred_path.exists() or not score_pred_path.exists():
        raise FileNotFoundError("Missing holdout outputs. Run 'bambi-analyse' first.")

    metrics = pd.read_csv(metrics_path)
    fail_pred = pd.read_csv(fail_pred_path)
    score_pred = pd.read_csv(score_pred_path)
    fail_by_model = pd.read_csv(out_dir / "holdout_failure_metrics_by_model.csv")
    score_by_model = pd.read_csv(out_dir / "holdout_score_metrics_by_model.csv")
    eff_by_model = pd.read_csv(out_dir / "sample_efficiency_by_model.csv")
    eff_overall = pd.read_csv(out_dir / "sample_efficiency_overall.csv")

    fail_cal = fail_pred.copy()
    fail_cal["bin"] = pd.qcut(fail_cal["p_success"], q=10, duplicates="drop")
    cal_df = (
        fail_cal.groupby("bin", as_index=False)
        .agg(pred_p=("p_success", "mean"), obs_rate=("success", "mean"), n=("success", "size"))
    )

    fig_cal = px.scatter(
        cal_df,
        x="pred_p",
        y="obs_rate",
        size="n",
        title="Failure model calibration (holdout)",
        hover_data=["n"],
    )
    fig_cal.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
    fig_cal.update_xaxes(range=[0, 1], title="Predicted P(success)")
    fig_cal.update_yaxes(range=[0, 1], title="Observed success rate")

    fig_fail_hist = px.histogram(
        fail_pred,
        x="p_success",
        color=fail_pred["success"].map({0: "failed", 1: "success"}),
        barmode="overlay",
        nbins=30,
        title="Failure model score separation (holdout)",
    )
    fig_fail_hist.update_xaxes(title="Predicted P(success)")
    fig_fail_hist.update_yaxes(title="Count")

    fig_scatter = px.scatter(
        score_pred,
        x="pred_overall_bias",
        y="overall_bias",
        color="model",
        hover_data=["article_id", "run", "residual"],
        title="Score model: predicted vs observed overall_bias (holdout)",
    )
    fig_scatter.add_shape(type="line", x0=-1, y0=-1, x1=1, y1=1, line=dict(dash="dash", color="gray"))
    fig_scatter.update_xaxes(range=[-1, 1], title="Predicted overall_bias")
    fig_scatter.update_yaxes(range=[-1, 1], title="Observed overall_bias")

    fig_scatter_facets = px.scatter(
        score_pred,
        x="pred_overall_bias",
        y="overall_bias",
        facet_col="model",
        facet_col_wrap=3,
        hover_data=["article_id", "run", "residual"],
        title="Score model: predicted vs observed by model (holdout)",
    )
    fig_scatter_facets.add_shape(
        type="line", x0=-1, y0=-1, x1=1, y1=1, line=dict(dash="dash", color="gray")
    )
    fig_scatter_facets.update_xaxes(range=[-1, 1], title="Predicted overall_bias")
    fig_scatter_facets.update_yaxes(range=[-1, 1], title="Observed overall_bias")
    fig_scatter_facets.update_layout(height=900)

    fig_resid = px.box(
        score_pred,
        x="model",
        y="residual",
        points="all",
        color="model",
        title="Score model residuals by model (holdout)",
    )
    fig_resid.update_layout(showlegend=False)
    fig_resid.update_yaxes(title="Residual (observed - predicted)")

    fail_m = fail_by_model.melt(
        id_vars=["model", "n_test"],
        value_vars=["brier", "accuracy_0.5", "roc_auc"],
        var_name="metric",
        value_name="value",
    )
    fig_fail_by_model = px.bar(
        fail_m, x="model", y="value", color="metric", barmode="group",
        title="Failure model metrics by model (holdout)"
    )

    score_m = score_by_model.melt(
        id_vars=["model", "n_test"],
        value_vars=["mae", "rmse", "r2", "corr"],
        var_name="metric",
        value_name="value",
    )
    fig_score_by_model = px.bar(
        score_m, x="model", y="value", color="metric", barmode="group",
        title="Score model metrics by model (holdout)"
    )

    eff_long = eff_overall.melt(
        id_vars=["k_samples", "n_models"],
        value_vars=["wasserstein_mean", "ks_mean", "quantile_mae_mean"],
        var_name="metric",
        value_name="error",
    )
    fig_eff_overall = px.line(
        eff_long,
        x="k_samples",
        y="error",
        color="metric",
        markers=True,
        title="Sample-efficiency (overall): distribution error vs k samples/model",
    )
    fig_eff_overall.update_xaxes(title="k samples per model")
    fig_eff_overall.update_yaxes(title="Distribution error (lower is better)")

    fig_eff_model = px.line(
        eff_by_model,
        x="k_samples",
        y="wasserstein_mean",
        color="model",
        markers=True,
        title="Sample-efficiency by model (Wasserstein distance)",
    )
    fig_eff_model.update_xaxes(title="k samples per model")
    fig_eff_model.update_yaxes(title="Wasserstein distance (lower is better)")

    fail_auc = float(
        metrics[(metrics["component"] == "failure") & (metrics["metric"] == "roc_auc")]["value"].iloc[0]
    )
    score_r2 = float(
        metrics[(metrics["component"] == "score") & (metrics["metric"] == "r2")]["value"].iloc[0]
    )
    eff_note = _sample_efficiency_comment(eff_overall)

    css = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1200px; margin: auto; padding: 20px; color: #222; }
h1 { color: #333; border-bottom: 2px solid #ddd; padding-bottom: 10px; }
h2 { color: #555; margin-top: 34px; }
table { border-collapse: collapse; width: 100%; margin: 14px 0 20px; }
th, td { border: 1px solid #ddd; padding: 8px 10px; text-align: right; }
th { background: #f5f5f5; }
td:first-child, th:first-child { text-align: left; }
.section-note { color: #666; font-size: 0.92em; }
.viz-note { background: #f7f9ff; border: 1px solid #d8e2ff; border-radius: 8px; padding: 10px 12px; margin: 8px 0 14px; }
"""

    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>polibias bambi report</title>",
        f"<style>{css}</style>",
        "</head><body>",
        "<h1>polibias — Bambi Holdout Report</h1>",
        "<p class='section-note'>50/50 holdout diagnostics for failure and score models.</p>",
        "<h2>Holdout Metrics</h2>",
        metrics.to_html(index=False, float_format="%.4f"),
        "<h2>Failure Model Diagnostics</h2>",
        "<div class='viz-note'><b>What this should mean:</b> Calibration points should lie close to diagonal if probability estimates are trustworthy."
        "<br><b>What it means here:</b> Use distance from diagonal to spot over/under-confidence by bin.</div>",
        fig_cal.to_html(full_html=False, include_plotlyjs="cdn"),
        f"<div class='viz-note'><b>What this should mean:</b> Better separation means success/failure get different predicted probabilities."
        f"<br><b>What it means here:</b> Holdout ROC-AUC is {fail_auc:.3f}.</div>",
        fig_fail_hist.to_html(full_html=False, include_plotlyjs=False),
        "<h2>Score Model Diagnostics</h2>",
        "<div class='viz-note'><b>What this should mean:</b> Points near diagonal indicate accurate score reconstruction.</div>",
        fig_scatter.to_html(full_html=False, include_plotlyjs=False),
        f"<div class='viz-note'><b>What it means here:</b> Holdout R² is {score_r2:.3f}; this is the aggregate fit quality.</div>",
        fig_scatter_facets.to_html(full_html=False, include_plotlyjs=False),
        "<div class='viz-note'><b>What this should mean:</b> Residual boxes centered near 0 and narrow imply better per-model calibration and stability.</div>",
        fig_resid.to_html(full_html=False, include_plotlyjs=False),
        "<h2>Per-Model Predictive Metrics</h2>",
        fig_fail_by_model.to_html(full_html=False, include_plotlyjs=False),
        fig_score_by_model.to_html(full_html=False, include_plotlyjs=False),
        "<h2>Sample-Efficiency Curves</h2>",
        "<div class='viz-note'><b>What this should mean:</b> If curves flatten quickly, a few samples are enough to approximate each model's scoring distribution."
        f"<br><b>What it means here:</b> {eff_note}</div>",
        fig_eff_overall.to_html(full_html=False, include_plotlyjs=False),
        fig_eff_model.to_html(full_html=False, include_plotlyjs=False),
        "<h2>Posterior Effect Plots</h2>",
    ]

    for p in [out_dir / "bambi_failure_effects.html", out_dir / "bambi_score_effects.html"]:
        if p.exists():
            txt = p.read_text(encoding="utf-8")
            body_start = txt.find("<body>")
            body_end = txt.rfind("</body>")
            if body_start != -1 and body_end != -1:
                html.append(txt[body_start + 6:body_end])

    html.append("</body></html>")
    out_file = out_dir / "bambi_report.html"
    out_file.write_text("\n".join(html), encoding="utf-8")
    return out_file


def run_bambi_analyse(settings, options: BambiAuditOptions | None = None) -> None:
    """Run two-part Bayesian audit + holdout evaluation under run_dir/bayes."""
    options = options or BambiAuditOptions()

    out_dir = settings.run_dir / "bayes"
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTENSOR_BASE_COMPILEDIR", "/tmp/pytensor")
    os.environ.setdefault("PYTENSOR_FLAGS", "base_compiledir=/tmp/pytensor")

    raw = build_bias_frame(settings)
    if raw.empty:
        print("  No bias data found. Run 'score' and 'analyse' first.")
        return

    df = _build_features(settings, raw, no_imputation=options.no_imputation)
    if options.complete_articles_only:
        df = _filter_complete_articles(df)
        if df.empty:
            print("  No rows remain after complete-articles filter.")
            return

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
        f"- Complete-articles-only filter: {options.complete_articles_only}",
        f"- No-imputation mode: {options.no_imputation}",
        f"- Holdout fraction: {options.test_fraction}",
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
            f"PYTHONPATH=src python -m polibias bambi-analyse --run-dir {settings.run_name}",
            "```",
        ])
        (out_dir / "bambi_audit.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"  Wrote Bayesian preflight outputs to: {out_dir}")
        print("  Bambi not installed; wrote instructions to bambi_audit.md")
        return

    try:
        result = _fit_and_evaluate(df, out_dir, options)
        lines.extend([
            "## Bambi fit",
            "- Status: success",
            f"- Failure model: `{result['failure_formula']}`",
            f"- Score model: `{result['score_formula']}`",
            f"- Failure train/test: {result['n_failure_train']}/{result['n_failure_test']}",
            f"- Score train/test: {result['n_score_train']}/{result['n_score_test']}",
            "",
            "Failure holdout metrics:",
        ])
        for k, v in result["failure_metrics"].items():
            lines.append(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")
        lines.append("")
        lines.append("Score holdout metrics:")
        for k, v in result["score_metrics"].items():
            lines.append(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")
        lines.append("")
        lines.append("Sample-efficiency summary:")
        lines.append(f"- {result['sample_efficiency_summary']}")

        lines.extend([
            "",
            "Outputs:",
            "- `holdout_metrics.csv`",
            "- `holdout_failure_predictions.csv`",
            "- `holdout_score_predictions.csv`",
            "- `holdout_failure_metrics_by_model.csv`",
            "- `holdout_score_metrics_by_model.csv`",
            "- `sample_efficiency_overall.csv`",
            "- `sample_efficiency_by_model.csv`",
            "- `bambi_failure_summary.csv`",
            "- `bambi_score_summary.csv`",
            "- `bambi_failure_effects.html`",
            "- `bambi_score_effects.html`",
        ])

        (out_dir / "bambi_audit.md").write_text("\n".join(lines), encoding="utf-8")

        settings_json = {
            "draws": options.draws,
            "tune": options.tune,
            "chains": options.chains,
            "cores": options.cores,
            "target_accept": options.target_accept,
            "random_seed": options.random_seed,
            "collapse_runs": options.collapse_runs,
            "complete_articles_only": options.complete_articles_only,
            "no_imputation": options.no_imputation,
            "test_fraction": options.test_fraction,
        }
        (out_dir / "bambi_run_settings.json").write_text(
            json.dumps(settings_json, indent=2), encoding="utf-8"
        )
        print(f"  Wrote Bayesian outputs to: {out_dir}")
    except Exception as e:
        lines.extend([
            "## Bambi fit",
            "- Status: failed",
            f"- Error: `{e}`",
        ])
        (out_dir / "bambi_audit.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"  Bambi fit failed; see {out_dir / 'bambi_audit.md'}")


def run_bambi_viz(settings) -> None:
    out_dir = settings.run_dir / "bayes"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = _build_bambi_report_html(out_dir)
    print(f"  Wrote Bambi report: {report}")


# Backward compatibility wrapper
def run_bambi_audit(settings, options: BambiAuditOptions | None = None) -> None:
    run_bambi_analyse(settings, options)
