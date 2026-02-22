# Bayesian LLM Audit

This report audits scoring behavior, not real-world bias truth.

## Data
- Rows: 744
- Models: 6
- Articles: 31
- Runs: 4
- Complete-articles-only filter: False
- No-imputation mode: False
- Holdout fraction: 0.5

Top repeated rounded overall_bias values:
- 0.000: n=71, share=0.174
- -0.500: n=30, share=0.074
- -0.250: n=29, share=0.071
- -0.750: n=20, share=0.049
- -0.375: n=17, share=0.042
- 0.150: n=13, share=0.032
- -1.000: n=13, share=0.032
- -0.125: n=11, share=0.027

## Bambi fit
- Status: success
- Failure model: `success ~ model + confidence_z + log_words_z + (1|article_id)`
- Score model: `y01 ~ model + confidence_z + log_words_z + (1|article_id)`
- Failure train/test: 372/372
- Score train/test: 207/201

Failure holdout metrics:
- brier: 0.2065
- log_loss: 0.6008
- accuracy_0.5: 0.6909
- roc_auc: 0.7445
- n_test: 372

Score holdout metrics:
- mae: 0.1910
- rmse: 0.2530
- r2: 0.4776
- corr: 0.7046
- n_test: 201

Sample-efficiency summary:
- As k increases from 2 to 51, distribution reconstruction errors strongly improve (Wasserstein 0.207 → 0.036).

Outputs:
- `holdout_metrics.csv`
- `holdout_failure_predictions.csv`
- `holdout_score_predictions.csv`
- `holdout_failure_metrics_by_model.csv`
- `holdout_score_metrics_by_model.csv`
- `sample_efficiency_overall.csv`
- `sample_efficiency_by_model.csv`
- `bambi_failure_summary.csv`
- `bambi_score_summary.csv`
- `bambi_failure_effects.html`
- `bambi_score_effects.html`