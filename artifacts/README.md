# Versioned Artifacts

This folder stores selected expensive-to-recompute outputs that should be versioned.

Tracked artifacts:
- `bias_data_main.csv`: canonical bias table snapshot copied from `data/runs/run_results/bias_data.csv`.

Policy:
- Keep only small, high-value outputs here.
- Do not add generated runtime caches (`__pycache__`, `*.egg-info`, raw run trees).
