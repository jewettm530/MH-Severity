# Results must be regenerated

The source code and existing model-ready CSV files were updated to recognize signed missing-value codes (`-999` and `-9999`) and to audit ambiguous `99/-99` values.

The model metrics originally included in this archive were generated before those corrections. A complete retraining attempt was started during the audit, but the full 28-model repeated-cross-validation run exceeded the available execution window. Therefore, treat the included metric values as historical until the pipeline is rerun locally.

Run:

```bash
python3 scripts/05_train_random_forest.py
python3 scripts/06_evaluate_results.py
python3 scripts/07_cross_category_analysis.py
python3 scripts/08_validate_outputs.py
python3 scripts/09_create_results_visualizations.py
```

`06_evaluate_results.py` now selects the best configuration using repeated cross-validation R² rather than held-out test R².
