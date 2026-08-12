# Predicting Mental Health Symptom Severity from Behavioral and Resting-State fMRI Data

This project uses the **Transdiagnostic Connectome Project (OpenNeuro ds005237)** to predict continuous mental-health symptom severity with Random Forest regression. We compare behavioral questionnaire features, resting-state fMRI features, and multimodal feature sets across depression, anxiety, perceived stress, and anhedonia.

## Project links

- **Project website:** [https://jewettm530.github.io/MH_Severity_Behavioral_and_Scans/](https://jewettm530.github.io/MH-Severity/)
- **Final presentation:** [PDF](Presentations/AI_MH_Severity_Final_Presentation.pdf)
- **Project proposal:** [PDF](Presentations/AI_MH_Severity_Proposal_Presentation.pdf)
- **Streamlit dashboard source:** [interface.py](interface.py)
- **Dataset:** [Transdiagnostic Connectome Project on OpenNeuro](https://openneuro.org/datasets/ds005237/versions/1.1.3)

> Add the deployed Streamlit application URL here once available.

## Prediction targets

- **QIDS:** depression severity
- **STAI-State:** state anxiety
- **PSS:** perceived stress
- **SHAPS:** anhedonia

## Research questions

1. Which data source best predicts continuous symptom severity: behavioral questionnaires, resting-state fMRI, or both?
2. How much predictive information is shared across depression, anxiety, stress, and anhedonia domains?
3. Does predictive performance remain when direct symptom-scale questionnaires are excluded?

## Experiment design

For each target, the pipeline compares nine core feature configurations:

1. Imaging only
2. Behavioral — all eligible scales
3. Behavioral — clinically relevant symptom categories
4. Behavioral — target symptom category
5. Behavioral — non-symptom features
6. Multimodal — all behavioral + imaging
7. Multimodal — relevant behavioral + imaging
8. Multimodal — category behavioral + imaging
9. Multimodal — non-symptom behavioral + imaging

This produces **36 core model configurations** across four targets.

The **non-symptom** scope removes direct symptom-scale questionnaires to test whether broader behavioral, personality, and clinical features retain predictive value without relying on closely overlapping symptom measures.

## Data configuration

The raw TCP dataset is not stored in GitHub. Each teammate creates a local `.env` file from the shared template:

```bash
cp .env.example .env
```

Then set:

```text
TCP_DATA_DIR=/absolute/path/to/TCP_dataset
```

The dataset directory should contain:

```text
phenotype/
motion_FD/
fMRI_timeseries_clean_denoised_GSR_parcellated/
```

## Missing values

Known unavailable-value codes are converted to missing values:

```text
999, 9999, -999, -9999
```

Values of `99` and `-99` are audited but not automatically removed because `99` can be valid for some instruments. See `outputs/validation/repeated_nine_audit.csv`.

## Behavioral preprocessing

The behavioral pipeline:

- standardizes participant IDs;
- removes administrative, timing, note, and item-level fields;
- retains scale totals and meaningful summary variables;
- removes mostly missing and constant columns;
- loads age, sex, site, and group once from `demos.csv`;
- removes the target questionnaire from its own predictor set to prevent direct target leakage.

## Imaging features

Each available resting-state run is interpreted as **timepoints × 488 parcels**. The extraction code verifies the parcel axis before feature calculation.

The imaging representation includes:

- global signal and temporal summaries;
- whole-connectome correlation summaries;
- graph-organization features;
- parcel-level absolute connectivity strength for all 488 parcels;
- run count for quality-control reporting.

Fixed technical variables are excluded from model predictors.

Regional imaging features are reduced **inside each training/CV fold** using median imputation, variance filtering, `SelectKBest(f_regression)` (up to 30 parcels), and standardized PCA (up to 15 components).

## Model and validation

All experiments use `RandomForestRegressor` with a fixed random seed.

For a given target:

- all modalities use the same participant cohort;
- the same deterministic 80/20 train/test split is used;
- the training set is evaluated with 5 stratified folds repeated 3 times;
- model selection is based on **mean repeated cross-validation R²**;
- the held-out test set is reported only after model selection.

Metrics include MAE, RMSE, R², cross-validation variability, sample sizes, and raw/transformed feature counts.

## Run order

From the repository root:

```bash
python3 scripts/01_explore_dataset.py
python3 scripts/02_prepare_behavioral_data.py
python3 scripts/03_extract_imaging_features.py
python3 scripts/04_merge_features.py
python3 scripts/05_train_random_forest.py
python3 scripts/06_evaluate_results.py
python3 scripts/07_cross_category_analysis.py
python3 scripts/08_validate_outputs.py
python3 scripts/09_create_results_visualizations.py
python3 scripts/10_update_github_pages_assets.py
```

Launch the dashboard locally with:

```bash
streamlit run interface.py
```

## Important outputs

```text
outputs/
├── behavioral_by_file/
├── behavioral_features.csv
├── imaging_features.csv
├── experiment_datasets/
├── cross_category_datasets/
├── validation/
└── results/
    ├── model_metrics_comparison.csv
    ├── model_metrics_comparison_sorted.csv
    ├── best_model_by_target.csv
    ├── modality_scope_summary.csv
    ├── cross_category_metrics.csv
    ├── predictions/
    ├── feature_importance/
    ├── models/
    └── figures/
```

## Validation audits

`scripts/08_validate_outputs.py` checks:

- signed repeated-nine values;
- outcome validity and missing targets;
- duplicate participant IDs;
- Spearman feature-to-outcome correlations;
- features with `|ρ| ≥ 0.80`;
- imaging feature count, missingness, and constant columns.

## Current interpretation

Behavioral questionnaire features provide the strongest prediction of symptom severity in the current experiments. Imaging-only models perform near or below R² = 0, and adding the current imaging representation generally does not improve the strongest behavioral-only models. Anxiety and depression are more predictable than anhedonia.

The non-symptom feature experiments also test whether behavioral performance persists after direct symptom-scale questionnaires are removed.

These findings **do not establish that resting-state fMRI contains no symptom-related information**. They show that the current sample, imaging representation, and Random Forest modeling approach did not extract additional out-of-sample predictive value beyond behavioral measures.

## Limitations

- sample size is limited relative to imaging dimensionality;
- motion and scanner-site effects warrant explicit confound analysis or harmonization;
- parcel-strength features do not preserve individual connectivity edges;
- the core comparison uses Random Forest regression;
- external validation has not yet been performed.

## Responsible use

This application is for research and education. Its predictions are **not diagnoses** and must not replace evaluation by a licensed healthcare professional.

## Dataset citation

Chopra, S., Cocuzza, C. V., Lawhead, C., et al. (2025). *The Transdiagnostic Connectome Project: an open dataset for studying brain-behavior relationships in psychiatry.* **Scientific Data, 12**, 923. https://doi.org/10.1038/s41597-025-04895-z
