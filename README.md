# Predicting Mental Health Symptom Severity from Behavioral and Resting-State fMRI Data

This project uses the **Transdiagnostic Connectome Project (OpenNeuro ds005237)** to predict continuous symptom-severity scores with Random Forest regression.

## Prediction targets

- **QIDS:** depression severity
- **STAI-State:** state anxiety
- **SHAPS:** anhedonia
- **PSS:** perceived stress

## Experiments

For each target, the pipeline compares seven feature configurations:

1. Behavioral — all eligible scales
2. Behavioral — clinically relevant categories
3. Behavioral — target category
4. Behavioral - non-sympton category
5. Imaging only
6. Multimodal — all behavioral plus imaging
7. Multimodal — relevant behavioral plus imaging
8. Multimodal — category behavioral plus imaging
9. Multimodal - non-symptom behavioral plus imaging

This produces **36 core models**.

## Data configuration

The dataset is not stored in GitHub. Each teammate creates a local `.env` file from the shared template:

```bash
cp .env.example .env
```

Then edit `.env`:

```text
TCP_DATA_DIR=/absolute/path/to/TCP_dataset
```

The folder must contain:

```text
phenotype/
motion_FD/
fMRI_timeseries_clean_denoised_GSR_parcellated/
```

## Missing values

The preprocessing pipeline converts these known unavailable-value codes to missing values:

```text
999, 9999, -999, -9999
```

Values of `99` and `-99` are **audited but not automatically removed**, because 99 can be a legitimate score in some instruments. Review `outputs/validation/repeated_nine_audit.csv` against the relevant instrument definitions.

## Behavioral preprocessing

Behavioral files are cleaned independently. The pipeline:

- standardizes participant IDs;
- removes administrative, timing, note, and item-level fields;
- retains scale totals and clinically meaningful summary variables;
- removes mostly missing and constant columns;
- loads age, sex, site, and group once from `demos.csv`;
- removes the target instrument from its own predictor set to prevent direct target leakage.

## Imaging features

Each available resting-state run is loaded as a **timepoints × 488 parcels** matrix. The code explicitly verifies the parcel axis and transposes files stored as 488 × timepoints.

Correlations are calculated per run and averaged using Fisher-z transformation. The output contains:

- global signal and temporal summaries;
- whole-connectome correlation summaries;
- graph-organization features at a predefined 10% density;
- parcel-level absolute connectivity strength for all 488 parcels;
- run count for quality-control reporting.

The fixed graph-density variable and acquisition-completeness variables are excluded from model predictors.

Regional parcel features are reduced **inside every training and cross-validation fold** using:

- median imputation;
- variance filtering;
- `SelectKBest(f_regression)` retaining up to 30 parcels;
- standardized PCA retaining up to 15 components.

## Model and validation

All experiments use `RandomForestRegressor` with a fixed random seed. For each target, all modalities use the same participant cohort and deterministic 80/20 train/test split.

The training portion is evaluated with:

- 5 stratified folds;
- 3 repeats;
- 15 cross-validation evaluations total.

Models are selected by **mean repeated cross-validation R²**. The held-out test set is then reported as an independent final evaluation; it is not used to choose the best configuration.

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
```

Launch the interactive dashboard with:

```bash
streamlit run interface.py
```

## Important outputs

```text
outputs/
├── behavioral_by_file/                 cleaned questionnaire summaries
├── behavioral_features.csv             merged behavioral features
├── imaging_features.csv                global, graph, and parcel features
├── experiment_datasets/                matched model-ready datasets
├── cross_category_datasets/            symptom-domain experiments
├── validation/                         automated quality-control reports
└── results/
    ├── model_metrics_comparison.csv
    ├── model_metrics_comparison_sorted.csv
    ├── best_model_by_target.csv
    ├── modality_scope_summary.csv
    ├── predictions/
    ├── feature_importance/
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

The current results show that behavioral questionnaires provide the strongest predictions. Imaging-only models perform near or below an R² of zero, and adding the current imaging representation generally does not improve behavioral models. Anxiety and depression are more predictable than anhedonia. Removing symptom-scale features did not eliminate predictive performance.

These results do **not** establish that resting-state fMRI contains no symptom-related information. They show that this sample, feature representation, and modeling approach did not extract additional out-of-sample value beyond behavioral measures.

## Limitations and future work

- sample size is limited relative to imaging dimensionality;
- motion and scanner-site effects need explicit confound analysis or harmonization;
- parcel strength does not preserve individual connectivity edges;
- only Random Forest regression is used in the core comparison;
- external validation has not yet been performed.

Future work should compare regularized linear models and boosting methods, network-level and edge-based imaging representations, nested hyperparameter tuning, subgroup performance, and external datasets.

## Responsible use

This application is for research and education. Its predictions are not diagnoses and must not replace evaluation by a licensed healthcare professional.
