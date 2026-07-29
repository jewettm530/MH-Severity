"""Streamlit dashboard for the TCP symptom-severity prediction project."""

from pathlib import Path

import pandas as pd
import streamlit as st


# =====================================================
# Page configuration
# =====================================================

st.set_page_config(
    page_title="Mental Health Symptom Severity Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================
# Project paths
# =====================================================

try:
    from scripts.config import RESULTS_DIR
except ImportError:
    # Fallback when Streamlit is launched from inside the scripts directory.
    try:
        from config import RESULTS_DIR
    except ImportError:
        PROJECT_ROOT = Path(__file__).resolve().parent
        RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"


MODEL_METRICS_FILE = RESULTS_DIR / "model_metrics_comparison.csv"
BEST_MODELS_FILE = RESULTS_DIR / "best_model_by_target.csv"


# =====================================================
# Display labels
# =====================================================

TARGET_LABELS = {
    "qids": "Depression — QIDS",
    "stai": "Anxiety — STAI-State",
    "shaps": "Anhedonia — SHAPS",
    "pss": "Stress — PSS",
}

TARGET_SHORT_LABELS = {
    "qids": "QIDS",
    "stai": "STAI-State",
    "shaps": "SHAPS",
    "pss": "PSS",
}

MODEL_LABELS = {
    "imaging_only": "Imaging Only",
    "behavioral_all": "All Behavioral",
    "behavioral_relevant": "Relevant Behavioral",
    "behavioral_category": "Categorical Behavioral",
    "multimodal_all": "All Behavioral + Imaging",
    "multimodal_relevant": "Relevant Behavioral + Imaging",
    "multimodal_category": "Categorical Behavioral + Imaging",
}


# =====================================================
# Helper functions
# =====================================================

def normalize_name(value: object) -> str:
    """Convert labels to consistent lowercase underscore format."""
    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def find_column(
    dataframe: pd.DataFrame,
    possible_names: list[str],
) -> str | None:
    """Find a column using a list of possible names."""
    normalized_columns = {
        normalize_name(column): column
        for column in dataframe.columns
    }

    for name in possible_names:
        normalized_name = normalize_name(name)

        if normalized_name in normalized_columns:
            return normalized_columns[normalized_name]

    return None


def standardize_target(value: object) -> str:
    """Standardize target names used by different result files."""
    normalized = normalize_name(value)

    target_aliases = {
        "qids": "qids",
        "qids_depression": "qids",
        "depression": "qids",
        "qvtot": "qids",
        "stai": "stai",
        "stai_state": "stai",
        "stai_state_anxiety": "stai",
        "anxiety": "stai",
        "shaps": "shaps",
        "anhedonia": "shaps",
        "shaps_anhedonia": "shaps",
        "pss": "pss",
        "stress": "pss",
        "perceived_stress": "pss",
    }

    return target_aliases.get(normalized, normalized)


def standardize_model_name(row: pd.Series) -> str:
    """
    Combine input type and behavioral scope into one model identifier.

    Examples:
        behavioral_only + all -> behavioral_all
        multimodal + relevant -> multimodal_relevant
        imaging_only -> imaging_only
    """
    input_type = normalize_name(row.get("input_type", ""))
    scope = normalize_name(row.get("scope", ""))

    direct_model = normalize_name(row.get("model_key", ""))

    if direct_model and direct_model != "nan":
        return direct_model

    if input_type in {"imaging", "imaging_only"}:
        return "imaging_only"

    if input_type in {"behavioral", "behavioral_only"}:
        return f"behavioral_{scope or 'all'}"

    if input_type in {"multimodal", "combined"}:
        return f"multimodal_{scope or 'all'}"

    existing_model = normalize_name(row.get("model", ""))

    if existing_model and existing_model != "nan":
        return existing_model

    return input_type


@st.cache_data
def load_metrics() -> pd.DataFrame:
    """Load and standardize the model-comparison results."""
    if not MODEL_METRICS_FILE.exists():
        return pd.DataFrame()

    dataframe = pd.read_csv(MODEL_METRICS_FILE)

    target_column = find_column(
        dataframe,
        ["target_key", "target", "target_name", "outcome"],
    )

    input_column = find_column(
        dataframe,
        ["input_type", "modality", "feature_type"],
    )

    scope_column = find_column(
        dataframe,
        ["behavior_scope", "scope", "predictor_scope", "feature_scope"],
    )

    model_column = find_column(
        dataframe,
        ["model", "model_name", "experiment", "experiment_name"],
    )

    if target_column:
        dataframe["target_key"] = dataframe[target_column].apply(
            standardize_target
        )
    else:
        dataframe["target_key"] = "unknown"

    dataframe["input_type"] = (
        dataframe[input_column]
        if input_column
        else ""
    )

    dataframe["scope"] = (
        dataframe[scope_column]
        if scope_column
        else ""
    )

    dataframe["model"] = (
        dataframe[model_column]
        if model_column
        else ""
    )

    dataframe["model_key"] = dataframe.apply(
        standardize_model_name,
        axis=1,
    )

    dataframe["target_label"] = dataframe["target_key"].map(
        TARGET_LABELS
    ).fillna(dataframe["target_key"])

    dataframe["model_label"] = dataframe["model_key"].map(
        MODEL_LABELS
    ).fillna(
        dataframe["model_key"]
        .str.replace("_", " ")
        .str.title()
    )

    metric_aliases = {
        "test_r2": [
            "test_r2",
            "test_r²",
            "r2_test",
            "test_score_r2",
        ],
        "test_mae": [
            "test_mae",
            "mae_test",
            "test_mean_absolute_error",
        ],
        "test_rmse": [
            "test_rmse",
            "rmse_test",
            "test_root_mean_squared_error",
        ],
        "cv_r2_mean": [
            "cv_r2_mean",
            "mean_cv_r2",
            "cv_mean_r2",
            "cv_r2",
        ],
        "cv_r2_std": [
            "cv_r2_std",
            "std_cv_r2",
            "cv_std_r2",
        ],
        "cv_mae_mean": [
            "cv_mae_mean",
            "mean_cv_mae",
            "cv_mean_mae",
            "cv_mae",
        ],
        "cv_rmse_mean": [
            "cv_rmse_mean",
            "mean_cv_rmse",
            "cv_mean_rmse",
            "cv_rmse",
        ],
        "n_participants": [
            "n_participants",
            "n_total",
            "sample_size",
            "n_samples",
            "n",
        ],
        "n_features": [
            "n_features",
            "features_used",
            "transformed_features",
        ],
        "n_features_raw": [
            "n_features_raw",
            "raw_features",
        ],
    }

    for standard_name, aliases in metric_aliases.items():
        matched_column = find_column(dataframe, aliases)

        if matched_column:
            dataframe[standard_name] = pd.to_numeric(
                dataframe[matched_column],
                errors="coerce",
            )

    return dataframe


@st.cache_data
def load_best_models(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load the saved best-model file when available.

    If it is unavailable or cannot be standardized, select the model
    with the highest mean cross-validation R² for each target.
    """
    if BEST_MODELS_FILE.exists():
        best = pd.read_csv(BEST_MODELS_FILE)

        target_column = find_column(
            best,
            ["target_key", "target", "target_name", "outcome"],
        )

        if target_column:
            best["target_key"] = best[target_column].apply(
                standardize_target
            )

            # Merge against the standardized metrics table so the
            # dashboard uses consistent labels and metric columns.
            model_column = find_column(
                best,
                [
                    "model_key",
                    "model",
                    "model_name",
                    "best_model",
                    "experiment_name",
                    "experiment",
                ],
            )

            if model_column:
                best["raw_model_name"] = best[model_column].apply(
                    normalize_name
                )

                possible_matches = metrics.copy()

                matched_rows = []

                for _, best_row in best.iterrows():
                    target_rows = possible_matches[
                        possible_matches["target_key"]
                        == best_row["target_key"]
                    ]

                    direct_match = target_rows[
                        target_rows["model_key"]
                        == best_row["raw_model_name"]
                    ]

                    if direct_match.empty:
                        direct_match = target_rows[
                            target_rows["model_label"]
                            .str.lower()
                            == str(best_row[model_column]).strip().lower()
                        ]

                    if not direct_match.empty:
                        matched_rows.append(direct_match.iloc[0])

                if matched_rows:
                    return pd.DataFrame(matched_rows).reset_index(
                        drop=True
                    )

    if metrics.empty:
        return pd.DataFrame()

    ranking_metric = (
        "cv_r2_mean"
        if "cv_r2_mean" in metrics.columns
        else "test_r2"
    )

    if ranking_metric not in metrics.columns:
        return pd.DataFrame()

    usable = metrics.dropna(
        subset=["target_key", ranking_metric]
    ).copy()

    best_indices = usable.groupby("target_key")[
        ranking_metric
    ].idxmax()

    return usable.loc[best_indices].reset_index(drop=True)


def format_metric(
    value: object,
    digits: int = 3,
) -> str:
    """Safely format a numeric metric."""
    if pd.isna(value):
        return "—"

    return f"{float(value):.{digits}f}"


def build_metric_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create a presentation-friendly model metrics table."""
    columns = {
        "target_label": "Target",
        "model_label": "Best Model",
        "test_r2": "Test R²",
        "test_mae": "Test MAE",
        "test_rmse": "Test RMSE",
        "cv_r2_mean": "CV R² Mean",
        "cv_r2_std": "CV R² SD",
        "cv_mae_mean": "CV MAE Mean",
        "cv_rmse_mean": "CV RMSE Mean",
        "n_participants": "Participants",
        "n_features": "Features Used",
        "n_features_raw": "Raw Features",
    }

    available = [
        column
        for column in columns
        if column in dataframe.columns
    ]

    result = dataframe[available].rename(columns=columns).copy()

    metric_columns = [
        "Test R²",
        "Test MAE",
        "Test RMSE",
        "CV R² Mean",
        "CV R² SD",
        "CV MAE Mean",
        "CV RMSE Mean",
    ]

    for column in metric_columns:
        if column in result.columns:
            result[column] = result[column].map(format_metric)

    integer_columns = [
        "Participants",
        "Features Used",
        "Raw Features",
    ]

    for column in integer_columns:
        if column in result.columns:
            result[column] = result[column].apply(
                lambda value: (
                    "—"
                    if pd.isna(value)
                    else f"{int(value):,}"
                )
            )

    return result


# =====================================================
# Load project results
# =====================================================

metrics = load_metrics()
best_models = load_best_models(metrics)


# =====================================================
# Sidebar
# =====================================================

with st.sidebar:
    st.header("Project Navigation")

    st.markdown(
        """
        Use the tabs at the top of the page to explore:

        - Project dashboard
        - Best-performing models
        - Interactive model comparisons
        - Findings, limitations, and future work
        """
    )

    st.divider()

    st.subheader("Prediction Targets")

    st.markdown(
        """
        - **QIDS:** Depression
        - **STAI-State:** Anxiety
        - **SHAPS:** Anhedonia
        - **PSS:** Perceived stress
        """
    )

    st.divider()

    st.warning(
        """
        This application is for educational and research purposes.

        Its outputs are not medical diagnoses.
        """
    )


# =====================================================
# Application header
# =====================================================

st.title("Mental Health Symptom Severity Predictor")

st.markdown(
    """
    Explore how behavioral questionnaires, resting-state fMRI features,
    and multimodal data perform when predicting mental health symptom
    severity in the Transdiagnostic Connectome Project dataset.
    """
)


# =====================================================
# Main tabs
# =====================================================

dashboard_tab, best_tab, comparison_tab, visualizations_tab, findings_tab = st.tabs(
    [
        "Dashboard",
        "Best Models",
        "Interactive Comparison",
        "Key Visualizations",
        "Findings & Outcomes",
    ]
)


# =====================================================
# Tab 1: Dashboard
# =====================================================

with dashboard_tab:
    st.header("Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Prediction Targets",
            value="4",
        )

    with col2:
        model_count = (
            len(metrics)
            if not metrics.empty
            else 28
        )

        st.metric(
            label="Models Evaluated",
            value=f"{model_count}",
        )

    with col3:
        participant_count = 241

        if (
            not metrics.empty
            and "n_participants" in metrics.columns
            and metrics["n_participants"].notna().any()
        ):
            participant_count = int(
                metrics["n_participants"].max()
            )

        st.metric(
            label="Dataset Participants",
            value=f"{participant_count}",
        )

    with col4:
        st.metric(
            label="Model Type",
            value="Random Forest",
        )

    st.divider()

    st.header("Main Findings")

    finding_col1, finding_col2, finding_col3 = st.columns(3)

    with finding_col1:
        st.success(
            """
            **Behavioral data performed best**

            Questionnaire features consistently produced the strongest
            symptom-severity predictions.
            """
        )

    with finding_col2:
        st.info(
            """
            **Imaging performance was limited**

            Resting-state fMRI features alone generally produced weak or
            negative predictive R² values.
            """
        )

    with finding_col3:
        st.warning(
            """
            **Multimodal data added little**

            Adding imaging features generally did not improve upon the
            strongest behavioral models.
            """
        )

    st.divider()

    st.header("Best Model by Target")

    if best_models.empty:
        st.error(
            f"""
            No model results could be loaded.

            Expected results file:

            `{MODEL_METRICS_FILE}`
            """
        )
    else:
        summary_columns = [
            column
            for column in [
                "target_label",
                "model_label",
                "test_r2",
                "cv_r2_mean",
            ]
            if column in best_models.columns
        ]

        summary_table = best_models[summary_columns].copy()

        summary_table = summary_table.rename(
            columns={
                "target_label": "Target",
                "model_label": "Best Model",
                "test_r2": "Test R²",
                "cv_r2_mean": "CV R² Mean",
            }
        )

        for column in ["Test R²", "CV R² Mean"]:
            if column in summary_table.columns:
                summary_table[column] = summary_table[column].map(
                    format_metric
                )

        st.dataframe(
            summary_table,
            use_container_width=True,
            hide_index=True,
        )

        if "test_r2" in best_models.columns:
            chart_data = (
                best_models[
                    ["target_label", "test_r2"]
                ]
                .dropna()
                .set_index("target_label")
            )

            if not chart_data.empty:
                st.subheader("Best Held-Out Test Performance")

                st.bar_chart(
                    chart_data,
                    horizontal=True,
                    x_label="Test R²",
                    y_label="Prediction target",
                )

    st.divider()

    st.subheader("How to Use This Application")

    st.markdown(
        """
        1. Open **Best Models** to review the strongest model for each
           symptom target.
        2. Open **Interactive Comparison** to select a target and inspect
           a particular predictor/model combination.
        3. Open **Findings & Outcomes** for interpretation, limitations,
           and future improvements.
        """
    )


# =====================================================
# Tab 2: Best models
# =====================================================

with best_tab:
    st.header("Best Performing Model for Each Target")

    st.markdown(
        """
        The table below summarizes the strongest model selected for each
        prediction target. Cross-validation performance should be considered
        alongside held-out test performance when judging model reliability.
        """
    )

    if best_models.empty:
        st.warning(
            "Best-model information is unavailable."
        )
    else:
        st.dataframe(
            build_metric_table(best_models),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        selected_best_target = st.selectbox(
            "View details for a target",
            options=best_models["target_key"].tolist(),
            format_func=lambda value: TARGET_LABELS.get(
                value,
                value,
            ),
            key="best_target_selector",
        )

        selected_best = best_models[
            best_models["target_key"] == selected_best_target
        ].iloc[0]

        detail_col1, detail_col2, detail_col3 = st.columns(3)

        with detail_col1:
            st.metric(
                "Best Model",
                selected_best["model_label"],
            )

        with detail_col2:
            st.metric(
                "Test R²",
                format_metric(
                    selected_best.get("test_r2")
                ),
            )

        with detail_col3:
            st.metric(
                "Mean CV R²",
                format_metric(
                    selected_best.get("cv_r2_mean")
                ),
            )

        if {
            "test_mae",
            "test_rmse",
        }.intersection(selected_best.index):
            error_col1, error_col2, error_col3 = st.columns(3)

            with error_col1:
                st.metric(
                    "Test MAE",
                    format_metric(
                        selected_best.get("test_mae")
                    ),
                )

            with error_col2:
                st.metric(
                    "Test RMSE",
                    format_metric(
                        selected_best.get("test_rmse")
                    ),
                )

            with error_col3:
                feature_value = selected_best.get("n_features")

                st.metric(
                    "Features Used",
                    (
                        "—"
                        if pd.isna(feature_value)
                        else f"{int(feature_value):,}"
                    ),
                )

        st.caption(
            """
            R² measures the proportion of outcome variation explained by the
            model. MAE and RMSE measure prediction error, so lower values are
            better.
            """
        )


# =====================================================
# Tab 3: Interactive comparison
# =====================================================

with comparison_tab:
    st.header("Interactive Model Comparison")

    st.markdown(
        """
        Select a symptom target and a model configuration. The selected model
        will be shown alongside the other available models for the same target.
        """
    )

    if metrics.empty:
        st.warning(
            "Model-comparison results are unavailable."
        )
    else:
        available_targets = [
            target
            for target in TARGET_LABELS
            if target in metrics["target_key"].unique()
        ]

        if not available_targets:
            available_targets = sorted(
                metrics["target_key"].dropna().unique()
            )

        selector_col1, selector_col2 = st.columns(2)

        with selector_col1:
            selected_target = st.selectbox(
                "Prediction target",
                options=available_targets,
                format_func=lambda value: TARGET_LABELS.get(
                    value,
                    value,
                ),
            )

        target_models = metrics[
            metrics["target_key"] == selected_target
        ].copy()

        available_model_keys = (
            target_models[
                ["model_key", "model_label"]
            ]
            .drop_duplicates()
            .sort_values("model_label")
        )

        model_label_lookup = dict(
            zip(
                available_model_keys["model_key"],
                available_model_keys["model_label"],
            )
        )

        with selector_col2:
            selected_model = st.selectbox(
                "Predictor and model configuration",
                options=available_model_keys[
                    "model_key"
                ].tolist(),
                format_func=lambda value: model_label_lookup.get(
                    value,
                    value,
                ),
            )

        selected_rows = target_models[
            target_models["model_key"] == selected_model
        ]

        if selected_rows.empty:
            st.warning(
                "No result was found for this selection."
            )
        else:
            selected_result = selected_rows.iloc[0]

            st.divider()

            st.subheader(
                f"{TARGET_LABELS.get(selected_target, selected_target)}: "
                f"{selected_result['model_label']}"
            )

            metric_col1, metric_col2, metric_col3, metric_col4 = (
                st.columns(4)
            )

            with metric_col1:
                st.metric(
                    "Test R²",
                    format_metric(
                        selected_result.get("test_r2")
                    ),
                )

            with metric_col2:
                st.metric(
                    "CV R² Mean",
                    format_metric(
                        selected_result.get("cv_r2_mean")
                    ),
                )

            with metric_col3:
                st.metric(
                    "Test MAE",
                    format_metric(
                        selected_result.get("test_mae")
                    ),
                )

            with metric_col4:
                st.metric(
                    "Test RMSE",
                    format_metric(
                        selected_result.get("test_rmse")
                    ),
                )

            if (
                "cv_r2_std" in selected_result.index
                and not pd.isna(
                    selected_result.get("cv_r2_std")
                )
            ):
                st.caption(
                    "Cross-validation R² standard deviation: "
                    f"{format_metric(selected_result['cv_r2_std'])}"
                )

            st.divider()

            comparison_metric_options = {}

            if "test_r2" in target_models.columns:
                comparison_metric_options[
                    "Held-Out Test R²"
                ] = "test_r2"

            if "cv_r2_mean" in target_models.columns:
                comparison_metric_options[
                    "Mean Cross-Validation R²"
                ] = "cv_r2_mean"

            if "test_mae" in target_models.columns:
                comparison_metric_options[
                    "Held-Out Test MAE"
                ] = "test_mae"

            if "test_rmse" in target_models.columns:
                comparison_metric_options[
                    "Held-Out Test RMSE"
                ] = "test_rmse"

            if comparison_metric_options:
                selected_metric_label = st.selectbox(
                    "Comparison metric",
                    options=list(
                        comparison_metric_options.keys()
                    ),
                )

                selected_metric = comparison_metric_options[
                    selected_metric_label
                ]

                chart_data = (
                    target_models[
                        ["model_label", selected_metric]
                    ]
                    .dropna()
                    .drop_duplicates(
                        subset=["model_label"]
                    )
                    .sort_values(
                        selected_metric,
                        ascending=False,
                    )
                    .set_index("model_label")
                )

                st.subheader(
                    f"{selected_metric_label} Across Models"
                )

                st.bar_chart(
                    chart_data,
                    horizontal=True,
                    x_label=selected_metric_label,
                    y_label="Model",
                )

                if selected_metric in {
                    "test_mae",
                    "test_rmse",
                }:
                    st.caption(
                        "Lower values indicate lower prediction error."
                    )
                else:
                    st.caption(
                        "Higher R² values indicate stronger predictive "
                        "performance."
                    )

            st.divider()

            st.subheader("All Models for This Target")

            st.dataframe(
                build_metric_table(target_models),
                use_container_width=True,
                hide_index=True,
            )

# =====================================================
# Tab 4: Key visualizations
# =====================================================

with visualizations_tab:
    st.header("Key Results Visualizations")

    st.markdown(
        """
        These visualizations summarize the project's main findings across
        behavioral, imaging-only, and multimodal prediction models.
        """
    )

    if metrics.empty:
        st.warning(
            "Model results are unavailable. Run the training and evaluation "
            "scripts before viewing the visualizations."
        )

    else:
        # -------------------------------------------------
        # Visualization 1: Best performance by target
        # -------------------------------------------------
        st.subheader("Best Model Performance by Target")

        st.markdown(
            """
            This chart shows the held-out test R² for the best-performing
            model for each symptom target. Higher values indicate that the
            model explains more variation in symptom severity.
            """
        )

        if (
            not best_models.empty
            and "test_r2" in best_models.columns
        ):
            best_chart = (
                best_models[
                    ["target_label", "test_r2"]
                ]
                .dropna()
                .sort_values("test_r2")
                .set_index("target_label")
            )

            st.bar_chart(
                best_chart,
                horizontal=True,
                x_label="Held-out test R²",
                y_label="Symptom target",
                use_container_width=True,
            )

            st.caption(
                """
                Anxiety had the strongest predictive performance, while
                anhedonia was the most difficult outcome to predict.
                """
            )

        else:
            st.info(
                "Best-model test R² results are unavailable."
            )

        st.divider()

        # -------------------------------------------------
        # Visualization 2: Modalities across targets
        # -------------------------------------------------
        st.subheader("Behavioral, Imaging, and Multimodal Performance")

        st.markdown(
            """
            Select a performance metric to compare the three major sources
            of predictor data across all four symptom targets.
            """
        )

        modality_metric_options = {}

        if "cv_r2_mean" in metrics.columns:
            modality_metric_options[
                "Mean cross-validation R²"
            ] = "cv_r2_mean"

        if "test_r2" in metrics.columns:
            modality_metric_options[
                "Held-out test R²"
            ] = "test_r2"

        if "test_mae" in metrics.columns:
            modality_metric_options[
                "Held-out test MAE"
            ] = "test_mae"

        selected_modality_metric_label = st.selectbox(
            "Performance metric",
            options=list(modality_metric_options.keys()),
            key="visualization_modality_metric",
        )

        selected_modality_metric = modality_metric_options[
            selected_modality_metric_label
        ]

        modality_data = metrics.copy()

        def get_modality_label(row):
            model_key = str(row.get("model_key", ""))

            if model_key == "imaging_only":
                return "Imaging Only"

            if model_key.startswith("behavioral_"):
                return "Behavioral"

            if model_key.startswith("multimodal_"):
                return "Multimodal"

            return "Other"

        modality_data["modality_group"] = modality_data.apply(
            get_modality_label,
            axis=1,
        )

        modality_data = modality_data[
            modality_data["modality_group"].isin(
                ["Behavioral", "Imaging Only", "Multimodal"]
            )
        ]

        # Keep the strongest model within each broad modality for each target.
        if selected_modality_metric in {
            "test_mae",
            "test_rmse",
            "cv_mae_mean",
            "cv_rmse_mean",
        }:
            modality_summary = (
                modality_data.groupby(
                    ["target_label", "modality_group"],
                    as_index=False,
                )[selected_modality_metric]
                .min()
            )
        else:
            modality_summary = (
                modality_data.groupby(
                    ["target_label", "modality_group"],
                    as_index=False,
                )[selected_modality_metric]
                .max()
            )

        modality_pivot = modality_summary.pivot(
            index="target_label",
            columns="modality_group",
            values=selected_modality_metric,
        )

        preferred_order = [
            "Behavioral",
            "Imaging Only",
            "Multimodal",
        ]

        modality_pivot = modality_pivot[
            [
                column
                for column in preferred_order
                if column in modality_pivot.columns
            ]
        ]

        st.bar_chart(
            modality_pivot,
            x_label="Symptom target",
            y_label=selected_modality_metric_label,
            use_container_width=True,
        )

        if selected_modality_metric in {
            "test_mae",
            "test_rmse",
        }:
            st.caption(
                """
                Lower values indicate better performance. Behavioral models
                generally produced the lowest prediction errors.
                """
            )
        else:
            st.caption(
                """
                Higher values indicate better performance. Behavioral
                models consistently outperformed imaging-only models, and
                adding imaging generally provided little benefit.
                """
            )

        st.divider()

        # -------------------------------------------------
        # Visualization 3: CV versus test performance
        # -------------------------------------------------
        st.subheader("Cross-Validation vs. Held-Out Test Performance")

        st.markdown(
            """
            Comparing cross-validation and held-out test scores helps identify
            models whose apparent performance may depend heavily on a single
            train/test split.
            """
        )

        selected_cv_target = st.selectbox(
            "Choose a symptom target",
            options=sorted(
                metrics["target_key"].dropna().unique()
            ),
            format_func=lambda value: TARGET_LABELS.get(
                value,
                value,
            ),
            key="visualization_cv_target",
        )

        cv_test_data = metrics[
            metrics["target_key"] == selected_cv_target
        ].copy()

        required_cv_columns = {
            "model_label",
            "cv_r2_mean",
            "test_r2",
        }

        if required_cv_columns.issubset(cv_test_data.columns):
            cv_test_chart = (
                cv_test_data[
                    [
                        "model_label",
                        "cv_r2_mean",
                        "test_r2",
                    ]
                ]
                .dropna()
                .drop_duplicates(subset=["model_label"])
                .rename(
                    columns={
                        "cv_r2_mean": "Mean CV R²",
                        "test_r2": "Test R²",
                    }
                )
                .set_index("model_label")
            )

            st.bar_chart(
                cv_test_chart,
                horizontal=True,
                x_label="R²",
                y_label="Model",
                use_container_width=True,
            )

            st.caption(
                """
                Large gaps between cross-validation and test R² can indicate
                that the held-out result is unstable or unusually favorable.
                Model selection should therefore prioritize repeated
                cross-validation performance.
                """
            )

        else:
            st.info(
                "Both cross-validation and test R² values are required "
                "for this chart."
            )

        st.divider()

        # -------------------------------------------------
        # Visualization 4: Behavioral feature scopes
        # -------------------------------------------------
        st.subheader("Behavioral Feature-Set Comparison")

        st.markdown(
            """
            This chart compares the full behavioral dataset with the smaller
            relevant and same-category behavioral feature sets.
            """
        )

        selected_behavior_target = st.selectbox(
            "Choose a target for behavioral comparison",
            options=sorted(
                metrics["target_key"].dropna().unique()
            ),
            format_func=lambda value: TARGET_LABELS.get(
                value,
                value,
            ),
            key="visualization_behavior_target",
        )

        behavioral_rows = metrics[
            (
                metrics["target_key"]
                == selected_behavior_target
            )
            & (
                metrics["model_key"].str.startswith(
                    "behavioral_",
                    na=False,
                )
            )
        ].copy()

        behavioral_metric_options = {}

        if "cv_r2_mean" in behavioral_rows.columns:
            behavioral_metric_options[
                "Mean cross-validation R²"
            ] = "cv_r2_mean"

        if "test_r2" in behavioral_rows.columns:
            behavioral_metric_options[
                "Held-out test R²"
            ] = "test_r2"

        selected_behavior_metric_label = st.selectbox(
            "Behavioral comparison metric",
            options=list(behavioral_metric_options.keys()),
            key="visualization_behavior_metric",
        )

        selected_behavior_metric = behavioral_metric_options[
            selected_behavior_metric_label
        ]

        behavioral_chart = (
            behavioral_rows[
                ["model_label", selected_behavior_metric]
            ]
            .dropna()
            .drop_duplicates(subset=["model_label"])
            .sort_values(selected_behavior_metric)
            .set_index("model_label")
        )

        st.bar_chart(
            behavioral_chart,
            horizontal=True,
            x_label=selected_behavior_metric_label,
            y_label="Behavioral feature set",
            use_container_width=True,
        )

        st.caption(
            """
            Smaller relevant or category-specific feature sets sometimes
            performed similarly to the full behavioral dataset, showing that
            additional features do not automatically improve prediction.
            """
        )

        st.divider()

        # -------------------------------------------------
        # Visualization 5: Main interpretation
        # -------------------------------------------------
        st.subheader("What the Results Show")

        finding_col1, finding_col2, finding_col3 = st.columns(3)

        with finding_col1:
            st.success(
                """
                **Behavioral signal was strongest**

                Self-report questionnaire features provided the most useful
                information for predicting symptom severity.
                """
            )

        with finding_col2:
            st.info(
                """
                **Imaging signal was limited**

                The extracted resting-state fMRI features did not produce
                strong symptom predictions on their own.
                """
            )

        with finding_col3:
            st.warning(
                """
                **More data was not always better**

                Multimodal and full-feature models often failed to outperform
                smaller behavioral models.
                """
            )

# -----------------------------------------------------
# Target and Model Guide
# -----------------------------------------------------
st.subheader("Understanding the Targets and Models")

with st.expander("What do the targets and model types mean?"):
    st.markdown("""
    ### Prediction Targets

    - **QIDS — Depression:**  
      Predicts depression symptom severity using the Quick Inventory of
      Depressive Symptomatology score.

    - **STAI-State — Anxiety:**  
      Predicts current anxiety severity using the State-Trait Anxiety
      Inventory state score.

    - **PSS — Perceived Stress:**  
      Predicts how much stress a participant reports experiencing.

    - **SHAPS — Anhedonia:**  
      Predicts reduced ability to experience pleasure or enjoyment.

    ### Model Types

    - **All Behavioral:**  
      Uses the full set of eligible behavioral questionnaire features,
      excluding the target questionnaire itself.

    - **Relevant Behavioral:**  
      Uses behavioral questionnaires from symptom domains considered
      clinically relevant to the selected target.

    - **Categorical Behavioral:**  
      Uses only questionnaires assigned to the same symptom category as
      the prediction target. For example, a depression-category model uses
      depression-related questionnaires to predict QIDS.

    - **Imaging Only:**  
      Uses features extracted from resting-state fMRI data, including
      connectivity summaries, graph features, regional parcel-strength
      features, and PCA components.

    - **Multimodal All:**  
      Combines imaging features with the full behavioral feature set.

    - **Multimodal Relevant:**  
      Combines imaging features with the relevant behavioral feature set.

    - **Multimodal Category:**  
      Combines imaging features with behavioral features from the target's
      symptom category.
    """)

# =====================================================
# Tab 5: Findings and outcomes
# =====================================================

with findings_tab:
    st.header("Key Findings, Outcomes, and Lessons Learned")

    st.subheader("Key Findings")

    st.markdown(
        """
        - **Behavioral questionnaire data consistently produced the strongest
          predictions** across depression, anxiety, stress, and anhedonia.
        - **Imaging-only models had limited predictive performance**, with
          R² values often near zero or negative.
        - **Multimodal models generally did not outperform behavioral-only
          models**, suggesting that the extracted imaging features added
          limited predictive information.
        - Anxiety was the most accurately predicted target, while anhedonia
          was the most difficult and least stable target to predict.
        """
    )

    st.subheader("Project Outcomes")

    st.markdown(
        """
        - Developed a reproducible pipeline for preprocessing behavioral and
          resting-state fMRI data.
        - Extracted global, graph-based, and regional imaging features.
        - Compared behavioral-only, imaging-only, and multimodal Random Forest
          regression models.
        - Evaluated broad, relevant, and symptom-category-specific behavioral
          feature sets.
        - Created an interactive application for reviewing and comparing the
          model results.
        """
    )

    st.subheader("Lessons Learned")

    st.markdown(
        """
        - More features do not automatically lead to better predictions.
        - Data quality, feature engineering, and leakage prevention are as
          important as the machine-learning algorithm.
        - Cross-validation is necessary because a single test split can give
          unstable or misleading results.
        - Simpler behavioral models can perform as well as larger multimodal
          models while remaining easier to interpret.
        """
    )

    st.divider()

    limitation_col, future_col = st.columns(2)

    with limitation_col:
        st.subheader("Limitations")

        st.markdown(
            """
            - Relatively small sample size for high-dimensional imaging data
            - Use of a single dataset
            - Possible scanner-site and head-motion effects
            - Limited predictive signal from the selected imaging features
            - Evaluation focused on Random Forest regression
            - Results may not generalize to clinical diagnosis or other
              populations
            """
        )

    with future_col:
        st.subheader("Future Improvements")

        st.markdown(
            """
            - Compare Random Forest with regularized linear models and
              gradient-boosting methods
            - Improve imaging feature engineering and network-level features
            - Add motion correction and site harmonization analyses
            - Evaluate models using external datasets
            - Increase the participant sample
            - Add feature-importance and prediction-error visualizations
            - Examine subgroup performance and model fairness
            """
        )

    st.divider()

    st.warning(
        """
        **Responsible use:** This project demonstrates research-oriented
        symptom-severity prediction. It should not be used to diagnose,
        treat, or make clinical decisions about an individual.
        """
    )


# =====================================================
# Footer
# =====================================================

st.divider()

st.caption(
    """
    Developed using Python, Streamlit, pandas, and scikit-learn with data
    from the Transdiagnostic Connectome Project.
    """
)