"""Create presentation-ready visualizations from model and validation outputs."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import OUTPUT_DIR, RESULTS_DIR


FIGURE_DIR = RESULTS_DIR / "figures"
VALIDATION_DIR = OUTPUT_DIR / "validation"

MODEL_METRICS_FILE = RESULTS_DIR / "model_metrics_comparison.csv"
CROSS_CATEGORY_METRICS_FILE = RESULTS_DIR / "cross_category_metrics.csv"
SPEARMAN_FILE = (
    VALIDATION_DIR / "spearman_feature_outcome_correlations.csv"
)

TARGET_ORDER = ["qids", "stai", "pss", "shaps"]

TARGET_LABELS = {
    "qids": "Depression (QIDS)",
    "stai": "Anxiety (STAI-State)",
    "pss": "Stress (PSS)",
    "shaps": "Anhedonia (SHAPS)",
}

BASE_CATEGORY_LABELS = {
    "qids": "Depression",
    "stai": "Anxiety",
    "pss": "Stress",
    "shaps": "Anhedonia",
}


def require_columns(
    dataframe: pd.DataFrame,
    required: set[str],
    source_name: str,
) -> bool:
    """Print a useful warning when required columns are unavailable."""
    missing = required.difference(dataframe.columns)

    if missing:
        print(
            f"Skipping {source_name}. Missing columns: "
            f"{sorted(missing)}"
        )
        return False

    return True


def filter_successful_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    """Keep successful model rows when a status column is available."""
    if "status" not in metrics.columns:
        return metrics.copy()

    status = (
        metrics["status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return metrics[status == "ok"].copy()


def model_label(row: pd.Series) -> str:
    """Create a readable label for one model configuration."""
    input_type = str(row.get("input_type", "")).strip().lower()
    behavior_scope = str(
        row.get("behavior_scope", "none")
    ).strip().lower()

    if input_type == "imaging_only":
        return "Imaging only"

    scope_labels = {
        "all": "All",
        "relevant": "Relevant",
        "category": "Category",
        "none": "",
    }

    scope = scope_labels.get(
        behavior_scope,
        behavior_scope.replace("_", " ").title(),
    )

    if input_type == "behavioral_only":
        return f"Behavioral: {scope}".strip(": ")

    if input_type == "multimodal":
        return f"Multimodal: {scope}".strip(": ")

    return (
        str(row.get("experiment_name", input_type))
        .replace("_", " ")
        .title()
    )


def model_performance(metrics: pd.DataFrame) -> None:
    """Create one detailed model-comparison chart per target."""
    required = {
        "target_key",
        "input_type",
        "behavior_scope",
        "cv_R2_mean",
    }

    if not require_columns(
        metrics,
        required,
        "per-target model comparison",
    ):
        return

    for target in TARGET_ORDER:
        group = metrics[
            metrics["target_key"] == target
        ].copy()

        if group.empty:
            continue

        group["cv_R2_mean"] = pd.to_numeric(
            group["cv_R2_mean"],
            errors="coerce",
        )

        group = group.dropna(subset=["cv_R2_mean"])

        if group.empty:
            continue

        group["label"] = group.apply(model_label, axis=1)
        group = group.sort_values("cv_R2_mean")

        figure, axis = plt.subplots(figsize=(10, 6))

        bars = axis.barh(
            group["label"],
            group["cv_R2_mean"],
        )

        axis.axvline(0, linewidth=1)
        axis.set_xlabel("Mean repeated cross-validation R²")
        axis.set_ylabel("Model configuration")
        axis.set_title(
            f"{TARGET_LABELS.get(target, target.upper())}: "
            "model comparison"
        )

        for bar, value in zip(bars, group["cv_R2_mean"]):
            offset = 0.008 if value >= 0 else -0.008

            axis.text(
                value + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=8,
            )

        figure.tight_layout()

        output_path = (
            FIGURE_DIR
            / f"{target}_model_comparison_cv_r2.png"
        )

        figure.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(figure)
        print(f"Saved {output_path}")


def cv_vs_test(metrics: pd.DataFrame) -> None:
    """Compare repeated-CV and held-out test R² across all models."""
    required = {
        "target_key",
        "cv_R2_mean",
        "test_R2",
    }

    if not require_columns(
        metrics,
        required,
        "cross-validation versus test chart",
    ):
        return

    plot_data = metrics.copy()

    plot_data["cv_R2_mean"] = pd.to_numeric(
        plot_data["cv_R2_mean"],
        errors="coerce",
    )

    plot_data["test_R2"] = pd.to_numeric(
        plot_data["test_R2"],
        errors="coerce",
    )

    plot_data = plot_data.dropna(
        subset=["cv_R2_mean", "test_R2"]
    )

    if plot_data.empty:
        print("No valid rows for CV-versus-test chart.")
        return

    figure, axis = plt.subplots(figsize=(8, 7))

    for target, group in plot_data.groupby("target_key"):
        axis.scatter(
            group["cv_R2_mean"],
            group["test_R2"],
            label=TARGET_LABELS.get(
                target,
                str(target).upper(),
            ),
        )

    low = min(
        plot_data["cv_R2_mean"].min(),
        plot_data["test_R2"].min(),
    )

    high = max(
        plot_data["cv_R2_mean"].max(),
        plot_data["test_R2"].max(),
    )

    axis.plot(
        [low, high],
        [low, high],
        linestyle="--",
        linewidth=1,
    )

    axis.set_xlabel("Mean repeated-CV R²")
    axis.set_ylabel("Held-out test R²")
    axis.set_title(
        "Cross-validation versus held-out performance"
    )
    axis.legend(frameon=False)

    figure.tight_layout()

    output_path = FIGURE_DIR / "cv_vs_test_r2.png"

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)
    print(f"Saved {output_path}")


def create_model_performance_comparison(
    metrics: pd.DataFrame,
) -> None:
    """
    Compare the best behavioral, imaging-only, and multimodal model
    for each target using mean repeated cross-validation R².

    The best scope within each broad modality is retained for each
    outcome.
    """
    required = {
        "target_key",
        "input_type",
        "cv_R2_mean",
    }

    if not require_columns(
        metrics,
        required,
        "modality performance comparison",
    ):
        return

    plot_data = metrics.copy()

    def assign_modality(input_type: object) -> str:
        normalized = (
            str(input_type)
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        if normalized in {"behavioral", "behavioral_only"}:
            return "Behavioral"

        if normalized in {"imaging", "imaging_only"}:
            return "Imaging only"

        if normalized in {
            "multimodal",
            "combined",
            "behavioral_and_imaging",
        }:
            return "Multimodal"

        return "Other"

    plot_data["modality"] = (
        plot_data["input_type"].apply(assign_modality)
    )

    plot_data = plot_data[
        plot_data["modality"].isin(
            ["Behavioral", "Imaging only", "Multimodal"]
        )
    ].copy()

    plot_data["cv_R2_mean"] = pd.to_numeric(
        plot_data["cv_R2_mean"],
        errors="coerce",
    )

    plot_data = plot_data.dropna(
        subset=[
            "target_key",
            "modality",
            "cv_R2_mean",
        ]
    )

    if plot_data.empty:
        print(
            "No valid model metrics were available for "
            "the modality comparison."
        )
        return

    modality_summary = (
        plot_data.groupby(
            ["target_key", "modality"],
            as_index=False,
        )["cv_R2_mean"]
        .max()
    )

    pivot = modality_summary.pivot(
        index="target_key",
        columns="modality",
        values="cv_R2_mean",
    )

    pivot = pivot.reindex(
        [
            target
            for target in TARGET_ORDER
            if target in pivot.index
        ]
    )

    modality_order = [
        "Behavioral",
        "Imaging only",
        "Multimodal",
    ]

    pivot = pivot[
        [
            modality
            for modality in modality_order
            if modality in pivot.columns
        ]
    ]

    pivot.index = [
        TARGET_LABELS.get(target, str(target).upper())
        for target in pivot.index
    ]

    figure, axis = plt.subplots(figsize=(12, 7))

    pivot.plot(
        kind="bar",
        ax=axis,
        width=0.78,
    )

    axis.axhline(0, linewidth=1)
    axis.set_title(
        "Predictive Performance by Data Modality",
        fontsize=16,
        pad=16,
    )
    axis.set_xlabel("Mental health symptom target")
    axis.set_ylabel("Mean repeated cross-validation R²")
    axis.tick_params(axis="x", rotation=0)
    axis.legend(
        title="Predictor data",
        frameon=False,
    )

    for container in axis.containers:
        labels = []

        for bar in container:
            height = bar.get_height()

            labels.append(
                ""
                if np.isnan(height)
                else f"{height:.2f}"
            )

        axis.bar_label(
            container,
            labels=labels,
            padding=3,
            fontsize=9,
        )

    figure.text(
        0.5,
        0.01,
        (
            "Higher R² indicates stronger prediction. "
            "Negative R² means performance was worse than "
            "predicting the outcome mean."
        ),
        ha="center",
        fontsize=9,
    )

    figure.tight_layout(rect=(0, 0.04, 1, 1))

    output_path = (
        FIGURE_DIR / "model_performance_by_modality.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)
    print(f"Saved {output_path}")


def create_cross_category_contribution_chart() -> None:
    """
    Create one chart per target showing change in repeated-CV R²
    relative to the target-category-only baseline.
    """
    if not CROSS_CATEGORY_METRICS_FILE.exists():
        print(
            "Cross-category metrics were not found at "
            f"{CROSS_CATEGORY_METRICS_FILE}. "
            "Run scripts/07_cross_category_analysis.py first."
        )
        return

    data = pd.read_csv(CROSS_CATEGORY_METRICS_FILE)

    required = {
        "target_key",
        "combination_label",
        "cv_R2_mean",
    }

    if not require_columns(
        data,
        required,
        "cross-category contribution charts",
    ):
        return

    if "status" in data.columns:
        data = filter_successful_rows(data)

    data["cv_R2_mean"] = pd.to_numeric(
        data["cv_R2_mean"],
        errors="coerce",
    )

    # Prefer the delta already created by 07_cross_category_analysis.py.
    if "delta_cv_R2" in data.columns:
        data["delta_cv_R2"] = pd.to_numeric(
            data["delta_cv_R2"],
            errors="coerce",
        )
    else:
        data["delta_cv_R2"] = np.nan

        for target, group in data.groupby("target_key"):
            baseline_rows = group[
                group.get(
                    "added_categories",
                    pd.Series(
                        index=group.index,
                        dtype=object,
                    ),
                )
                .astype(str)
                .isin(["[]", "", "nan"])
            ]

            if baseline_rows.empty:
                baseline_rows = group[
                    group["combination_label"]
                    .astype(str)
                    .str.count(r"\+")
                    == 0
                ]

            if baseline_rows.empty:
                print(
                    f"No cross-category baseline found for {target}."
                )
                continue

            baseline_score = baseline_rows.iloc[0][
                "cv_R2_mean"
            ]

            data.loc[group.index, "delta_cv_R2"] = (
                group["cv_R2_mean"] - baseline_score
            )

    data = data.dropna(
        subset=[
            "target_key",
            "combination_label",
            "delta_cv_R2",
        ]
    )

    for target in TARGET_ORDER:
        target_data = data[
            data["target_key"] == target
        ].copy()

        if target_data.empty:
            continue

        # Remove the category-only baseline because its delta is zero.
        target_data = target_data[
            target_data["delta_cv_R2"].abs() > 1e-12
        ].copy()

        if target_data.empty:
            print(
                f"No non-baseline cross-category rows for {target}."
            )
            continue

        target_data["display_combination"] = (
            target_data["combination_label"]
            .astype(str)
            .str.replace("_", " ", regex=False)
            .str.replace("+", " + ", regex=False)
        )

        target_data = target_data.sort_values(
            "delta_cv_R2",
            ascending=True,
        )

        figure_height = max(
            5,
            0.55 * len(target_data) + 2,
        )

        figure, axis = plt.subplots(
            figsize=(11, figure_height)
        )

        bars = axis.barh(
            target_data["display_combination"],
            target_data["delta_cv_R2"],
        )

        axis.axvline(0, linewidth=1)
        axis.set_title(
            (
                "Change in Predictive Performance from Adding "
                f"Symptom Categories\n"
                f"{TARGET_LABELS.get(target, target.upper())}"
            ),
            fontsize=15,
            pad=14,
        )
        axis.set_xlabel(
            "Change in mean repeated cross-validation R² "
            "from baseline"
        )
        axis.set_ylabel("Predictor categories")

        for bar, value in zip(
            bars,
            target_data["delta_cv_R2"],
        ):
            offset = 0.005 if value >= 0 else -0.005

            axis.text(
                value + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{value:+.3f}",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=9,
            )

        figure.text(
            0.5,
            0.01,
            (
                "Positive values indicate improvement over the "
                f"{BASE_CATEGORY_LABELS.get(target, target).lower()}"
                "-only baseline."
            ),
            ha="center",
            fontsize=9,
        )

        figure.tight_layout(rect=(0, 0.04, 1, 1))

        output_path = (
            FIGURE_DIR
            / f"{target}_cross_category_contributions.png"
        )

        figure.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(figure)
        print(f"Saved {output_path}")


def top_correlations() -> None:
    """Create one top-feature Spearman chart per outcome."""
    if not SPEARMAN_FILE.exists():
        print(
            "Spearman correlation file not found at "
            f"{SPEARMAN_FILE}. "
            "Run scripts/08_validate_outputs.py first."
        )
        return

    correlations = pd.read_csv(SPEARMAN_FILE)

    required = {
        "target",
        "feature",
        "spearman_rho",
        "abs_spearman_rho",
    }

    if not require_columns(
        correlations,
        required,
        "top Spearman correlation charts",
    ):
        return

    for target, group in correlations.groupby("target"):
        top = (
            group.nlargest(12, "abs_spearman_rho")
            .sort_values("spearman_rho")
        )

        if top.empty:
            continue

        figure, axis = plt.subplots(figsize=(10, 7))

        axis.barh(
            top["feature"].str.replace(
                "__",
                ": ",
                regex=False,
            ),
            top["spearman_rho"],
        )

        axis.axvline(-0.8, linestyle="--", linewidth=1)
        axis.axvline(0.8, linestyle="--", linewidth=1)
        axis.set_xlabel("Spearman correlation with outcome")
        axis.set_ylabel("Behavioral feature")
        axis.set_title(
            "Strongest Behavioral Associations with "
            f"{TARGET_LABELS.get(target, str(target).upper())}"
        )

        figure.tight_layout()

        output_path = (
            FIGURE_DIR
            / f"{target}_top_spearman_correlations.png"
        )

        figure.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(figure)
        print(f"Saved {output_path}")


def spearman_correlation_matrix() -> None:
    """
    Create a heatmap of the strongest feature-to-outcome
    Spearman correlations.
    """
    if not SPEARMAN_FILE.exists():
        print(
            "Spearman correlation file not found at "
            f"{SPEARMAN_FILE}. "
            "Run scripts/08_validate_outputs.py first."
        )
        return

    correlations = pd.read_csv(SPEARMAN_FILE)

    required = {
        "target",
        "feature",
        "spearman_rho",
        "abs_spearman_rho",
    }

    if not require_columns(
        correlations,
        required,
        "Spearman correlation matrix",
    ):
        return

    if correlations.empty:
        print("No Spearman correlations were available.")
        return

    top_features = (
        correlations.sort_values(
            "abs_spearman_rho",
            ascending=False,
        )
        .groupby("target")
        .head(8)["feature"]
        .drop_duplicates()
        .tolist()
    )

    matrix = (
        correlations[
            correlations["feature"].isin(top_features)
        ]
        .pivot_table(
            index="feature",
            columns="target",
            values="spearman_rho",
            aggfunc="first",
        )
        .reindex(columns=TARGET_ORDER)
    )

    if matrix.empty:
        print("No data were available for the Spearman matrix.")
        return

    matrix["max_abs"] = matrix.abs().max(axis=1)

    matrix = (
        matrix.sort_values("max_abs", ascending=True)
        .drop(columns="max_abs")
    )

    display_labels = (
        matrix.index
        .str.replace("__", ": ", regex=False)
        .str.replace("_", " ", regex=False)
    )

    figure_height = max(8, len(matrix) * 0.35)

    figure, axis = plt.subplots(
        figsize=(10, figure_height)
    )

    masked_values = np.ma.masked_invalid(matrix.values)

    image = axis.imshow(
        masked_values,
        aspect="auto",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )

    axis.set_xticks(range(len(matrix.columns)))
    axis.set_xticklabels(
        [
            TARGET_LABELS.get(column, column.upper())
            for column in matrix.columns
        ],
        rotation=15,
        ha="right",
    )

    axis.set_yticks(range(len(matrix.index)))
    axis.set_yticklabels(display_labels)
    axis.set_xlabel("Prediction outcome")
    axis.set_ylabel("Behavioral feature")
    axis.set_title(
        "Strongest Feature-to-Outcome Spearman Correlations"
    )

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix.iloc[
                row_index,
                column_index,
            ]

            if pd.notna(value):
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Spearman ρ")

    figure.tight_layout()

    output_path = (
        FIGURE_DIR
        / "spearman_feature_outcome_correlation_matrix.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)
    print(f"Saved {output_path}")


def main() -> None:
    """Generate all model and validation visualizations."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_METRICS_FILE.exists():
        raise FileNotFoundError(
            "Model metrics were not found at "
            f"{MODEL_METRICS_FILE}. "
            "Run scripts/05_train_random_forest.py first."
        )

    metrics = pd.read_csv(MODEL_METRICS_FILE)
    metrics = filter_successful_rows(metrics)

    if metrics.empty:
        raise ValueError(
            "No successful model rows were available in "
            f"{MODEL_METRICS_FILE}."
        )

    model_performance(metrics)
    cv_vs_test(metrics)
    create_model_performance_comparison(metrics)
    create_cross_category_contribution_chart()
    top_correlations()
    spearman_correlation_matrix()

    print(f"Saved figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()