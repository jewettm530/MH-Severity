"""Create presentation-ready visualizations from model and validation outputs."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from config import OUTPUT_DIR, RESULTS_DIR

FIGURE_DIR = RESULTS_DIR / "figures"
VALIDATION_DIR = OUTPUT_DIR / "validation"


def model_label(row: pd.Series) -> str:
    if row["input_type"] == "imaging_only":
        return "Imaging only"
    scope = str(row["behavior_scope"]).title()
    modality = "Behavioral" if row["input_type"] == "behavioral_only" else "Multimodal"
    return f"{modality}: {scope}"


def model_performance(metrics: pd.DataFrame) -> None:
    targets = ["qids", "stai", "pss", "shaps"]
    for target in targets:
        group = metrics[metrics["target_key"] == target].copy()
        if group.empty:
            continue
        group["label"] = group.apply(model_label, axis=1)
        group = group.sort_values("cv_R2_mean")
        plt.figure(figsize=(10, 6))
        plt.barh(group["label"], group["cv_R2_mean"])
        plt.axvline(0, linewidth=1)
        plt.xlabel("Mean repeated cross-validation R²")
        plt.title(f"{target.upper()} model comparison")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"{target}_model_comparison_cv_r2.png", dpi=220)
        plt.close()


def cv_vs_test(metrics: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 7))
    for target, group in metrics.groupby("target_key"):
        plt.scatter(group["cv_R2_mean"], group["test_R2"], label=target.upper())
    low = min(metrics["cv_R2_mean"].min(), metrics["test_R2"].min())
    high = max(metrics["cv_R2_mean"].max(), metrics["test_R2"].max())
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Mean repeated-CV R²")
    plt.ylabel("Held-out test R²")
    plt.title("Cross-validation versus held-out performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "cv_vs_test_r2.png", dpi=220)
    plt.close()


def top_correlations() -> None:
    path = VALIDATION_DIR / "spearman_feature_outcome_correlations.csv"
    if not path.exists():
        return
    corr = pd.read_csv(path)
    for target, group in corr.groupby("target"):
        top = group.nlargest(12, "abs_spearman_rho").sort_values("spearman_rho")
        plt.figure(figsize=(10, 7))
        plt.barh(top["feature"].str.replace("__", ": ", regex=False), top["spearman_rho"])
        plt.axvline(-0.8, linestyle="--", linewidth=1)
        plt.axvline(0.8, linestyle="--", linewidth=1)
        plt.xlabel("Spearman correlation with outcome")
        plt.title(f"Strongest behavioral associations with {target.upper()}")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"{target}_top_spearman_correlations.png", dpi=220)
        plt.close()

def spearman_correlation_matrix() -> None:
    """
    Create a heatmap of the strongest feature-to-outcome Spearman
    correlations across all four prediction targets.
    """
    path = VALIDATION_DIR / "spearman_feature_outcome_correlations.csv"

    if not path.exists():
        print(
            "Spearman correlation file not found. "
            "Run scripts/08_validate_outputs.py first."
        )
        return

    correlations = pd.read_csv(path)

    if correlations.empty:
        print("No Spearman correlations were available.")
        return

    # Select the strongest features for each outcome.
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
        .reindex(columns=["qids", "stai", "pss", "shaps"])
    )

    # Order features by their strongest absolute association.
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

    fig, ax = plt.subplots(
        figsize=(10, figure_height)
    )

    image = ax.imshow(
        matrix.fillna(0).values,
        aspect="auto",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(
        [column.upper() for column in matrix.columns]
    )

    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(display_labels)

    ax.set_xlabel("Prediction outcome")
    ax.set_ylabel("Behavioral feature")
    ax.set_title(
        "Strongest Feature-to-Outcome Spearman Correlations"
    )

    # Add correlation values inside cells.
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix.iloc[row_index, column_index]

            if pd.notna(value):
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Spearman ρ")

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / "spearman_feature_outcome_correlation_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved {output_path}")

def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(RESULTS_DIR / "model_metrics_comparison.csv")
    metrics = metrics[metrics["status"] == "ok"].copy()
    model_performance(metrics)
    cv_vs_test(metrics)
    top_correlations()
    spearman_correlation_matrix()
    print(f"Saved figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
