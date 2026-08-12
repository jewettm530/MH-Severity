"""Refresh GitHub Pages image assets from the latest pipeline outputs.

Run after:
    python3 scripts/08_validate_outputs.py
    python3 scripts/09_create_results_visualizations.py

Then commit the updated docs/images files.
"""

from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_IMAGES = PROJECT_ROOT / "docs" / "images"

ASSETS = {
    PROJECT_ROOT / "outputs/results/figures/model_performance_by_modality.png":
        DOCS_IMAGES / "model-performance-by-modality.png",
    PROJECT_ROOT / "outputs/results/figures/spearman_feature_outcome_correlation_matrix.png":
        DOCS_IMAGES / "spearman-correlation-matrix.png",
    PROJECT_ROOT / "outputs/results/figures/qids_cross_category_contributions.png":
        DOCS_IMAGES / "qids-cross-category.png",
    PROJECT_ROOT / "outputs/results/figures/stai_cross_category_contributions.png":
        DOCS_IMAGES / "stai-cross-category.png",
    PROJECT_ROOT / "outputs/results/figures/pss_cross_category_contributions.png":
        DOCS_IMAGES / "pss-cross-category.png",
    PROJECT_ROOT / "outputs/results/figures/shaps_cross_category_contributions.png":
        DOCS_IMAGES / "shaps-cross-category.png",
    PROJECT_ROOT / "outputs/demos_visualizations/group_distribution.png":
        DOCS_IMAGES / "participant-groups.png",
    PROJECT_ROOT / "outputs/demos_visualizations/age_distribution.png":
        DOCS_IMAGES / "age-distribution.png",
}


def main() -> None:
    DOCS_IMAGES.mkdir(parents=True, exist_ok=True)

    missing = []
    copied = []

    for source, destination in ASSETS.items():
        if not source.exists():
            missing.append(source)
            continue

        shutil.copy2(source, destination)
        copied.append(destination)

    for path in copied:
        print(f"Updated {path.relative_to(PROJECT_ROOT)}")

    if missing:
        print("\nMissing source figures:")
        for path in missing:
            print(f"  {path.relative_to(PROJECT_ROOT)}")
        raise SystemExit(
            "\nGenerate the missing figures before publishing GitHub Pages."
        )

    print("\nGitHub Pages visual assets are up to date.")


if __name__ == "__main__":
    main()
