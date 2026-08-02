"""Summarize modality performance and save feature importance outputs."""
import matplotlib
matplotlib.use("Agg")
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from config import EXPERIMENT_DIR, RESULTS_DIR, MODELS_DIR, IMPORTANCE_DIR
from model_utils import clean_X


def transformed_feature_names(pipeline):
    preprocess = pipeline.named_steps['preprocess']
    try:
        return list(preprocess.get_feature_names_out())
    except Exception:
        # Fallback names preserve alignment with feature_importances_ even if
        # an older sklearn version cannot propagate names through FeatureUnion.
        n = pipeline.named_steps['model'].n_features_in_
        return [f"transformed_feature_{i:03d}" for i in range(n)]


def save_importance(experiment_name):
    model_path = MODELS_DIR / f'{experiment_name}.joblib'
    data_path = EXPERIMENT_DIR / f'{experiment_name}.csv'
    if not model_path.exists() or not data_path.exists():
        return
    pipeline = joblib.load(model_path)
    X = clean_X(pd.read_csv(data_path, low_memory=False))
    names = transformed_feature_names(pipeline)
    importance = pipeline.named_steps['model'].feature_importances_
    n = min(len(names), len(importance))
    output = pd.DataFrame({'feature': names[:n], 'importance': importance[:n]})
    output = output.sort_values('importance', ascending=False)
    IMPORTANCE_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(IMPORTANCE_DIR / f'{experiment_name}.csv', index=False)


def main():
    metrics = pd.read_csv(RESULTS_DIR / 'model_metrics_comparison.csv')
    metrics = metrics[metrics['status'] == 'ok'].copy()
    metrics = metrics.sort_values(['target_key', 'cv_R2_mean'], ascending=[True, False])
    metrics.to_csv(RESULTS_DIR / 'model_metrics_comparison_sorted.csv', index=False)

    best = (
    metrics.sort_values(['target_key', 'cv_R2_mean'], ascending=[True, False])
    .groupby('target_key', as_index=False)
    .head(1)
    .reset_index(drop=True)
    )
    best.to_csv(RESULTS_DIR / 'best_model_by_target.csv', index=False)

    modality = metrics.groupby(['target_key', 'input_type', 'behavior_scope'], as_index=False).agg(
        test_R2=('test_R2', 'mean'),
        test_MAE=('test_MAE', 'mean'),
        test_RMSE=('test_RMSE', 'mean'),
        cv_R2_mean=('cv_R2_mean', 'mean'),
        cv_R2_std=('cv_R2_std', 'mean'),
    )
    modality.to_csv(RESULTS_DIR / 'modality_scope_summary.csv', index=False)

    print('\nBest model for each target:')
    print('Models are selected by repeated cross-validation R²; test metrics remain held out.')
    print(best[['target_key', 'experiment_name', 'cv_R2_mean', 'cv_R2_std', 'test_R2', 'test_MAE']].to_string(index=False))

    for target, group in metrics.groupby('target_key'):
        plot = group.copy()
        plot['label'] = plot.apply(
            lambda r: 'imaging only' if r['input_type'] == 'imaging_only'
            else f"{r['input_type'].replace('_', ' ')}\n{r['behavior_scope']}", axis=1)
        plt.figure(figsize=(11, 6))
        plt.bar(plot['label'], plot['cv_R2_mean'])
        plt.axhline(0, linewidth=1)
        plt.ylabel('Mean repeated-CV R²')
        plt.title(f'{target.upper()} modality and scope comparison')
        plt.xticks(rotation=40, ha='right')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f'{target}_modality_scope_cv_R2.png', dpi=200)
        plt.close()

    for experiment_name in best['experiment_name']:
        save_importance(experiment_name)


if __name__ == '__main__':
    main()
