"""Shared modeling utilities for fair, repeatable Random Forest comparisons."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    SUBJECT_ID_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    N_SPLITS,
    N_REPEATS,
    IMAGING_REGIONAL_SELECT_K,
    IMAGING_REGIONAL_PCA_COMPONENTS,
)


def make_bins(y, max_bins=5):
    for q in range(max_bins, 1, -1):
        try:
            bins = pd.qcut(y.rank(method='first'), q=q, labels=False, duplicates='drop')
            if bins.nunique() > 1 and bins.value_counts().min() >= N_SPLITS:
                return bins.astype(int)
        except ValueError:
            continue
    return pd.Series(np.zeros(len(y), dtype=int), index=y.index)


def clean_X(df):
    X = df.drop(columns=[SUBJECT_ID_COLUMN, 'target'], errors='ignore').dropna(axis=1, how='all')

    # Retain run count in imaging_features.csv for quality control, but do not
    # let acquisition completeness become a biological predictor.
    technical = [
        c for c in X.columns
        if c.endswith('n_rest_runs_used')
        or c.endswith('n_timepoints')
        or c.endswith('n_parcels')
        or c.endswith('graph_density')
    ]
    X = X.drop(columns=technical, errors='ignore')

    object_cols = X.select_dtypes(include=['object']).columns.tolist()
    high_cardinality = [
        c for c in object_cols
        if X[c].nunique(dropna=True) > max(20, 0.5 * len(X))
    ]
    return X.drop(columns=high_cardinality, errors='ignore')


def build_pipeline(X):
    numeric = X.select_dtypes(include=['number', 'bool']).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    regional = [
        c for c in numeric
        if 'imaging__rest_avg_parcel_strength_' in c
    ]
    ordinary_numeric = [c for c in numeric if c not in regional]

    transformers = []
    if ordinary_numeric:
        transformers.append((
            'num',
            Pipeline([('imputer', SimpleImputer(strategy='median'))]),
            ordinary_numeric,
        ))

    if regional:
        # Both operations are fit only on the current training fold:
        # - SelectKBest retains parcels with the strongest training-only
        #   univariate relationship to the target.
        # - PCA captures distributed regional patterns without using y.
        select_k = min(IMAGING_REGIONAL_SELECT_K, len(regional))
        pca_components = min(IMAGING_REGIONAL_PCA_COMPONENTS, len(regional))
        regional_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('variance', VarianceThreshold()),
            ('regional_reduction', FeatureUnion([
                ('selected', SelectKBest(score_func=f_regression, k=select_k)),
                ('pca', Pipeline([
                    ('scale', StandardScaler()),
                    ('pca', PCA(
                        n_components=pca_components,
                        random_state=RANDOM_STATE,
                    )),
                ])),
            ])),
        ])
        transformers.append(('regional', regional_pipeline, regional))

    if categorical:
        transformers.append((
            'cat',
            Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
            ]),
            categorical,
        ))

    preprocess = ColumnTransformer(transformers, remainder='drop')
    model = RandomForestRegressor(
        n_estimators=750,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([('preprocess', preprocess), ('model', model)])


def fixed_split(df):
    """Same target/cohort always receives the same 80/20 split."""
    y = pd.to_numeric(df['target'], errors='coerce')
    keep = y.notna()
    df = df.loc[keep].copy()
    y = y.loc[keep]
    X = clean_X(df)
    bins = make_bins(y)
    stratify = bins if bins.nunique() > 1 else None
    return train_test_split(
        X, y, df[SUBJECT_ID_COLUMN], test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=stratify,
    )


def repeated_stratified_splits(X, y):
    bins = make_bins(y)
    if bins.nunique() <= 1:
        from sklearn.model_selection import RepeatedKFold
        return list(RepeatedKFold(
            n_splits=N_SPLITS,
            n_repeats=N_REPEATS,
            random_state=RANDOM_STATE,
        ).split(X, y))

    splits = []
    for repeat in range(N_REPEATS):
        cv = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE + repeat,
        )
        splits.extend(cv.split(X, bins))
    return list(splits)


def regression_metrics(y_true, prediction):
    return {
        'MAE': float(mean_absolute_error(y_true, prediction)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, prediction))),
        'R2': float(r2_score(y_true, prediction)),
    }


def fit_and_score(df):
    X_train, X_test, y_train, y_test, id_train, id_test = fixed_split(df)
    pipeline = build_pipeline(X_train)
    cv_splits = repeated_stratified_splits(X_train, y_train)
    cv = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv_splits,
        n_jobs=-1,
        scoring={
            'mae': 'neg_mean_absolute_error',
            'rmse': 'neg_root_mean_squared_error',
            'r2': 'r2',
        },
    )
    pipeline.fit(X_train, y_train)
    n_features_raw = int(X_train.shape[1])
    n_features_transformed = get_transformed_feature_count(pipeline, X_train)
    prediction = pipeline.predict(X_test)
    test = regression_metrics(y_test, prediction)
    result = {
        'n_total': int(len(df)),
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'n_features': n_features_transformed,
        'n_features_raw': n_features_raw,
        'cv_folds_total': int(len(cv_splits)),
        'cv_MAE_mean': float(-cv['test_mae'].mean()),
        'cv_MAE_std': float(cv['test_mae'].std()),
        'cv_RMSE_mean': float(-cv['test_rmse'].mean()),
        'cv_RMSE_std': float(cv['test_rmse'].std()),
        'cv_R2_mean': float(cv['test_r2'].mean()),
        'cv_R2_std': float(cv['test_r2'].std()),
        'cv_R2_min': float(cv['test_r2'].min()),
        'cv_R2_max': float(cv['test_r2'].max()),
        'test_MAE': test['MAE'],
        'test_RMSE': test['RMSE'],
        'test_R2': test['R2'],
    }
    predictions = pd.DataFrame({
        SUBJECT_ID_COLUMN: id_test.values,
        'y_true': y_test.values,
        'y_pred': prediction,
    })
    return pipeline, result, predictions

def get_transformed_feature_count(pipeline, X):
    """
    Return the number of features produced by the fitted preprocessing steps.
    """
    if "preprocessor" in pipeline.named_steps:
        transformed = pipeline.named_steps["preprocessor"].transform(X)
        return int(transformed.shape[1])

    # If preprocessing is split across multiple steps, transform through
    # everything before the final model.
    transformed = X

    for step_name, step in list(pipeline.named_steps.items())[:-1]:
        transformed = step.transform(transformed)

    return int(transformed.shape[1])