"""Audit cleaned data, experiment datasets, imaging features, and outcomes.

Run after feature preparation/merging and before interpreting model results:
    python3 scripts/08_validate_outputs.py

Outputs are written to outputs/validation/.
"""
from __future__ import annotations

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

from config import (
    AMBIGUOUS_REPEATED_NINES,
    CLEAN_BEHAVIOR_DIR,
    EXPERIMENT_DIR,
    OUTPUT_DIR,
    RESULTS_DIR,
    SUBJECT_ID_COLUMN,
    TARGETS,
)

VALIDATION_DIR = OUTPUT_DIR / "validation"
REPEATED_NINE_PATTERN = re.compile(r"^-?9{2,}$")


def repeated_nine_mask(series: pd.Series) -> pd.Series:
    """Identify signed values made only of at least two repeated 9s."""
    as_text = series.astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)
    return as_text.str.match(REPEATED_NINE_PATTERN, na=False)


def audit_repeated_nines() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    files = sorted(CLEAN_BEHAVIOR_DIR.glob("*.csv"))
    for path in files:
        df = pd.read_csv(path, low_memory=False)
        for column in df.columns:
            if column == SUBJECT_ID_COLUMN:
                continue
            mask = repeated_nine_mask(df[column])
            if not mask.any():
                continue
            counts = df.loc[mask, column].astype("string").value_counts(dropna=False)
            for value, count in counts.items():
                normalized = str(value).replace(".0", "")
                ambiguous = normalized in {str(v) for v in AMBIGUOUS_REPEATED_NINES}
                rows.append({
                    "file": path.name,
                    "column": column,
                    "value": value,
                    "count": int(count),
                    "action": "review instrument range" if ambiguous else "should be missing",
                })
    return pd.DataFrame(rows)


def target_correlations() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute feature-to-outcome Spearman correlations without imputing data."""
    rows: list[dict[str, object]] = []
    flags: list[dict[str, object]] = []
    for target_key in TARGETS:
        path = EXPERIMENT_DIR / f"{target_key}__behavioral_only__all.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        y = pd.to_numeric(df["target"], errors="coerce")
        for column in df.columns:
            if column in {SUBJECT_ID_COLUMN, "target"}:
                continue
            x = pd.to_numeric(df[column], errors="coerce")
            paired = pd.concat([x, y], axis=1).dropna()
            if len(paired) < 10 or paired.iloc[:, 0].nunique() < 2:
                continue
            rho = paired.iloc[:, 0].corr(paired.iloc[:, 1], method="spearman")
            record = {
                "target": target_key,
                "feature": column,
                "spearman_rho": float(rho),
                "abs_spearman_rho": float(abs(rho)),
                "n_complete": int(len(paired)),
            }
            rows.append(record)
            if abs(rho) >= 0.80:
                flags.append(record)
    all_corr = pd.DataFrame(rows)
    if not all_corr.empty:
        all_corr = all_corr.sort_values(["target", "abs_spearman_rho"], ascending=[True, False])
    return all_corr, pd.DataFrame(flags)


def audit_imaging() -> pd.DataFrame:
    path = OUTPUT_DIR / "imaging_features.csv"
    if not path.exists():
        return pd.DataFrame([{"check": "file_exists", "status": "fail", "detail": str(path)}])
    df = pd.read_csv(path, low_memory=False)
    regional = [c for c in df.columns if "parcel_strength_" in c]
    numeric = df.select_dtypes(include=[np.number])
    constant = [c for c in numeric.columns if numeric[c].nunique(dropna=True) <= 1]
    checks = [
        {"check": "participants", "status": "info", "detail": len(df)},
        {"check": "total_columns", "status": "info", "detail": df.shape[1]},
        {"check": "regional_parcel_features", "status": "pass" if len(regional) >= 400 else "warn", "detail": len(regional)},
        {"check": "missing_cells", "status": "pass" if int(df.isna().sum().sum()) == 0 else "warn", "detail": int(df.isna().sum().sum())},
        {"check": "duplicate_ids", "status": "pass" if not df.iloc[:, 0].duplicated().any() else "fail", "detail": int(df.iloc[:, 0].duplicated().sum())},
        {"check": "constant_numeric_features", "status": "warn" if constant else "pass", "detail": ", ".join(constant) or "none"},
    ]
    return pd.DataFrame(checks)


def audit_experiments() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(EXPERIMENT_DIR.glob("*.csv")):
        if path.name == "experiment_manifest.csv":
            continue
        df = pd.read_csv(path, low_memory=False)
        target_key = path.stem.split("__", 1)[0]
        target = pd.to_numeric(df.get("target"), errors="coerce")
        config = TARGETS.get(target_key, {})
        valid = target.between(config.get("valid_min", -np.inf), config.get("valid_max", np.inf))
        rows.append({
            "experiment": path.stem,
            "rows": len(df),
            "columns": df.shape[1],
            "duplicate_ids": int(df[SUBJECT_ID_COLUMN].duplicated().sum()) if SUBJECT_ID_COLUMN in df else np.nan,
            "missing_targets": int(target.isna().sum()),
            "targets_outside_valid_range": int((~valid & target.notna()).sum()),
            "remaining_nonambiguous_repeated_nines": int(sum(
                repeated_nine_mask(df[c]).sum()
                for c in df.columns
                if c not in {SUBJECT_ID_COLUMN, "target"}
            )),
        })
    return pd.DataFrame(rows)


def main() -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    repeated = audit_repeated_nines()
    correlations, correlation_flags = target_correlations()
    imaging = audit_imaging()
    experiments = audit_experiments()

    repeated.to_csv(VALIDATION_DIR / "repeated_nine_audit.csv", index=False)
    correlations.to_csv(VALIDATION_DIR / "spearman_feature_outcome_correlations.csv", index=False)
    correlation_flags.to_csv(VALIDATION_DIR / "spearman_over_0_80_flags.csv", index=False)
    imaging.to_csv(VALIDATION_DIR / "imaging_feature_audit.csv", index=False)
    experiments.to_csv(VALIDATION_DIR / "experiment_output_audit.csv", index=False)

    summary = {
        "repeated_nine_candidates": int(len(repeated)),
        "spearman_correlations_computed": int(len(correlations)),
        "spearman_abs_ge_0_80": int(len(correlation_flags)),
        "experiment_files_checked": int(len(experiments)),
    }
    (VALIDATION_DIR / "validation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved validation outputs to {VALIDATION_DIR}")


if __name__ == "__main__":
    main()
