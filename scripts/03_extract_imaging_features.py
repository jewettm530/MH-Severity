"""
03_extract_imaging_features.py

Purpose
-------
Create a richer, still-manageable representation of each participant's
resting-state parcellated fMRI data.

The output contains four feature families:
1. Global signal summaries.
2. Global functional-connectivity summaries.
3. Graph/network-organization summaries at one predefined density.
4. Parcel-level connectivity strength for every parcel.

The parcel-strength columns are intentionally retained here. The modeling
pipeline performs supervised selection and PCA *inside each train/CV fold*.
Doing PCA or target-based selection in this extraction script would allow
information from held-out participants to influence feature construction.

Run from the project root:
    python3 scripts/03_extract_imaging_features.py
"""

from pathlib import Path
import warnings

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path

from config import (
    FMRI_DIR,
    OUTPUT_DIR,
    REST_FILE_PATTERNS,
    GRAPH_DENSITY,
    EXPECTED_N_PARCELS,
)


EPS = 1e-12


def load_timeseries(h5_path: Path) -> np.ndarray:
    """Load a timepoints × parcels matrix and clean non-finite values."""
    with h5py.File(h5_path, "r") as handle:
        if "dataset" not in handle:
            raise ValueError(f"No 'dataset' key in {h5_path}")
        ts = np.asarray(handle["dataset"][()], dtype=np.float64)

    if ts.ndim != 2:
        raise ValueError(f"Expected a 2D array, received shape={ts.shape}")

    # TCP uses 488 atlas parcels. Identify the parcel axis explicitly so a
    # 488 × timepoints matrix is never mistaken for timepoints × parcels.
    if ts.shape[1] == EXPECTED_N_PARCELS:
        pass
    elif ts.shape[0] == EXPECTED_N_PARCELS:
        warnings.warn(
            f"Detected parcels × timepoints orientation {ts.shape} in "
            f"{h5_path.name}; transposing."
        )
        ts = ts.T
    else:
        raise ValueError(
            f"Cannot identify parcel axis from shape {ts.shape}; expected "
            f"one axis to contain {EXPECTED_N_PARCELS} parcels."
        )

    # Replace isolated invalid samples with the parcel median instead of zero,
    # which would create artificial correlations.
    ts = ts.copy()
    for parcel in range(ts.shape[1]):
        values = ts[:, parcel]
        finite = np.isfinite(values)
        if not finite.any():
            ts[:, parcel] = 0.0
        elif not finite.all():
            ts[~finite, parcel] = np.median(values[finite])

    return ts


def safe_correlation(ts: np.ndarray) -> np.ndarray:
    """Return a finite parcel × parcel Pearson-correlation matrix."""
    parcel_std = np.std(ts, axis=0)
    constant = parcel_std <= EPS

    corr = np.corrcoef(ts, rowvar=False)
    corr = np.asarray(corr, dtype=np.float64)
    corr[~np.isfinite(corr)] = 0.0
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    # Constant parcels have undefined correlations. Keep them in the stable
    # parcel index but set all off-diagonal correlations to zero.
    if constant.any():
        corr[constant, :] = 0.0
        corr[:, constant] = 0.0
        corr[constant, constant] = 1.0

    return corr


def threshold_absolute_connectivity(corr: np.ndarray, density: float) -> np.ndarray:
    """Create a binary graph from the strongest absolute correlations."""
    if not 0 < density < 1:
        raise ValueError("GRAPH_DENSITY must be between 0 and 1")

    n_parcels = corr.shape[0]
    i_upper, j_upper = np.triu_indices(n_parcels, k=1)
    weights = np.abs(corr[i_upper, j_upper])

    n_possible = len(weights)
    n_keep = max(1, int(round(density * n_possible)))
    keep_indices = np.argpartition(weights, -n_keep)[-n_keep:]

    adjacency = np.zeros((n_parcels, n_parcels), dtype=np.uint8)
    i_keep = i_upper[keep_indices]
    j_keep = j_upper[keep_indices]
    adjacency[i_keep, j_keep] = 1
    adjacency[j_keep, i_keep] = 1
    return adjacency


def binary_clustering_coefficients(adjacency: np.ndarray) -> np.ndarray:
    """Compute the undirected binary clustering coefficient per parcel."""
    a = adjacency.astype(np.float64, copy=False)
    degree = a.sum(axis=1)
    triangles_twice = np.diag(a @ a @ a)
    denominator = degree * (degree - 1)
    clustering = np.divide(
        triangles_twice,
        denominator,
        out=np.zeros_like(degree, dtype=np.float64),
        where=denominator > 0,
    )
    return clustering


def global_efficiency(adjacency: np.ndarray) -> float:
    """Mean reciprocal shortest-path distance for an undirected graph."""
    distances = shortest_path(
        csr_matrix(adjacency), directed=False, unweighted=True
    )
    n = distances.shape[0]
    mask = np.isfinite(distances) & (distances > 0)
    reciprocal = np.zeros_like(distances, dtype=np.float64)
    reciprocal[mask] = 1.0 / distances[mask]
    return float(reciprocal.sum() / (n * (n - 1)))


def summarize_signal(ts: np.ndarray) -> dict[str, float]:
    """Extract run-level temporal/signal summaries."""
    x0 = ts[:-1]
    x1 = ts[1:]
    x0_centered = x0 - x0.mean(axis=0)
    x1_centered = x1 - x1.mean(axis=0)
    numerator = np.sum(x0_centered * x1_centered, axis=0)
    denominator = np.sqrt(
        np.sum(x0_centered**2, axis=0) * np.sum(x1_centered**2, axis=0)
    )
    lag1 = np.divide(
        numerator,
        denominator,
        out=np.zeros(ts.shape[1], dtype=np.float64),
        where=denominator > EPS,
    )
    parcel_std = np.std(ts, axis=0)
    return {
        "global_signal_std": float(np.std(ts)),
        "global_signal_abs_mean": float(np.mean(np.abs(ts))),
        "parcel_signal_std_mean": float(np.mean(parcel_std)),
        "parcel_signal_std_sd": float(np.std(parcel_std)),
        "temporal_lag1_mean": float(np.mean(lag1)),
        "temporal_lag1_sd": float(np.std(lag1)),
    }


def summarize_connectivity(corr: np.ndarray) -> dict[str, float]:
    """Extract connectivity, graph, and regional features from mean FC."""
    n_parcels = corr.shape[0]
    i_upper, j_upper = np.triu_indices(n_parcels, k=1)
    upper = corr[i_upper, j_upper]
    positive = upper[upper > 0]
    negative = upper[upper < 0]

    off_diagonal_abs = np.abs(corr.copy())
    np.fill_diagonal(off_diagonal_abs, 0.0)
    parcel_abs_strength = off_diagonal_abs.sum(axis=1) / max(n_parcels - 1, 1)

    off_diagonal_pos = np.clip(corr, 0.0, None)
    np.fill_diagonal(off_diagonal_pos, 0.0)
    parcel_pos_strength = off_diagonal_pos.sum(axis=1) / max(n_parcels - 1, 1)

    off_diagonal_neg = np.clip(-corr, 0.0, None)
    np.fill_diagonal(off_diagonal_neg, 0.0)
    parcel_neg_strength = off_diagonal_neg.sum(axis=1) / max(n_parcels - 1, 1)

    adjacency = threshold_absolute_connectivity(corr, GRAPH_DENSITY)
    degree = adjacency.sum(axis=1).astype(np.float64)
    clustering = binary_clustering_coefficients(adjacency)
    n_components, component_labels = connected_components(
        csr_matrix(adjacency), directed=False
    )
    component_sizes = np.bincount(component_labels, minlength=n_components)

    features: dict[str, float] = {
        "conn_mean": float(np.mean(upper)),
        "conn_std": float(np.std(upper)),
        "conn_median": float(np.median(upper)),
        "conn_abs_mean": float(np.mean(np.abs(upper))),
        "conn_abs_q25": float(np.quantile(np.abs(upper), 0.25)),
        "conn_abs_q75": float(np.quantile(np.abs(upper), 0.75)),
        "conn_pos_mean": float(np.mean(positive)) if positive.size else 0.0,
        "conn_neg_mean": float(np.mean(negative)) if negative.size else 0.0,
        "conn_pos_fraction": float(np.mean(upper > 0)),
        "parcel_abs_strength_mean": float(np.mean(parcel_abs_strength)),
        "parcel_abs_strength_sd": float(np.std(parcel_abs_strength)),
        "parcel_abs_strength_max": float(np.max(parcel_abs_strength)),
        "parcel_pos_strength_sd": float(np.std(parcel_pos_strength)),
        "parcel_neg_strength_sd": float(np.std(parcel_neg_strength)),
        "graph_density": float(adjacency.sum() / (n_parcels * (n_parcels - 1))),
        "graph_degree_sd": float(np.std(degree)),
        "graph_degree_min": float(np.min(degree)),
        "graph_degree_max": float(np.max(degree)),
        "graph_clustering_mean": float(np.mean(clustering)),
        "graph_clustering_sd": float(np.std(clustering)),
        "graph_global_efficiency": global_efficiency(adjacency),
        "graph_component_count": float(n_components),
        "graph_largest_component_fraction": float(component_sizes.max() / n_parcels),
    }

    for parcel_idx, value in enumerate(parcel_abs_strength):
        features[f"parcel_strength_{parcel_idx:03d}"] = float(value)
    return features

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    subject_dirs = sorted(path for path in FMRI_DIR.iterdir() if path.is_dir())
    print(f"Found {len(subject_dirs)} subject folders in {FMRI_DIR}")
    print(f"Graph density: {GRAPH_DENSITY:.0%}")

    rows: list[dict[str, float | str]] = []
    expected_n_parcels: int | None = None

    for subject_number, subject_dir in enumerate(subject_dirs, start=1):
        signal_rows: list[dict[str, float]] = []
        correlations: list[np.ndarray] = []

        for filename in REST_FILE_PATTERNS:
            h5_path = subject_dir / filename
            if not h5_path.exists():
                continue
            try:
                ts = load_timeseries(h5_path)
                if expected_n_parcels is None:
                    expected_n_parcels = ts.shape[1]
                elif ts.shape[1] != expected_n_parcels:
                    raise ValueError(
                        f"Expected {expected_n_parcels} parcels; received {ts.shape[1]}"
                    )
                print(
                    f"[{subject_number}/{len(subject_dirs)}] "
                    f"{subject_dir.name}: {filename}, shape={ts.shape}"
                )
                signal_rows.append(summarize_signal(ts))
                correlations.append(safe_correlation(ts))
            except Exception as error:
                print(f"WARNING: failed {subject_dir.name}/{filename}: {error}")

        if not correlations:
            print(f"WARNING: no usable resting runs for {subject_dir.name}")
            continue

        # Fisher-z average is preferable to directly averaging correlations.
        clipped = [np.clip(corr, -0.999999, 0.999999) for corr in correlations]
        mean_corr = np.tanh(np.mean([np.arctanh(corr) for corr in clipped], axis=0))
        np.fill_diagonal(mean_corr, 1.0)

        features = pd.DataFrame(signal_rows).mean(numeric_only=True).to_dict()
        features.update(summarize_connectivity(mean_corr))

        row: dict[str, float | str] = {
            "participant_id": subject_dir.name,
            "n_rest_runs_used": float(len(correlations)),
        }
        row.update({f"rest_avg_{name}": float(value) for name, value in features.items()})
        rows.append(row)

    imaging = pd.DataFrame(rows)
    output_path = OUTPUT_DIR / "imaging_features.csv"
    imaging.to_csv(output_path, index=False)

    regional_columns = [c for c in imaging if "parcel_strength_" in c]
    nonregional_columns = [
        c for c in imaging.columns
        if c not in regional_columns and c != "participant_id"
    ]
    print(f"\nSaved: {output_path}")
    print(f"Participants: {len(imaging)}")
    print(f"Global/graph/QC columns: {len(nonregional_columns)}")
    print(f"Regional parcel-strength columns: {len(regional_columns)}")
    print(f"Total columns: {imaging.shape[1]}")


if __name__ == "__main__":
    main()
