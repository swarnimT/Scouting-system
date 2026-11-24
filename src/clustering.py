





"""Standardize per-90 features, cluster players, project to 2D, and export."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.preprocess import FEATURES, PCT_FEATURES, PREPROCESSED_PATH, PROJECT_ROOT

logger = logging.getLogger(__name__)

PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
X_SCALED_PATH: Path = PROCESSED_DIR / "X_scaled.npy"
SCALER_PATH: Path = MODELS_DIR / "scaler.joblib"
PROCESSED_PATH: Path = PROCESSED_DIR / "players_processed.parquet"

DEFAULT_K: int = 6
K_RANGE: range = range(4, 11)
RANDOM_STATE: int = 42
N_INIT: int = 10
TOP_N_FEATURES: int = 3

# Tactical archetype names mapped from centroid profiles
CLUSTER_LABELS: dict[int, str] = {
    0: "Central Defender / Interceptor",
    1: "Attacking Fullback / Crosser",
    2: "Playmaker / Inverted Winger",
    3: "Ball-Winning Midfielder",
    4: "Target Forward / Finisher",
    5: "Direct Winger / Ball Carrier",
}

BASE_COLUMNS: list[str] = [
    "player_name",
    "player_name_clean",
    "squad",
    "comp",
    "pos",
    "age",
    "minutes_90s",
]
OUTPUT_COLUMNS: list[str] = [
    *BASE_COLUMNS,
    *FEATURES,
    *PCT_FEATURES,
    "cluster_id",
    "cluster_label",
    "pca_1",
    "pca_2",
]


def load_preprocessed(path: Path = PREPROCESSED_PATH) -> pd.DataFrame:
    """Load the parquet written by ``src.preprocess``.

    Raises:
        FileNotFoundError: If preprocessing has not been run.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.preprocess` first.")
    return pd.read_parquet(path)


def scale_features(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Fit a StandardScaler on the 14 features and persist the matrix and scaler."""
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES].to_numpy(dtype=float))

    X_SCALED_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(X_SCALED_PATH, X)
    joblib.dump(scaler, SCALER_PATH)
    logger.info("Saved scaled matrix %s and scaler %s", X_SCALED_PATH, SCALER_PATH)
    return X, scaler


def evaluate_k(X: np.ndarray, k_range: Sequence[int] = K_RANGE) -> pd.DataFrame:
    """Print and return inertia and silhouette score for each k."""
    rows: list[dict[str, float]] = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        labels = model.fit_predict(X)
        sil = float(silhouette_score(X, labels))
        rows.append({"k": k, "inertia": float(model.inertia_), "silhouette": sil})
        print(f"k={k:>2}  inertia={model.inertia_:12.1f}  silhouette={sil:.4f}")
    return pd.DataFrame(rows)


def fit_kmeans(X: np.ndarray, k: int) -> KMeans:
    """Fit KMeans with the project's fixed random state and n_init."""
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
    model.fit(X)
    return model


def top_features_per_cluster(
    model: KMeans, n: int = TOP_N_FEATURES
) -> dict[int, list[tuple[str, float]]]:
    """Return the ``n`` highest-z-score features for each centroid.

    The input is standardized, so centroid values are z-scores of the cluster means.
    """
    summary: dict[int, list[tuple[str, float]]] = {}
    for cluster_id, centroid in enumerate(model.cluster_centers_):
        top_idx = np.argsort(centroid)[::-1][:n]
        summary[cluster_id] = [(FEATURES[i], float(centroid[i])) for i in top_idx]
    return summary


def fit_pca(X: np.ndarray) -> tuple[PCA, np.ndarray]:
    """Project the scaled matrix to two principal components."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X)
    variance = float(pca.explained_variance_ratio_.sum()) * 100
    logger.info("PCA explains %.1f%% of variance in 2 components", variance)
    return pca, coords


def build_processed_frame(df: pd.DataFrame, labels: np.ndarray, coords: np.ndarray) -> pd.DataFrame:
    """Attach cluster and PCA columns while preserving all metadata, bio, and metric columns."""
    out = df.reset_index(drop=True).copy()
    out["cluster_id"] = labels.astype(int)
    out["cluster"] = labels.astype(int)
    out["cluster_label"] = [CLUSTER_LABELS.get(int(c), f"Cluster {int(c)}") for c in labels]
    out["archetype"] = out["cluster_label"]
    out["pca_1"] = coords[:, 0]
    out["pca_2"] = coords[:, 1]
    out["pca_x"] = coords[:, 0]
    out["pca_y"] = coords[:, 1]
    return out


def run_clustering(
    df: pd.DataFrame,
    n_clusters: int = DEFAULT_K,
    k_range: Sequence[int] = K_RANGE,
) -> pd.DataFrame:
    """Scale, sweep k, fit the final model, add PCA, and export the dataset.

    Args:
        df: Output of ``src.preprocess.preprocess``.
        n_clusters: The k used for the final model.
        k_range: Values of k to evaluate and print.

    Returns:
        The processed dataset, also saved to ``PROCESSED_PATH``.
    """
    X, _ = scale_features(df)

    print("\n=== K sweep ===")
    evaluate_k(X, k_range)

    print(f"\n=== Final model: k={n_clusters} ===")
    model = fit_kmeans(X, n_clusters)
    sizes = np.bincount(model.labels_, minlength=n_clusters)
    for cluster_id, features in top_features_per_cluster(model).items():
        described = ", ".join(f"{name} (z={z:+.2f})" for name, z in features)
        print(f"cluster {cluster_id} (n={sizes[cluster_id]}): {described}")

    _, coords = fit_pca(X)
    out = build_processed_frame(df, model.labels_, coords)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PROCESSED_PATH, index=False)

    root_data_path = PROJECT_ROOT / "data" / "players_processed.parquet"
    out.to_parquet(root_data_path, index=False)
    logger.info("Saved %d processed players to %s and %s", len(out), PROCESSED_PATH, root_data_path)
    return out


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point. Use ``--k`` to change the final cluster count."""
    parser = argparse.ArgumentParser(description="Cluster players and export the dataset.")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="k for the final model")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = load_preprocessed()
    run_clustering(df, n_clusters=args.k)


if __name__ == "__main__":
    main()
