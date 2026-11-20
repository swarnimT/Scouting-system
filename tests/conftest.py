"""
tests/conftest.py
-----------------
Shared pytest fixtures: builds a minimal synthetic DataFrame that
mirrors the players_processed.parquet data contract so tests can run
without the real dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.similarity import METRIC_COLS, REQUIRED_META_COLS, SimilarityEngine


# ---------------------------------------------------------------------------
# Synthetic data factory
# ---------------------------------------------------------------------------

def _make_dummy_df(n: int = 50, seed: int = 42) -> pd.DataFrame:
    """
    Return a synthetic DataFrame matching the data contract.
    Uses a fixed random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    leagues = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]
    positions = ["FW", "MF", "DF"]
    broad_roles = {"FW": "Forward", "MF": "Midfielder", "DF": "Defender"}
    archetypes = ["Inverted Winger", "Deep-Lying Playmaker", "Ball-Winner",
                  "Target Striker", "Box-to-Box Midfielder"]

    pos_col = rng.choice(positions, size=n)
    cluster_col = rng.integers(0, 5, size=n)

    data: dict = {
        "player":     [f"Player_{i:03d}" for i in range(n)],
        "team":       [f"Team_{rng.integers(0,10):02d}" for _ in range(n)],
        "league":     rng.choice(leagues, size=n).tolist(),
        "position":   pos_col.tolist(),
        "broad_role": [broad_roles[p] for p in pos_col],
        "age":        rng.integers(17, 36, size=n).tolist(),
        "cluster":    cluster_col.tolist(),
        "archetype":  [archetypes[c % len(archetypes)] for c in cluster_col],
        "pca_x":      rng.uniform(-3, 3, size=n).tolist(),
        "pca_y":      rng.uniform(-3, 3, size=n).tolist(),
    }

    for col in METRIC_COLS:
        data[col] = rng.uniform(0, 2, size=n).tolist()

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def dummy_df() -> pd.DataFrame:
    return _make_dummy_df(n=60)


@pytest.fixture(scope="session")
def engine(dummy_df: pd.DataFrame, tmp_path_factory: pytest.TempPathFactory) -> SimilarityEngine:
    """
    SimilarityEngine loaded from a temporary parquet built from dummy data.
    Session-scoped so the matrix is built once for all tests.
    """
    tmp = tmp_path_factory.mktemp("data")
    parquet = tmp / "players_processed.parquet"
    dummy_df.to_parquet(parquet, index=False)

    eng = SimilarityEngine(parquet_path=parquet)
    eng._load()
    return eng
