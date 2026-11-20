"""
src/similarity.py  –  Partner 2: Similarity & Filtering Engine
===============================================================
Responsibilities
----------------
1. Compute the pairwise Cosine Similarity matrix over the 14-16 stylistic
   per-90 metrics produced by Partner 1's preprocessing step.
2. Normalise raw cosine similarity scores ([-1, 1]) into a 0-100 % match index.
3. Expose a flexible query engine: get_similar_players() with support for
   - Age cap filter
   - League filter (allow multiple leagues)
   - Position lock (exact position or broad role)
   - Same-cluster constraint (only within Partner 1's KMeans cluster)
   - Exclude the query player from results

Data contract (players_processed.parquet columns expected)
----------------------------------------------------------
  player          : str   – display name
  team            : str
  league          : str   – one of {Premier League, La Liga, Bundesliga,
                            Serie A, Ligue 1}
  position        : str   – e.g. "MF", "FW", "DF"
  broad_role      : str   – e.g. "Midfielder", "Forward", "Defender"
  age             : int
  cluster         : int   – KMeans label from Partner 1
  archetype       : str   – human-readable cluster name from Partner 1
  pca_x           : float – 2D PCA coordinate x
  pca_y           : float – 2D PCA coordinate y
  <metric_cols>   : float – 14-16 per-90 metrics (see METRIC_COLS below)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data contract: the exact column names Partner 1 will export.
# Update here if the schema is changed on Day-1 standup.
# ---------------------------------------------------------------------------
METRIC_COLS: list[str] = [
    "goals_per90",
    "assists_per90",
    "xg_per90",
    "xa_per90",
    "shots_per90",
    "key_passes_per90",
    "progressive_carries_per90",
    "progressive_passes_per90",
    "dribbles_per90",
    "pressures_per90",
    "tackles_per90",
    "interceptions_per90",
    "aerials_won_per90",
    "touches_att_pen_per90",
]

# Columns required to exist in the parquet (beyond metrics)
REQUIRED_META_COLS: list[str] = [
    "player",
    "team",
    "league",
    "position",
    "broad_role",
    "age",
    "cluster",
    "archetype",
    "pca_x",
    "pca_y",
]

PROCESSED_PATH = Path(__file__).parent.parent / "data" / "players_processed.parquet"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if any required column is absent."""
    missing = [c for c in REQUIRED_META_COLS + METRIC_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"players_processed.parquet is missing columns: {missing}\n"
            "Check the data contract with Partner 1."
        )


def _scale_metrics(df: pd.DataFrame) -> np.ndarray:
    """Return a StandardScaler-normalised feature matrix (n_players × n_metrics)."""
    scaler = StandardScaler()
    return scaler.fit_transform(df[METRIC_COLS].fillna(0))


def _build_similarity_matrix(scaled: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity → shape (n, n) in [-1, 1].
    Cosine similarity is the right choice for per-90 rate stats because it is
    scale-invariant and captures playing-style direction, not raw volume.
    """
    sim = cosine_similarity(scaled)
    # Clip numerical noise outside valid range
    return np.clip(sim, -1.0, 1.0)


def _normalise_to_match_index(sim_matrix: np.ndarray) -> np.ndarray:
    """
    Convert cosine similarity ∈ [-1, 1] → match index ∈ [0, 100].
    Formula:  match% = (cosine_sim + 1) / 2 × 100
    """
    return (sim_matrix + 1) / 2 * 100


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SimilarityEngine:
    """
    Load once, query many times.

    Usage
    -----
    >>> engine = SimilarityEngine()
    >>> results = engine.get_similar_players("Bukayo Saka", top_n=10, max_age=23)
    """

    def __init__(self, parquet_path: Path | str = PROCESSED_PATH) -> None:
        self.parquet_path = Path(parquet_path)
        self._df: Optional[pd.DataFrame] = None
        self._match_matrix: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Lazy load / cache
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._df is not None:
            return  # already loaded

        if not self.parquet_path.exists():
            raise FileNotFoundError(
                f"Processed data not found at {self.parquet_path}.\n"
                "Partner 1 must run src/preprocess.py first."
            )

        df = pd.read_parquet(self.parquet_path)
        _validate_schema(df)

        # Reset index to ensure positional indexing aligns with matrix rows
        df = df.reset_index(drop=True)

        scaled = _scale_metrics(df)
        sim_raw = _build_similarity_matrix(scaled)
        self._match_matrix = _normalise_to_match_index(sim_raw)
        self._df = df
        logger.info("SimilarityEngine loaded %d players.", len(df))

    @property
    def df(self) -> pd.DataFrame:
        self._load()
        return self._df  # type: ignore[return-value]

    @property
    def match_matrix(self) -> np.ndarray:
        self._load()
        return self._match_matrix  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Core query
    # ------------------------------------------------------------------

    def get_similar_players(
        self,
        player_name: str,
        top_n: int = 10,
        *,
        max_age: Optional[int] = None,
        leagues: Optional[list[str]] = None,
        lock_position: bool = False,
        same_cluster: bool = False,
        exclude_same_team: bool = False,
    ) -> pd.DataFrame:
        """
        Return the top-N most similar players to `player_name`.

        Parameters
        ----------
        player_name     : Exact display name (case-sensitive). Use player_search()
                          to resolve fuzzy input before calling this method.
        top_n           : Number of results to return (default 10).
        max_age         : If set, only players aged ≤ max_age are considered.
        leagues         : If set, restrict results to these league(s).
        lock_position   : If True, only return players sharing the same
                          `position` value as the query player.
        same_cluster    : If True, only return players in the same KMeans cluster.
        exclude_same_team: If True, exclude players from the same club.

        Returns
        -------
        pd.DataFrame with columns:
            player, team, league, position, broad_role, age, cluster,
            archetype, pca_x, pca_y, match_pct, <METRIC_COLS>
        Sorted descending by match_pct.
        """
        df = self.df
        matrix = self.match_matrix

        # --- Locate query player ---
        mask_query = df["player"] == player_name
        if not mask_query.any():
            raise ValueError(
                f"Player '{player_name}' not found. "
                "Use engine.player_search() for fuzzy matching."
            )
        query_idx = int(mask_query.idxmax())
        query_row = df.loc[query_idx]

        # --- Retrieve similarity scores for all players ---
        scores: np.ndarray = matrix[query_idx]  # shape (n_players,)

        # Build working copy with scores
        result = df.copy()
        result["match_pct"] = scores.round(2)

        # --- Exclude self ---
        result = result[result["player"] != player_name]

        # --- Apply filters ---
        if max_age is not None:
            result = result[result["age"] <= max_age]

        if leagues:
            leagues_lower = [lg.strip().lower() for lg in leagues]
            result = result[result["league"].str.lower().isin(leagues_lower)]

        if lock_position:
            result = result[result["position"] == query_row["position"]]

        if same_cluster:
            result = result[result["cluster"] == query_row["cluster"]]

        if exclude_same_team:
            result = result[result["team"] != query_row["team"]]

        # --- Sort and trim ---
        result = result.sort_values("match_pct", ascending=False).head(top_n)

        # Return useful columns
        output_cols = REQUIRED_META_COLS + ["match_pct"] + METRIC_COLS
        return result[output_cols].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def player_search(self, query: str, top_k: int = 8) -> list[str]:
        """
        Case-insensitive substring search over player names.
        Returns up to top_k matching display names (sorted alphabetically).
        Useful for Streamlit search dropdowns.
        """
        df = self.df
        mask = df["player"].str.lower().str.contains(query.lower(), na=False)
        return sorted(df.loc[mask, "player"].tolist())[:top_k]

    def get_player_row(self, player_name: str) -> pd.Series:
        """Return the full data row for a single player."""
        df = self.df
        mask = df["player"] == player_name
        if not mask.any():
            raise ValueError(f"Player '{player_name}' not found.")
        return df.loc[mask.idxmax()]

    def get_all_players(self) -> list[str]:
        """Sorted list of all player names (for Streamlit dropdowns)."""
        return sorted(self.df["player"].tolist())

    def get_leagues(self) -> list[str]:
        """Distinct leagues present in the dataset."""
        return sorted(self.df["league"].unique().tolist())

    def get_positions(self) -> list[str]:
        """Distinct positions present in the dataset."""
        return sorted(self.df["position"].unique().tolist())

    def matrix_is_symmetric(self, tol: float = 1e-6) -> bool:
        """
        Sanity check: cosine similarity matrix must be symmetric.
        Used in unit tests.
        """
        m = self.match_matrix
        return bool(np.allclose(m, m.T, atol=tol))


# ---------------------------------------------------------------------------
# Module-level convenience (used by Streamlit via @st.cache_resource)
# ---------------------------------------------------------------------------

def load_engine(parquet_path: Path | str = PROCESSED_PATH) -> SimilarityEngine:
    """
    Factory — call once at app startup and cache with @st.cache_resource.
    """
    engine = SimilarityEngine(parquet_path)
    engine._load()  # eager load so first query is instant
    return engine
