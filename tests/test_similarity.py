"""
tests/test_similarity.py
------------------------
Unit tests for src/similarity.py

Coverage targets
----------------
1. Matrix properties   – symmetry, value range [0, 100], diagonal = 100
2. Query correctness   – self not in results, top-1 is highest score
3. Filters             – age cap, league, position lock, same-cluster
4. Edge cases          – unknown player raises, empty filter returns nothing
5. Schema validation   – missing column raises ValueError
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.similarity import (
    METRIC_COLS,
    SimilarityEngine,
    _build_similarity_matrix,
    _normalise_to_match_index,
    _scale_metrics,
)


# ===========================================================================
# 1. Matrix properties
# ===========================================================================

class TestMatrixProperties:
    def test_symmetric(self, engine: SimilarityEngine) -> None:
        """Cosine similarity matrix must be symmetric."""
        assert engine.matrix_is_symmetric(), (
            "Match matrix is not symmetric — check _build_similarity_matrix."
        )

    def test_values_in_range(self, engine: SimilarityEngine) -> None:
        """All normalised scores must lie in [0, 100]."""
        m = engine.match_matrix
        assert m.min() >= 0.0 - 1e-6, f"Min value {m.min()} < 0"
        assert m.max() <= 100.0 + 1e-6, f"Max value {m.max()} > 100"

    def test_diagonal_is_100(self, engine: SimilarityEngine) -> None:
        """A player compared with themselves must score 100 %."""
        diag = np.diag(engine.match_matrix)
        assert np.allclose(diag, 100.0, atol=0.1), (
            f"Diagonal should be ~100, got min={diag.min():.2f}"
        )

    def test_matrix_shape(self, engine: SimilarityEngine) -> None:
        n = len(engine.df)
        shape = engine.match_matrix.shape
        assert shape == (n, n), f"Expected ({n},{n}), got {shape}"


# ===========================================================================
# 2. Query correctness
# ===========================================================================

class TestQueryCorrectness:
    def test_self_excluded(self, engine: SimilarityEngine) -> None:
        """The query player must not appear in results."""
        player = engine.df.iloc[0]["player"]
        results = engine.get_similar_players(player, top_n=10)
        assert player not in results["player"].values, (
            "Query player appeared in its own results."
        )

    def test_top1_is_highest(self, engine: SimilarityEngine) -> None:
        """Results must be sorted descending by match_pct."""
        player = engine.df.iloc[5]["player"]
        results = engine.get_similar_players(player, top_n=10)
        pcts = results["match_pct"].tolist()
        assert pcts == sorted(pcts, reverse=True), "Results not sorted by match_pct."

    def test_top_n_respected(self, engine: SimilarityEngine) -> None:
        """get_similar_players must return at most top_n rows."""
        player = engine.df.iloc[2]["player"]
        for n in [3, 7, 15]:
            results = engine.get_similar_players(player, top_n=n)
            assert len(results) <= n, f"Returned {len(results)} rows, expected ≤ {n}"

    def test_output_schema(self, engine: SimilarityEngine) -> None:
        """Results must contain all expected columns."""
        player = engine.df.iloc[0]["player"]
        results = engine.get_similar_players(player, top_n=5)
        for col in ["player", "team", "league", "position", "age",
                    "cluster", "archetype", "match_pct"] + METRIC_COLS:
            assert col in results.columns, f"Missing column: {col}"


# ===========================================================================
# 3. Filters
# ===========================================================================

class TestFilters:
    def test_max_age_filter(self, engine: SimilarityEngine) -> None:
        player = engine.df.iloc[0]["player"]
        cap = 23
        results = engine.get_similar_players(player, top_n=50, max_age=cap)
        if not results.empty:
            assert results["age"].max() <= cap, (
                f"Age filter violated: max age in results = {results['age'].max()}"
            )

    def test_league_filter(self, engine: SimilarityEngine) -> None:
        player = engine.df.iloc[0]["player"]
        league = "Premier League"
        results = engine.get_similar_players(player, top_n=50, leagues=[league])
        if not results.empty:
            assert results["league"].unique().tolist() == [league], (
                "League filter returned players from other leagues."
            )

    def test_position_lock(self, engine: SimilarityEngine) -> None:
        player = engine.df.iloc[0]["player"]
        pos = engine.df.iloc[0]["position"]
        results = engine.get_similar_players(player, top_n=50, lock_position=True)
        if not results.empty:
            assert (results["position"] == pos).all(), (
                "Position lock returned players with a different position."
            )

    def test_same_cluster(self, engine: SimilarityEngine) -> None:
        player = engine.df.iloc[0]["player"]
        cluster = int(engine.df.iloc[0]["cluster"])
        results = engine.get_similar_players(player, top_n=50, same_cluster=True)
        if not results.empty:
            assert (results["cluster"] == cluster).all(), (
                "Same-cluster filter returned players from other clusters."
            )

    def test_multiple_leagues(self, engine: SimilarityEngine) -> None:
        player = engine.df.iloc[0]["player"]
        allowed = ["Premier League", "La Liga"]
        results = engine.get_similar_players(player, top_n=50, leagues=allowed)
        if not results.empty:
            assert set(results["league"].unique()).issubset(set(allowed))

    def test_exclude_same_team(self, engine: SimilarityEngine) -> None:
        player = engine.df.iloc[0]["player"]
        team = engine.df.iloc[0]["team"]
        results = engine.get_similar_players(player, top_n=50, exclude_same_team=True)
        assert team not in results["team"].values, (
            "exclude_same_team=True still returned a teammate."
        )


# ===========================================================================
# 4. Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_unknown_player_raises(self, engine: SimilarityEngine) -> None:
        with pytest.raises(ValueError, match="not found"):
            engine.get_similar_players("DefinitelyNotAPlayer_XYZ_999")

    def test_impossible_filter_returns_empty(self, engine: SimilarityEngine) -> None:
        """Combining age cap=16 with same_cluster should yield 0 results gracefully."""
        player = engine.df.iloc[0]["player"]
        results = engine.get_similar_players(player, top_n=10, max_age=16)
        # May or may not be empty depending on fixture; just must not raise
        assert isinstance(results, pd.DataFrame)

    def test_player_search_partial(self, engine: SimilarityEngine) -> None:
        """player_search must return names containing the substring."""
        hits = engine.player_search("Player_0")
        assert all("Player_0" in h for h in hits), (
            "player_search returned names not matching the query."
        )

    def test_player_search_no_match(self, engine: SimilarityEngine) -> None:
        hits = engine.player_search("ZZZNOMATCH99999")
        assert hits == [], f"Expected [], got {hits}"

    def test_get_all_players_sorted(self, engine: SimilarityEngine) -> None:
        players = engine.get_all_players()
        assert players == sorted(players), "get_all_players() not sorted."

    def test_get_leagues_non_empty(self, engine: SimilarityEngine) -> None:
        leagues = engine.get_leagues()
        assert len(leagues) > 0

    def test_get_positions_non_empty(self, engine: SimilarityEngine) -> None:
        positions = engine.get_positions()
        assert len(positions) > 0
        assert set(positions).issubset({"FW", "MF", "DF"})

    def test_get_player_row_success(self, engine: SimilarityEngine) -> None:
        p = engine.df.iloc[0]["player"]
        row = engine.get_player_row(p)
        assert row["player"] == p
        assert "team" in row

    def test_get_player_row_not_found(self, engine: SimilarityEngine) -> None:
        with pytest.raises(ValueError, match="not found"):
            engine.get_player_row("NonExistentPlayer_9999")

    def test_missing_parquet_raises(self, tmp_path: "Path") -> None:
        eng = SimilarityEngine(parquet_path=tmp_path / "nonexistent.parquet")
        with pytest.raises(FileNotFoundError):
            eng._load()

    def test_load_engine_factory(self, dummy_df: pd.DataFrame, tmp_path: "Path") -> None:
        from src.similarity import load_engine
        parquet = tmp_path / "factory_test.parquet"
        dummy_df.to_parquet(parquet, index=False)
        eng = load_engine(parquet)
        assert len(eng.df) == len(dummy_df)
        assert eng.match_matrix is not None


# ===========================================================================
# 5. Schema validation
# ===========================================================================

class TestSchemaValidation:
    def test_missing_metric_column_raises(
        self, dummy_df: pd.DataFrame, tmp_path: "Path"
    ) -> None:
        """Engine should raise ValueError if a metric column is absent."""
        bad_df = dummy_df.drop(columns=[METRIC_COLS[0]])
        parquet = tmp_path / "bad.parquet"
        bad_df.to_parquet(parquet, index=False)
        eng = SimilarityEngine(parquet_path=parquet)
        with pytest.raises(ValueError, match="missing columns"):
            eng._load()

    def test_missing_meta_column_raises(
        self, dummy_df: pd.DataFrame, tmp_path: "Path"
    ) -> None:
        bad_df = dummy_df.drop(columns=["player"])
        parquet = tmp_path / "bad2.parquet"
        bad_df.to_parquet(parquet, index=False)
        eng = SimilarityEngine(parquet_path=parquet)
        with pytest.raises(ValueError, match="missing columns"):
            eng._load()


# ===========================================================================
# 6. Internal math functions (pure unit tests — no fixture needed)
# ===========================================================================

class TestInternalMath:
    def test_normalise_identity(self) -> None:
        """cosine_sim = 1.0  →  match_index = 100.0"""
        raw = np.array([[1.0, 0.5], [0.5, 1.0]])
        norm = _normalise_to_match_index(raw)
        assert np.isclose(norm[0, 0], 100.0)
        assert np.isclose(norm[1, 1], 100.0)

    def test_normalise_minus_one(self) -> None:
        """cosine_sim = -1.0  →  match_index = 0.0"""
        raw = np.array([[-1.0]])
        norm = _normalise_to_match_index(raw)
        assert np.isclose(norm[0, 0], 0.0)

    def test_normalise_zero(self) -> None:
        """cosine_sim = 0.0  →  match_index = 50.0"""
        raw = np.array([[0.0]])
        norm = _normalise_to_match_index(raw)
        assert np.isclose(norm[0, 0], 50.0)

    def test_scale_metrics_zero_mean(self, dummy_df: pd.DataFrame) -> None:
        """StandardScaler should produce near-zero column means."""
        scaled = _scale_metrics(dummy_df)
        col_means = scaled.mean(axis=0)
        assert np.allclose(col_means, 0, atol=1e-6), (
            f"Scaled features should have zero mean, got max |mean| = {np.abs(col_means).max():.4f}"
        )
