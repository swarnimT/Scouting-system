"""
tests/test_real_data.py
-----------------------
Integration tests verifying queries against the generated European dataset.
"""
from __future__ import annotations

from src.similarity import load_engine


def test_real_dataset_engine_loads() -> None:
    engine = load_engine()
    assert len(engine.df) >= 1400
    assert set(engine.get_leagues()) == {
        "Bundesliga",
        "La Liga",
        "Ligue 1",
        "Premier League",
        "Serie A",
    }


def test_real_dataset_top_players_and_market_values() -> None:
    engine = load_engine()
    saka_hits = engine.player_search("Bukayo Saka")
    assert "Bukayo Saka" in saka_hits

    saka = engine.get_player_row("Bukayo Saka")
    assert saka["team"] == "Arsenal"
    assert saka["league"] == "Premier League"
    assert saka["market_value_in_eur"] > 50_000_000
    assert isinstance(saka["image_url"], str)

    sims = engine.get_similar_players("Bukayo Saka", top_n=5)
    assert len(sims) == 5
    assert "market_value_in_eur" in sims.columns
    assert "image_url" in sims.columns
    assert "Bukayo Saka" not in sims["player"].values
