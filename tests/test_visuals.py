"""
tests/test_visuals.py
----------------------
Unit tests for app/visuals.py (Radar Chart & Cluster Map).
"""
from __future__ import annotations

import pandas as pd
import pytest
import plotly.graph_objects as go

from app.visuals import render_radar_chart, render_cluster_map
from src.similarity import METRIC_COLS


def test_render_radar_chart_success(dummy_df: pd.DataFrame) -> None:
    p1 = dummy_df.iloc[0]["player"]
    p2 = dummy_df.iloc[1]["player"]
    fig = render_radar_chart(dummy_df, p1, p2, METRIC_COLS)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # two players


def test_render_radar_chart_unknown_player(dummy_df: pd.DataFrame) -> None:
    p1 = dummy_df.iloc[0]["player"]
    with pytest.raises(ValueError, match="not found"):
        render_radar_chart(dummy_df, p1, "NonExistentPlayer_XYZ", METRIC_COLS)


def test_render_cluster_map_success(dummy_df: pd.DataFrame) -> None:
    target = dummy_df.iloc[0]["player"]
    matches = dummy_df.iloc[1:4]["player"].tolist()
    fig = render_cluster_map(dummy_df, [target] + matches)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 2  # base scatter + target trace + matches trace
