





"""Plotly visuals for tactical clusters and percentile radar comparisons."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.preprocess import PCT_FEATURES, normalize_name

logger = logging.getLogger(__name__)

TARGET_COLOR: str = "#FFD700"
SIMILAR_COLOR: str = "#FFFFFF"


def _find_player(df: pd.DataFrame, name: str) -> pd.Series:
    """Return the row whose normalized name matches ``name``.

    If several rows match, the one with the most 90s is returned.

    Raises:
        ValueError: If no player matches.
    """
    key = normalize_name(name)
    matches = df.loc[df["player_name_clean"] == key]
    if matches.empty:
        raise ValueError(f"No player matches {name!r}.")
    return matches.sort_values("minutes_90s", ascending=False).iloc[0]


def plot_tactical_clusters(
    df: pd.DataFrame,
    target_player: str,
    similar_players: Sequence[str],
) -> go.Figure:
    """Draw the PCA scatter colored by cluster, with the target and similar players highlighted.

    Args:
        df: Processed dataset with ``pca_1``, ``pca_2``, and ``cluster_label``.
        target_player: Name of the player to highlight with a star.
        similar_players: Names of similar players to mark with diamonds.
            Names that are not found are logged and skipped.

    Returns:
        A Plotly figure.

    Raises:
        ValueError: If ``target_player`` is not in ``df``.
    """
    target = _find_player(df, target_player)

    fig = px.scatter(
        df,
        x="pca_1",
        y="pca_2",
        color="cluster_label",
        hover_name="player_name",
        hover_data={
            "squad": True,
            "pos": True,
            "minutes_90s": ":.1f",
            "pca_1": False,
            "pca_2": False,
        },
        opacity=0.5,
        title="Tactical clusters (PCA projection)",
        labels={"pca_1": "PCA 1", "pca_2": "PCA 2", "cluster_label": "Archetype"},
    )

    fig.add_trace(
        go.Scatter(
            x=[target["pca_1"]],
            y=[target["pca_2"]],
            mode="markers",
            name=f"Target: {target['player_name']}",
            marker=dict(symbol="star", size=24, color=TARGET_COLOR, line=dict(color="black", width=1.5)),
            hovertext=[target["player_name"]],
            hoverinfo="text",
        )
    )

    similar_rows: list[pd.Series] = []
    for name in similar_players:
        try:
            similar_rows.append(_find_player(df, name))
        except ValueError:
            logger.warning("Similar player not found, skipping: %s", name)

    if similar_rows:
        similar = pd.DataFrame(similar_rows)
        fig.add_trace(
            go.Scatter(
                x=similar["pca_1"],
                y=similar["pca_2"],
                mode="markers",
                name="Similar players",
                marker=dict(
                    symbol="diamond",
                    size=14,
                    color=SIMILAR_COLOR,
                    line=dict(color="black", width=1.2),
                ),
                hovertext=similar["player_name"],
                hoverinfo="text",
            )
        )

    return fig


def plot_radar_comparison(
    df: pd.DataFrame,
    player_1: str,
    player_2: str,
    features_pct: Sequence[str] | None = None,
) -> go.Figure:
    """Overlay two players' within-position percentile ranks on a radar chart.

    Args:
        df: Processed dataset containing the ``_pct`` columns.
        player_1: First player's name.
        player_2: Second player's name.
        features_pct: ``_pct`` columns to plot. Defaults to all 14 features.

    Returns:
        A Plotly figure with one filled polygon per player.

    Raises:
        ValueError: If either player is not in ``df``.
    """
    cols = list(features_pct) if features_pct is not None else list(PCT_FEATURES)
    p1 = _find_player(df, player_1)
    p2 = _find_player(df, player_2)

    labels = [col.removesuffix("_pct").replace("_per90", "") for col in cols]
    theta = labels + labels[:1]

    fig = go.Figure()
    for player in (p1, p2):
        values = [float(player[col]) for col in cols]
        fig.add_trace(
            go.Scatterpolar(
                r=values + values[:1],
                theta=theta,
                fill="toself",
                name=str(player["player_name"]),
                opacity=0.6,
            )
        )

    fig.update_layout(
        title=f"{p1['player_name']} vs {p2['player_name']}: percentile rank within position",
        polar=dict(radialaxis=dict(range=[0, 100], tickvals=[0, 25, 50, 75, 100])),
        showlegend=True,
    )
    return fig
