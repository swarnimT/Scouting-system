"""
app/visuals.py  –  Plotly visuals for Radar Chart & 2D Tactical Cluster Map
===========================================================================
Used by app/main.py for Panels 3 & 4.
"""

from __future__ import annotations

from collections.abc import Sequence
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

TARGET_COLOR: str = "#FFD700"
SIMILAR_COLOR: str = "#00E676"


def render_radar_chart(
    df: pd.DataFrame,
    player_1: str,
    player_2: str,
    metrics: Sequence[str],
) -> go.Figure:
    """
    Overlay two players' normalized metric profiles on a polar radar chart.

    Parameters
    ----------
    df       : DataFrame containing player rows and metric columns.
    player_1 : Query player name.
    player_2 : Comparison player name.
    metrics  : List of metric column names to plot.
    """
    p1_rows = df.loc[df["player"] == player_1]
    p2_rows = df.loc[df["player"] == player_2]

    if p1_rows.empty:
        raise ValueError(f"Player '{player_1}' not found in dataset.")
    if p2_rows.empty:
        raise ValueError(f"Player '{player_2}' not found in dataset.")

    p1 = p1_rows.iloc[0]
    p2 = p2_rows.iloc[0]

    labels = [
        m.replace("_per90", "").replace("_", " ").title()
        for m in metrics
    ]
    theta = labels + labels[:1]

    fig = go.Figure()

    for player, name, line_color in [
        (p1, player_1, "#00b4d8"),
        (p2, player_2, "#ff9e00"),
    ]:
        raw_vals = [float(player.get(m, 0.0)) for m in metrics]

        # Min-max normalization across dataset for legible radar comparison [0-100]
        norm_vals = []
        for m, val in zip(metrics, raw_vals):
            col_min = float(df[m].min()) if m in df else 0.0
            col_max = float(df[m].max()) if m in df else 1.0
            if col_max > col_min:
                scaled = (val - col_min) / (col_max - col_min) * 100.0
            else:
                scaled = 50.0
            norm_vals.append(round(scaled, 1))

        norm_closed = norm_vals + norm_vals[:1]
        raw_closed = raw_vals + raw_vals[:1]

        fig.add_trace(
            go.Scatterpolar(
                r=norm_closed,
                theta=theta,
                fill="toself",
                name=name,
                line=dict(color=line_color, width=2.5),
                opacity=0.55,
                customdata=raw_closed,
                hovertemplate=(
                    "<b>%{theta}</b><br>"
                    + f"{name}<br>"
                    + "Relative Rank: %{r:.1f}%<br>"
                    + "Per-90 Value: %{customdata:.2f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=f"Tactical Comparison: {player_1} vs {player_2}",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%"),
            bgcolor="#16213e",
        ),
        paper_bgcolor="#1a1a2e",
        font=dict(color="#e0e0e0"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def render_cluster_map(
    df: pd.DataFrame,
    highlight_players: Sequence[str],
) -> go.Figure:
    """
    2D PCA scatter plot showing tactical archetypes, highlighting target and matches.

    Parameters
    ----------
    df                : DataFrame with pca_x, pca_y, archetype (or cluster).
    highlight_players : List where index 0 is query player, and remainder are matches.
    """
    target_player = highlight_players[0] if highlight_players else ""
    similar_players = list(highlight_players[1:]) if len(highlight_players) > 1 else []

    cluster_col = "archetype" if "archetype" in df.columns else "cluster"

    fig = px.scatter(
        df,
        x="pca_x",
        y="pca_y",
        color=cluster_col,
        hover_name="player",
        hover_data={
            "team": True,
            "league": True,
            "position": True,
            "age": True,
            "pca_x": False,
            "pca_y": False,
        },
        opacity=0.45,
        title="2D Tactical Archetype Map (PCA Projection)",
        labels={"pca_x": "PCA 1", "pca_y": "PCA 2", cluster_col: "Archetype"},
    )

    # Highlight query player (star)
    target_match = df.loc[df["player"] == target_player]
    if not target_match.empty:
        trow = target_match.iloc[0]
        fig.add_trace(
            go.Scatter(
                x=[trow["pca_x"]],
                y=[trow["pca_y"]],
                mode="markers+text",
                name=f"Target: {target_player}",
                text=[target_player],
                textposition="top center",
                textfont=dict(color=TARGET_COLOR, size=13),
                marker=dict(
                    symbol="star",
                    size=22,
                    color=TARGET_COLOR,
                    line=dict(color="#000", width=1.5),
                ),
                hoverinfo="text",
                hovertext=f"Target: {target_player} ({trow['team']})",
            )
        )

    # Highlight similar matches (diamonds)
    sim_rows = df.loc[df["player"].isin(similar_players)]
    if not sim_rows.empty:
        fig.add_trace(
            go.Scatter(
                x=sim_rows["pca_x"],
                y=sim_rows["pca_y"],
                mode="markers",
                name="Top Matches",
                marker=dict(
                    symbol="diamond",
                    size=14,
                    color=SIMILAR_COLOR,
                    line=dict(color="#000", width=1.2),
                ),
                hoverinfo="text",
                hovertext=sim_rows["player"] + " (" + sim_rows["team"] + ")",
            )
        )

    fig.update_layout(
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
        xaxis=dict(gridcolor="#2a2a4e", zerolinecolor="#3a3a5e"),
        yaxis=dict(gridcolor="#2a2a4e", zerolinecolor="#3a3a5e"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig
