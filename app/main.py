"""
app/main.py  –  Partner 2: Streamlit Application & Caching
===========================================================
Multi-panel football scouting dashboard.

Layout
------
  Sidebar  : Global filters (league, age cap, position lock, cluster lock)
  Panel 1  : Player search + similarity scorecards
  Panel 2  : Head-to-Head metric comparison table
  Panel 3  : Radar / Pizza chart        ← rendered by Partner 1's visuals.py
  Panel 4  : 2D Tactical Cluster Map   ← rendered by Partner 1's visuals.py

Run locally
-----------
  streamlit run app/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make src/ importable when running from repo root or Streamlit Cloud
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

LOGO_PATH = ROOT / "assets" / "logo.png"

from src.similarity import SimilarityEngine, load_engine, METRIC_COLS

# ---------------------------------------------------------------------------
# Optional Partner 1 visuals (gracefully degraded if not yet merged)
# ---------------------------------------------------------------------------
try:
    from app.visuals import render_radar_chart, render_cluster_map  # type: ignore
    VISUALS_AVAILABLE = True
except ImportError:
    VISUALS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SCOUTIX — Football Scout AI",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — minimal, clean
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Metric card highlight */
    .match-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 16px;
        margin: 6px 0;
        color: #e0e0e0;
    }
    .match-pct-high  { color: #00e676; font-size: 1.4rem; font-weight: 700; }
    .match-pct-mid   { color: #ffea00; font-size: 1.4rem; font-weight: 700; }
    .match-pct-low   { color: #ff5252; font-size: 1.4rem; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Engine — loaded once, cached for the entire session
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="⚙️  Loading player data…")
def _get_engine() -> SimilarityEngine:
    return load_engine()


# ---------------------------------------------------------------------------
# Helper: colour-coded match % badge
# ---------------------------------------------------------------------------

def _match_badge(pct: float) -> str:
    css_class = (
        "match-pct-high" if pct >= 80
        else "match-pct-mid" if pct >= 65
        else "match-pct-low"
    )
    return f'<span class="{css_class}">{pct:.1f}%</span>'


def _format_market_value(val: float | None) -> str:
    """Format Transfermarkt valuation in €M or €K."""
    if pd.isna(val) or val is None or val <= 0:
        return "N/A"
    if val >= 1_000_000:
        return f"€{val / 1_000_000:.1f}M"
    if val >= 1_000:
        return f"€{val / 1_000:.0f}K"
    return f"€{int(val)}"


# ---------------------------------------------------------------------------
# Helper: human-readable metric label
# ---------------------------------------------------------------------------
METRIC_LABELS: dict[str, str] = {
    "goals_per90":               "Goals / 90",
    "assists_per90":             "Assists / 90",
    "xg_per90":                  "xG / 90",
    "xa_per90":                  "xA / 90",
    "shots_per90":               "Shots / 90",
    "key_passes_per90":          "Key Passes / 90",
    "progressive_carries_per90": "Prog. Carries / 90",
    "progressive_passes_per90":  "Prog. Passes / 90",
    "dribbles_per90":            "Dribbles / 90",
    "pressures_per90":           "Pressures / 90",
    "tackles_per90":             "Tackles / 90",
    "interceptions_per90":       "Interceptions / 90",
    "aerials_won_per90":         "Aerials Won / 90",
    "touches_att_pen_per90":     "Touches in Att. Pen. / 90",
}


# ---------------------------------------------------------------------------
# SIDEBAR — global filters
# ---------------------------------------------------------------------------

def _render_sidebar(engine: SimilarityEngine) -> dict:
    if LOGO_PATH.exists():
        st.sidebar.image(str(LOGO_PATH), use_container_width=True)
    st.sidebar.title("🔍 Scouting Filters")

    all_leagues = engine.get_leagues()
    selected_leagues = st.sidebar.multiselect(
        "🏆 Leagues",
        options=all_leagues,
        default=all_leagues,
        help="Restrict results to these competitions.",
    )

    max_age = st.sidebar.slider(
        "🎂 Max Age",
        min_value=16,
        max_value=40,
        value=35,
        step=1,
        help="Set to 23 to find U-23 replacement scouts.",
    )

    lock_position = st.sidebar.checkbox(
        "🔒 Same position only",
        value=False,
        help="Return only players sharing the same position code as the target.",
    )

    same_cluster = st.sidebar.checkbox(
        "🧩 Same archetype cluster only",
        value=False,
        help="Restrict to players in the same tactical cluster (Partner 1's KMeans label).",
    )

    exclude_same_team = st.sidebar.checkbox(
        "🚫 Exclude same club",
        value=True,
        help="Remove teammates of the query player.",
    )

    top_n = st.sidebar.slider(
        "📋 Results to show",
        min_value=3,
        max_value=25,
        value=10,
        step=1,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "SCOUTIX · Football Scout AI\n"
        "Analyse • Scout • Discover"
    )

    return {
        "leagues": selected_leagues if selected_leagues else all_leagues,
        "max_age": max_age,
        "lock_position": lock_position,
        "same_cluster": same_cluster,
        "exclude_same_team": exclude_same_team,
        "top_n": top_n,
    }


# ---------------------------------------------------------------------------
# PANEL 1 — Search & Scorecards
# ---------------------------------------------------------------------------

def _render_search_panel(engine: SimilarityEngine, filters: dict) -> tuple[str | None, pd.DataFrame]:
    """Return (selected_player_name, results_df)."""
    st.header("🔎 Find Similar Players")

    col_search, col_info = st.columns([3, 2])

    with col_search:
        raw_query = st.text_input(
            "Type a player name…",
            placeholder="e.g. Bukayo Saka",
            help="Partial names work — select from the dropdown below.",
        )

    suggestions = engine.player_search(raw_query) if raw_query else engine.get_all_players()

    with col_info:
        selected_player = st.selectbox(
            "Select player",
            options=suggestions,
            index=0 if suggestions else None,
            help="Exact name used for similarity lookup.",
        )

    if not selected_player:
        st.info("Start typing a player name to begin scouting.")
        return None, pd.DataFrame()

    # Show query player card
    try:
        qrow = engine.get_player_row(selected_player)
    except ValueError as e:
        st.error(str(e))
        return None, pd.DataFrame()

    with st.expander(f"📋 {selected_player} — profile", expanded=True):
        col_img, col_metrics = st.columns([1, 5])
        img_url = qrow.get("image_url")
        with col_img:
            if pd.notna(img_url) and str(img_url).startswith("http"):
                st.image(str(img_url), width=105)
            else:
                st.markdown("<div style='font-size:3rem;text-align:center;'>👤</div>", unsafe_allow_html=True)

        with col_metrics:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Club", qrow["team"])
            c2.metric("League", qrow["league"])
            c3.metric("Position", qrow["position"])
            c4.metric("Age", int(qrow["age"]))
            c5.metric("Market Value", _format_market_value(qrow.get("market_value_in_eur")))

            extra_info = []
            if pd.notna(qrow.get("foot")) and qrow.get("foot") != "Unknown":
                extra_info.append(f"Foot: **{str(qrow['foot']).capitalize()}**")
            if pd.notna(qrow.get("height_in_cm")):
                extra_info.append(f"Height: **{int(qrow['height_in_cm'])} cm**")
            if pd.notna(qrow.get("archetype")):
                extra_info.append(f"Archetype: **{qrow['archetype']}**")
            if extra_info:
                st.markdown(" · ".join(extra_info))

    # --- Run similarity query (cached by st.cache_data) ---
    results = _run_query(
        engine,
        selected_player,
        filters["top_n"],
        tuple(sorted(filters["leagues"])),
        filters["max_age"],
        filters["lock_position"],
        filters["same_cluster"],
        filters["exclude_same_team"],
    )

    if results.empty:
        st.warning("No players match your current filters. Try relaxing the sidebar constraints.")
        return selected_player, pd.DataFrame()

    # --- Scorecards ---
    st.subheader(f"Top {len(results)} Tactical Matches")
    _render_scorecards(results)

    return selected_player, results


@st.cache_data(show_spinner=False, ttl=3600)
def _run_query(
    _engine: SimilarityEngine,  # underscore prefix → not hashed by st.cache_data
    player_name: str,
    top_n: int,
    leagues: tuple[str, ...],
    max_age: int,
    lock_position: bool,
    same_cluster: bool,
    exclude_same_team: bool,
) -> pd.DataFrame:
    """
    Thin wrapper so Streamlit caches the result.
    Parameters are all hashable primitives after the engine object.
    """
    return _engine.get_similar_players(
        player_name,
        top_n=top_n,
        max_age=max_age,
        leagues=list(leagues),
        lock_position=lock_position,
        same_cluster=same_cluster,
        exclude_same_team=exclude_same_team,
    )


def _render_scorecards(results: pd.DataFrame) -> None:
    """3-column grid of match-% cards with player photos and valuations."""
    cols_per_row = 3
    rows = [
        results.iloc[i : i + cols_per_row]
        for i in range(0, len(results), cols_per_row)
    ]
    for row_df in rows:
        cols = st.columns(cols_per_row)
        for col, (_, r) in zip(cols, row_df.iterrows()):
            badge = _match_badge(r["match_pct"])
            mv_str = _format_market_value(r.get("market_value_in_eur"))
            img_url = r.get("image_url")
            img_tag = (
                f'<img src="{img_url}" style="width:48px;height:48px;border-radius:50%;object-fit:cover;float:right;margin-left:8px;border:1.5px solid #00e676;" />'
                if pd.notna(img_url) and str(img_url).startswith("http")
                else ""
            )
            col.markdown(
                f"""
                <div class="match-card">
                  {img_tag}
                  <b>{r['player']}</b><br>
                  <small>{r['team']} · {r['league']}</small><br>
                  <small>Age {int(r['age'])} · {r['position']} · {r.get('archetype','—')}</small><br>
                  <div style="margin-top:6px;display:flex;justify-content:space-between;align-items:center;">
                    <span>Match: {badge}</span>
                    <span style="color:#00b4d8;font-weight:600;font-size:0.95rem;">{mv_str}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# PANEL 2 — Head-to-Head metric comparison table
# ---------------------------------------------------------------------------

def _render_h2h_table(engine: SimilarityEngine, query_player: str, results: pd.DataFrame) -> None:
    st.header("📊 Head-to-Head Metric Comparison")

    if results.empty:
        st.info("Run a similarity search first.")
        return

    # Pick comparison players from results
    comparison_options = results["player"].tolist()
    selected_comparisons = st.multiselect(
        "Pick up to 5 players to compare",
        options=comparison_options,
        default=comparison_options[:3],
        max_selections=5,
    )

    if not selected_comparisons:
        return

    # Build comparison dataframe
    all_names = [query_player] + selected_comparisons
    rows = []
    for name in all_names:
        try:
            row = engine.get_player_row(name)
            entry = {
                "Player": name,
                "Team": row["team"],
                "Age": int(row["age"]),
                "Valuation": _format_market_value(row.get("market_value_in_eur")),
            }
            for m in METRIC_COLS:
                entry[METRIC_LABELS.get(m, m)] = round(float(row.get(m, 0)), 3)
            rows.append(entry)
        except ValueError:
            continue

    if not rows:
        return

    comp_df = pd.DataFrame(rows).set_index("Player")

    # Highlight the query player row
    def _highlight_query(row: pd.Series) -> list[str]:
        return (
            ["background-color: #0f3460; color: #e0e0e0"] * len(row)
            if row.name == query_player
            else [""] * len(row)
        )

    metric_cols_display = [METRIC_LABELS.get(m, m) for m in METRIC_COLS]
    styled = (
        comp_df[["Team", "Age", "Valuation"] + metric_cols_display]
        .style.apply(_highlight_query, axis=1)
        .format(precision=3, subset=metric_cols_display)
        .background_gradient(cmap="YlGn", subset=metric_cols_display, axis=0)
    )
    st.dataframe(styled, use_container_width=True)


# ---------------------------------------------------------------------------
# PANEL 3 & 4 — Partner 1 Visuals (graceful degradation)
# ---------------------------------------------------------------------------

def _render_visuals(engine: SimilarityEngine, query_player: str, results: pd.DataFrame) -> None:
    if not VISUALS_AVAILABLE:
        st.info(
            "📌 Radar chart & Cluster Map will appear here once "
            "`app/visuals.py` is merged from Partner 1's branch."
        )
        return

    if results.empty:
        return

    tab_radar, tab_cluster = st.tabs(["🕸️ Radar Chart", "🗺️ Cluster Map"])

    with tab_radar:
        top_match = results.iloc[0]["player"]
        fig_radar = render_radar_chart(
            engine.df, query_player, top_match, METRIC_COLS
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with tab_cluster:
        highlight_players = [query_player] + results["player"].tolist()
        fig_cluster = render_cluster_map(engine.df, highlight_players)
        st.plotly_chart(fig_cluster, use_container_width=True)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    # Hero header
    if LOGO_PATH.exists():
        col_logo, col_title = st.columns([1, 4])
        with col_logo:
            st.image(str(LOGO_PATH), width=160)
        with col_title:
            st.markdown(
                "## SCOUTIX — Football Scout AI",
                unsafe_allow_html=False,
            )
            st.caption(
                "Find tactically similar players across 2,000+ professionals · "
                "Powered by Cosine Similarity + KMeans Clustering"
            )
    else:
        st.markdown(
            "## ⚽ SCOUTIX — Football Scout AI",
            unsafe_allow_html=False,
        )
        st.caption(
            "Find tactically similar players across 2,000+ professionals · "
            "Powered by Cosine Similarity + KMeans Clustering"
        )
    st.markdown("---")

    # Load engine (cached for session)
    try:
        engine = _get_engine()
    except FileNotFoundError as exc:
        st.error(
            f"**Data not found.**\n\n{exc}\n\n"
            "Make sure `data/players_processed.parquet` exists before launching."
        )
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unexpected error loading engine: {exc}")
        st.stop()

    # Sidebar filters
    filters = _render_sidebar(engine)

    # Main panels
    query_player, results = _render_search_panel(engine, filters)

    if query_player and not results.empty:
        st.markdown("---")
        _render_h2h_table(engine, query_player, results)
        st.markdown("---")
        _render_visuals(engine, query_player, results)


if __name__ == "__main__":
    main()
