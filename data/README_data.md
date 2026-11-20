# data/

## `players_processed.parquet` — Data Contract

This file is **produced by Partner 1** (`src/preprocess.py` → `src/clustering.py`) and **consumed by Partner 2** (`src/similarity.py`, `app/main.py`).

Agree on this schema on **Day 1** before coding independently.

| Column | Type | Description |
|---|---|---|
| `player` | `str` | Display name (UTF-8, may contain accents) |
| `team` | `str` | Current club |
| `league` | `str` | One of: *Premier League, La Liga, Bundesliga, Serie A, Ligue 1* |
| `position` | `str` | FBRef position code: `FW`, `MF`, `DF` |
| `broad_role` | `str` | Human-readable: *Forward, Midfielder, Defender* |
| `age` | `int` | Age in years as of season start |
| `cluster` | `int` | KMeans cluster label (0 … k-1) |
| `archetype` | `str` | Human-readable cluster name (e.g. *Inverted Winger*) |
| `pca_x` | `float` | 2D PCA coordinate — x axis |
| `pca_y` | `float` | 2D PCA coordinate — y axis |
| `goals_per90` | `float` | Goals per 90 minutes |
| `assists_per90` | `float` | Assists per 90 |
| `xg_per90` | `float` | Expected goals per 90 |
| `xa_per90` | `float` | Expected assists per 90 |
| `shots_per90` | `float` | Shots per 90 |
| `key_passes_per90` | `float` | Key passes per 90 |
| `progressive_carries_per90` | `float` | Progressive carries per 90 |
| `progressive_passes_per90` | `float` | Progressive passes per 90 |
| `dribbles_per90` | `float` | Successful dribbles per 90 |
| `pressures_per90` | `float` | Pressures applied per 90 |
| `tackles_per90` | `float` | Tackles per 90 |
| `interceptions_per90` | `float` | Interceptions per 90 |
| `aerials_won_per90` | `float` | Aerial duels won per 90 |
| `touches_att_pen_per90` | `float` | Touches in attacking penalty area per 90 |

> **Goalkeepers and players under 900 minutes are excluded by Partner 1 during preprocessing.**

## Dummy data for development

Partner 2's tests (`tests/conftest.py`) auto-generate a 60-player synthetic parquet so you can run `pytest` without the real dataset. The real file should **not** be committed to git (add `data/*.parquet` to `.gitignore`).
