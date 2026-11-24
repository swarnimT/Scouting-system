"""
src/preprocess.py  –  Partner 1: Feature Engineering & Transfermarkt Merge
==========================================================================
Load, clean, consolidate mid-season transfers, enrich with Transfermarkt
bio/financial data, compute 14 per-90 metrics, and calculate within-position
percentile ranks.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Primary raw data path (fallback to legacy big5_players.csv if present)
RAW_PATH: Path = PROJECT_ROOT / "data" / "raw" / "players_data-2024_2025.csv"
if not RAW_PATH.exists():
    LEGACY_RAW = PROJECT_ROOT / "data" / "raw" / "big5_players.csv"
    if LEGACY_RAW.exists():
        RAW_PATH = LEGACY_RAW

TM_PATH: Path = PROJECT_ROOT / "data" / "raw" / "players.csv"
PREPROCESSED_PATH: Path = PROJECT_ROOT / "data" / "processed" / "players_preprocessed.parquet"

MIN_90S: float = 10.0

# FBref season-total column -> per-90 feature name
FEATURE_SOURCES: dict[str, str] = {
    "Gls": "G_per90",
    "npxG": "npxG_per90",
    "Sh": "Sh_per90",
    "SoT": "SoT_per90",
    "Ast": "Ast_per90",
    "xAG": "xAG_per90",
    "SCA": "SCA_per90",
    "Crs": "Crs_per90",
    "PrgP": "PrgP_per90",
    "PrgC": "PrgC_per90",
    "Succ": "Succ_per90",
    "TklW": "TklW_per90",
    "Int": "Int_per90",
    "Blocks": "Blocks_per90",
}

# Optional extra metrics if present in multi-table scrapers
EXTRA_SOURCES: dict[str, str] = {
    "Won": "Won_per90",
    "Att Pen": "AttPen_per90",
}

COUNTING_COLS: list[str] = list(FEATURE_SOURCES)
FEATURES: list[str] = list(FEATURE_SOURCES.values())
PCT_FEATURES: list[str] = [f"{feature}_pct" for feature in FEATURES]

# Mapping to long descriptive metric names required by similarity engine
METRIC_ALIASES: dict[str, str] = {
    "goals_per90": "G_per90",
    "assists_per90": "Ast_per90",
    "xg_per90": "npxG_per90",
    "xa_per90": "xAG_per90",
    "shots_per90": "Sh_per90",
    "key_passes_per90": "SCA_per90",
    "progressive_carries_per90": "PrgC_per90",
    "progressive_passes_per90": "PrgP_per90",
    "dribbles_per90": "Succ_per90",
    "pressures_per90": "Blocks_per90",
    "tackles_per90": "TklW_per90",
    "interceptions_per90": "Int_per90",
    "aerials_won_per90": "Won_per90",
    "touches_att_pen_per90": "AttPen_per90",
}

REQUIRED_COLS: list[str] = ["Player", "Pos", "Squad", "Comp", "Age", "90s", *COUNTING_COLS]
META_COLS: list[str] = ["Pos", "Squad", "Comp", "Age"]

BROAD_ROLES: dict[str, str] = {
    "FW": "Forward",
    "MF": "Midfielder",
    "DF": "Defender",
    "GK": "Goalkeeper",
}

OUTPUT_RENAME: dict[str, str] = {
    "Player": "player_name",
    "Squad": "squad",
    "Comp": "comp",
    "Pos": "pos",
    "Age": "age",
    "90s": "minutes_90s",
}

# Letters that NFKD cannot decompose into ASCII
_ASCII_OVERRIDES: dict[str, str] = {
    "ø": "o",
    "æ": "ae",
    "œ": "oe",
    "ß": "ss",
    "ł": "l",
    "đ": "d",
    "ð": "d",
    "þ": "th",
    "ı": "i",
}


def normalize_name(name: object) -> str:
    """Return a lowercase ASCII-folded, whitespace-normalized name for matching."""
    if not isinstance(name, str):
        return ""
    lowered = name.lower()
    for src, dst in _ASCII_OVERRIDES.items():
        lowered = lowered.replace(src, dst)
    folded = unicodedata.normalize("NFKD", lowered).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", folded).strip()


def clean_league(comp: str) -> str:
    """Strip 2-3 letter country prefixes like 'eng Premier League' -> 'Premier League'."""
    return re.sub(r"^[a-z]{2,3}\s+", "", str(comp)).strip()


def primary_position(pos: pd.Series) -> pd.Series:
    """Reduce multi-position strings such as 'MF,FW' to the first position."""
    return pos.fillna("").astype(str).str.split(",").str[0].str.strip().str.upper()


def parse_age(age: pd.Series) -> pd.Series:
    """Convert FBref ages like '25-123' (years-days) to whole years."""
    return pd.to_numeric(age.astype(str).str.split("-").str[0], errors="coerce")


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw FBref CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at {path}. Place the CSV there.")
    df = pd.read_csv(path)
    logger.info("Loaded %d raw rows from %s", len(df), path)
    return df


def validate_columns(df: pd.DataFrame) -> None:
    """Check that every required column exists."""
    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Raw CSV is missing required columns: {missing}")


def clean_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """Remove header repeats, multi-club aggregate rows, and goalkeepers."""
    mask = raw["Player"].ne("Player") & ~raw["Squad"].astype(str).str.contains(
        r"\d+ Squads", regex=True
    )
    df = raw.loc[mask].copy()
    df["Player"] = df["Player"].astype(str).str.strip()
    df["Pos"] = primary_position(df["Pos"])
    df = df.loc[df["Pos"] != "GK"].copy()
    df["Age"] = parse_age(df["Age"])
    if "Born" in df.columns:
        df["Born"] = pd.to_numeric(df["Born"], errors="coerce")

    # Clean country prefix from Comp (e.g. 'eng Premier League' -> 'Premier League')
    df["Comp"] = df["Comp"].map(clean_league)

    # Coerce numeric columns
    counting_to_coerce = ["90s", *COUNTING_COLS]
    for opt in EXTRA_SOURCES:
        if opt in df.columns:
            counting_to_coerce.append(opt)

    for col in counting_to_coerce:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def consolidate_players(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge club rows into one row per player for mid-season transfers.
    Numeric totals are summed across clubs; metadata comes from the club row
    with the highest minutes (90s).
    """
    key = ["Player", "Born"] if "Born" in df.columns else ["Player"]
    df = df.sort_values("90s", ascending=False)

    numeric_cols = ["90s", *COUNTING_COLS]
    for opt in EXTRA_SOURCES:
        if opt in df.columns:
            numeric_cols.append(opt)

    totals = df.groupby(key, sort=False)[numeric_cols].sum()
    primary = df.drop_duplicates(subset=key, keep="first").set_index(key)[META_COLS]
    return primary.join(totals).reset_index()


def add_per90_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 14 per-90 features. Requires 90s > 0 for every row."""
    df = df.copy()
    safe_90s = df["90s"].replace(0, np.nan)
    for source, feature in FEATURE_SOURCES.items():
        df[feature] = (df[source] / safe_90s).fillna(0.0)

    # Optional extra metrics if present
    for source, feature in EXTRA_SOURCES.items():
        if source in df.columns:
            df[feature] = (df[source] / safe_90s).fillna(0.0)
        else:
            df[feature] = 0.0

    return df


def add_within_position_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """Add {feature}_pct columns: percentile rank (0 to 100) within each position."""
    df = df.copy()
    ranks = df.groupby("Pos")[FEATURES].rank(pct=True, method="average") * 100.0
    for feature in FEATURES:
        df[f"{feature}_pct"] = ranks[feature]
    return df


def enrich_with_transfermarkt(df: pd.DataFrame, tm_path: Path = TM_PATH) -> pd.DataFrame:
    """
    Enrich players with Transfermarkt valuation and bio data:
    market_value_in_eur, height_in_cm, foot, image_url, sub_position, url.
    Uses two-tier matching:
      Pass 1: (clean_name, birth_year)
      Pass 2: fallback to clean_name on active roster (last_season >= 2023)
    """
    tm_cols = [
        "market_value_in_eur",
        "highest_market_value_in_eur",
        "height_in_cm",
        "foot",
        "image_url",
        "sub_position",
        "url",
    ]

    if not tm_path.exists():
        logger.warning("Transfermarkt players.csv not found at %s. Bio fields set to NaN.", tm_path)
        for c in tm_cols:
            df[c] = np.nan
        return df

    tm = pd.read_csv(tm_path, low_memory=False)
    # Filter active/recent players and sort by latest season & highest market value
    tm_active = tm[tm["last_season"] >= 2023].copy()
    tm_active["clean_name"] = tm_active["name"].map(normalize_name)
    tm_active["birth_year"] = pd.to_datetime(tm_active["date_of_birth"], errors="coerce").dt.year
    tm_sorted = tm_active.sort_values(by=["last_season", "market_value_in_eur"], ascending=[False, False])

    df = df.reset_index(drop=True)

    # Pass 1: Deduplicate TM by (clean_name, birth_year)
    tm_pass1 = tm_sorted.drop_duplicates(subset=["clean_name", "birth_year"], keep="first")
    merged = df.merge(
        tm_pass1[["clean_name", "birth_year", *tm_cols]],
        left_on=["player_name_clean", "birth_year"],
        right_on=["clean_name", "birth_year"],
        how="left",
    ).drop(columns=["clean_name"], errors="ignore")

    # Pass 2: Fallback for unmatched rows
    unmatched_indices = merged[merged["market_value_in_eur"].isna()].index
    if len(unmatched_indices) > 0:
        tm_pass2 = tm_sorted.drop_duplicates(subset=["clean_name"], keep="first")
        unmatched_sub = merged.loc[unmatched_indices, ["player_name_clean"]]
        fallback = unmatched_sub.merge(
            tm_pass2[["clean_name", *tm_cols]],
            left_on="player_name_clean",
            right_on="clean_name",
            how="left",
        )
        for col in tm_cols:
            merged.loc[unmatched_indices, col] = fallback[col].values

    # Clean default string for foot
    merged["foot"] = merged["foot"].fillna("Unknown")
    return merged


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns, add aliases for both Partner 1 and Partner 2 schemas,
    and enrich with Transfermarkt.
    """
    df = df.rename(columns=OUTPUT_RENAME)
    df["player_name_clean"] = df["player_name"].map(normalize_name)
    df["birth_year"] = pd.to_numeric(df.get("Born", np.nan), errors="coerce")

    # Enrich with Transfermarkt
    df = enrich_with_transfermarkt(df)

    # Provide Partner 2 schema aliases
    df["player"] = df["player_name"]
    df["team"] = df["squad"]
    df["league"] = df["comp"]
    df["position"] = df["pos"]
    df["broad_role"] = df["pos"].map(BROAD_ROLES).fillna("Midfielder")

    # Long metric names for similarity engine
    for long_name, short_name in METRIC_ALIASES.items():
        if short_name in df.columns:
            df[long_name] = df[short_name]
        else:
            df[long_name] = 0.0

    return df.reset_index(drop=True)


def preprocess(path: Path = RAW_PATH) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline:
    clean -> consolidate transfers -> filter 90s >= 10.0 -> add per90 & percentiles -> finalize.
    """
    raw = load_raw(path)
    validate_columns(raw)
    rows = clean_rows(raw)
    logger.info("Club rows after removing goalkeepers and aggregate rows: %d", len(rows))

    players = consolidate_players(rows)
    logger.info("Unique players after consolidating transfers: %d", len(players))

    # Apply 90s threshold on consolidated playing time
    players = players.loc[players["90s"] >= MIN_90S].copy()
    logger.info("Players with at least %.1f 90s: %d", MIN_90S, len(players))
    if players.empty:
        raise ValueError(f"No players left after filtering with 90s >= {MIN_90S}.")

    players = add_per90_features(players)
    players = add_within_position_percentiles(players)
    return finalize(players)


def save_preprocessed(df: pd.DataFrame, path: Path = PREPROCESSED_PATH) -> Path:
    """Write the preprocessed table to parquet and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def main() -> None:
    """Run preprocessing and save the intermediate parquet for clustering."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = preprocess()
    path = save_preprocessed(df)
    logger.info("Saved %d players to %s", len(df), path)


if __name__ == "__main__":
    main()
