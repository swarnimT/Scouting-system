
"""Load, clean, and feature-engineer FBref Big 5 player stats.

Input is a CSV of FBref season totals with one row per player per club.
Per-90 features are season totals divided by ``90s``.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
RAW_PATH: Path = PROJECT_ROOT / "data" / "raw" / "big5_players.csv"
PREPROCESSED_PATH: Path = PROJECT_ROOT / "data" / "processed" / "players_preprocessed.parquet"

MIN_90S: float = 10.0

# FBref season-total column -> per-90 feature name.
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
COUNTING_COLS: list[str] = list(FEATURE_SOURCES)
FEATURES: list[str] = list(FEATURE_SOURCES.values())
PCT_FEATURES: list[str] = [f"{feature}_pct" for feature in FEATURES]

REQUIRED_COLS: list[str] = ["Player", "Pos", "Squad", "Comp", "Age", "90s", *COUNTING_COLS]
META_COLS: list[str] = ["Pos", "Squad", "Comp", "Age"]
OUTPUT_RENAME: dict[str, str] = {
    "Player": "player_name",
    "Squad": "squad",
    "Comp": "comp",
    "Pos": "pos",
    "Age": "age",
    "90s": "minutes_90s",
}

# Letters that NFKD cannot decompose into ASCII.
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
    """Return a lowercase ASCII-folded, whitespace-normalized name for matching.

    Args:
        name: Raw player name. Non-string values return an empty string.

    Returns:
        The normalized name, e.g. ``"Ødegaard"`` becomes ``"odegaard"``.
    """
    if not isinstance(name, str):
        return ""
    lowered = name.lower()
    for src, dst in _ASCII_OVERRIDES.items():
        lowered = lowered.replace(src, dst)
    folded = unicodedata.normalize("NFKD", lowered).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", folded).strip()


def primary_position(pos: pd.Series) -> pd.Series:
    """Reduce multi-position strings such as ``"MF,FW"`` to the first position."""
    return pos.fillna("").astype(str).str.split(",").str[0].str.strip().str.upper()


def parse_age(age: pd.Series) -> pd.Series:
    """Convert FBref ages like ``"25-123"`` (years-days) to whole years."""
    return pd.to_numeric(age.astype(str).str.split("-").str[0], errors="coerce")


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw FBref CSV.

    Raises:
        FileNotFoundError: If the file is missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at {path}. Place the FBref CSV there.")
    df = pd.read_csv(path)
    logger.info("Loaded %d raw rows from %s", len(df), path)
    return df


def validate_columns(df: pd.DataFrame) -> None:
    """Check that every required column exists.

    Raises:
        ValueError: If any required column is missing.
    """
    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Raw CSV is missing required columns: {missing}")


def clean_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """Remove header repeats, multi-club aggregate rows, and goalkeepers.

    Numeric columns are coerced and missing values are filled with 0.
    """
    mask = raw["Player"].ne("Player") & ~raw["Squad"].astype(str).str.contains(
        r"\d+ Squads", regex=True
    )
    df = raw.loc[mask].copy()
    df["Player"] = df["Player"].astype(str).str.strip()
    df["Pos"] = primary_position(df["Pos"])
    df = df.loc[df["Pos"] != "GK"].copy()
    df["Age"] = parse_age(df["Age"])
    if "Born" in df.columns:
        df["Born"] = df["Born"].astype(str)
    for col in ["90s", *COUNTING_COLS]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def consolidate_players(df: pd.DataFrame) -> pd.DataFrame:
    """Merge club rows into one row per player.

    Numeric totals are summed. Metadata comes from the club row with the
    highest ``90s``.
    """
    key = ["Player", "Born"] if "Born" in df.columns else ["Player"]
    df = df.sort_values("90s", ascending=False)
    totals = df.groupby(key, sort=False)[["90s", *COUNTING_COLS]].sum()
    primary = df.drop_duplicates(subset=key, keep="first").set_index(key)[META_COLS]
    return primary.join(totals).reset_index()


def add_per90_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 14 per-90 features. Requires ``90s > 0`` for every row."""
    df = df.copy()
    for source, feature in FEATURE_SOURCES.items():
        df[feature] = df[source] / df["90s"]
    df[FEATURES] = df[FEATURES].fillna(0.0)
    return df


def add_within_position_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``{feature}_pct`` columns: percentile rank (0 to 100) within each position."""
    df = df.copy()
    ranks = df.groupby("Pos")[FEATURES].rank(pct=True, method="average") * 100.0
    for feature in FEATURES:
        df[f"{feature}_pct"] = ranks[feature]
    return df


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, add ``player_name_clean``, and select the output schema."""
    df = df.rename(columns=OUTPUT_RENAME)
    df["player_name_clean"] = df["player_name"].map(normalize_name)
    columns = [
        "player_name",
        "player_name_clean",
        "squad",
        "comp",
        "pos",
        "age",
        "minutes_90s",
        *FEATURES,
        *PCT_FEATURES,
    ]
    return df[columns].reset_index(drop=True)


def preprocess(path: Path = RAW_PATH) -> pd.DataFrame:
    """Run the full preprocessing pipeline.

    Returns:
        One row per eligible outfield player, with per-90 features and
        within-position percentile ranks.

    Raises:
        ValueError: If columns are missing or no players pass the filters.
    """
    raw = load_raw(path)
    validate_columns(raw)
    rows = clean_rows(raw)
    logger.info("Club rows after removing goalkeepers and aggregate rows: %d", len(rows))

    players = consolidate_players(rows)
    logger.info("Unique players after consolidating transfers: %d", len(players))

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





