"""Data cleaning and target construction for the IMDb Movies India dataset.

Raw fields arrive in dirty formats:
    Year      -> '(2019)'
    Votes     -> '"1,086"'
    Duration  -> '142 min' / 'TV Series' / '39 episodes'
    Rating    -> '7.6' (sometimes blank or missing)
    Genre     -> 'Drama' or 'Action, Thriller' (primary genre kept)
    Actor 1   -> lead actor
"""

from __future__ import annotations

import math
import re

import pandas as pd

RATING_THRESHOLD = 6.5

RAW_COLUMNS = ["Name", "Year", "Duration", "Genre", "Rating", "Votes", "Director", "Actor 1", "Actor 2", "Actor 3"]

FEATURE_COLUMNS = [
    "Genre",
    "Director",
    "Lead_Actor",
    "Actor_2",
    "Actor_3",
    "Year",
    "Duration_min",
    "Log_Votes",
    "Genre_Count",
    "High_Rated",
]


def _parse_year(value: object) -> int | None:
    """Extract the first 4-digit number from a year cell like '(2019)'."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    match = re.search(r"(\d{4})", str(value))
    if not match:
        return None
    year = int(match.group(1))
    return year if 1900 <= year <= 2026 else None


def _parse_votes(value: object) -> int | None:
    """Extract the numeric vote count from a cell like '1,086'."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    digits = re.sub(r"[^\d]", "", str(value))
    if not digits:
        return None
    return int(digits)


def _parse_duration(value: object) -> float | None:
    """Parse minutes from '142 min'; return None for episodes/TV series/unknown."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().lower()
    if "episode" in text or "season" in text or "series" in text:
        return None
    digits = re.sub(r"[^\d]", "", text)
    if not digits:
        return None
    minutes = float(digits)
    return minutes if 10 <= minutes <= 600 else None


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw CSV with the known IMDb Movies India schema (utf-8 with latin-1 fallback)."""
    try:
        return pd.read_csv(path, usecols=RAW_COLUMNS, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, usecols=RAW_COLUMNS, encoding="latin-1")


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Parse dirty fields, drop unusable rows, and build the High_Rated target."""
    df = raw.copy()

    df["Year"] = df["Year"].apply(_parse_year)
    df["Votes"] = df["Votes"].apply(_parse_votes)
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df["Duration_min"] = df["Duration"].apply(_parse_duration)
    df["Genre_Count"] = df["Genre"].astype("str").str.split(",").str.len()
    df["Genre"] = df["Genre"].astype("str").str.split(",").str[0].str.strip()
    df["Director"] = df["Director"].astype("str").str.strip()
    df["Lead_Actor"] = df["Actor 1"].astype("str").str.strip()
    df["Actor_2"] = df["Actor 2"].astype("str").str.strip()
    df["Actor_3"] = df["Actor 3"].astype("str").str.strip()

    for col in ["Name", "Genre", "Director", "Lead_Actor", "Actor_2", "Actor_3"]:
        df[col] = df[col].replace(["nan", "None", ""], pd.NA)

    df = df.dropna(subset=["Rating", "Genre", "Director", "Lead_Actor", "Year", "Votes"])
    df = df[df["Votes"] > 0]

    df["Duration_min"] = df["Duration_min"].fillna(df["Duration_min"].median())
    df["Actor_2"] = df["Actor_2"].fillna("Unknown")
    df["Actor_3"] = df["Actor_3"].fillna("Unknown")
    df["Genre_Count"] = df["Genre_Count"].clip(1, 5)

    df["High_Rated"] = (df["Rating"] >= RATING_THRESHOLD).astype(int)
    df["Log_Votes"] = df["Votes"].apply(lambda v: float(math.log1p(v)))

    return df.reset_index(drop=True)


def keep_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select only the high-signal columns used by the model."""
    return df[FEATURE_COLUMNS].copy()
