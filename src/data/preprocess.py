"""Data cleaning and preprocessing pipeline."""
import pandas as pd
import numpy as np


PRICE_MIN = 500
PRICE_MAX = 150_000
YEAR_MIN = 1990
ODOMETER_MAX = 400_000

TARGET = "price"
CATEGORICAL_COLS = ["manufacturer", "fuel", "transmission", "drive", "type", "paint_color", "state"]
NUMERIC_COLS = ["year", "odometer"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply filters and drop unusable rows."""
    df = df.copy()

    # Keep only relevant columns
    keep_cols = [TARGET] + CATEGORICAL_COLS + NUMERIC_COLS
    df = df[[c for c in keep_cols if c in df.columns]]

    # Drop rows with missing target
    df = df.dropna(subset=[TARGET])

    # Price range filter (remove junk listings)
    df = df[(df[TARGET] >= PRICE_MIN) & (df[TARGET] <= PRICE_MAX)]

    # Year filter
    if "year" in df.columns:
        df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= 2024)]

    # Odometer filter
    if "odometer" in df.columns:
        df = df[df["odometer"] <= ODOMETER_MAX]

    df = df.dropna(subset=NUMERIC_COLS)
    df = df.reset_index(drop=True)
    print(f"After cleaning: {len(df):,} rows")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer additional features."""
    df = df.copy()
    df["vehicle_age"] = 2024 - df["year"]
    df["age_x_odometer"] = df["vehicle_age"] * df["odometer"]
    return df


def run(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline."""
    df = clean(raw_df)
    df = add_features(df)
    return df
