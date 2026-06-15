"""Data loading utilities."""
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_raw(filename: str = "vehicles.csv") -> pd.DataFrame:
    """Load raw dataset from data/raw/."""
    path = DATA_DIR / "raw" / filename
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows from {path.name}")
    return df


def load_processed(filename: str = "vehicles_clean.csv") -> pd.DataFrame:
    """Load preprocessed dataset from data/processed/."""
    path = DATA_DIR / "processed" / filename
    return pd.read_csv(path)
