"""Feature encoding and train/test splitting."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

from src.data.preprocess import CATEGORICAL_COLS, TARGET

FEATURE_COLS = [
    "vehicle_age", "odometer", "age_x_odometer",
    "manufacturer", "fuel", "transmission", "drive", "type", "paint_color", "state"
]


def encode_and_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Encode categoricals with OrdinalEncoder and split into train/test sets.

    Returns X_train, X_test, y_train, y_test, encoder
    """
    available_cats = [c for c in CATEGORICAL_COLS if c in df.columns]
    available_nums = [c for c in ["vehicle_age", "odometer", "age_x_odometer"] if c in df.columns]
    features = available_nums + available_cats

    df_model = df[features + [TARGET]].dropna()
    X = df_model[features].copy()
    y = df_model[TARGET]

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X[available_cats] = encoder.fit_transform(X[available_cats])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test, encoder
