"""Unit tests for preprocessing pipeline."""
import pandas as pd
import numpy as np
import pytest
from src.data.preprocess import clean, add_features


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "price": [5000, 200, 200000, 15000, 30000],
        "year": [2010, 2005, 2015, 1985, 2018],
        "odometer": [80000, 150000, 50000, 200000, 30000],
        "manufacturer": ["toyota", "ford", "honda", "bmw", "toyota"],
        "fuel": ["gas", "gas", "gas", "diesel", "gas"],
        "transmission": ["automatic"] * 5,
        "drive": ["fwd"] * 5,
        "type": ["sedan"] * 5,
        "paint_color": ["white"] * 5,
        "state": ["ca"] * 5,
    })


def test_clean_removes_extreme_prices(sample_df):
    result = clean(sample_df)
    assert result["price"].max() <= 150_000
    assert result["price"].min() >= 500


def test_clean_removes_old_vehicles(sample_df):
    result = clean(sample_df)
    assert result["year"].min() >= 1990


def test_add_features_vehicle_age(sample_df):
    cleaned = clean(sample_df)
    result = add_features(cleaned)
    assert "vehicle_age" in result.columns
    assert (result["vehicle_age"] >= 0).all()


def test_add_features_interaction(sample_df):
    cleaned = clean(sample_df)
    result = add_features(cleaned)
    assert "age_x_odometer" in result.columns
