"""Unit tests for evaluation metrics."""
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from src.evaluation.metrics import evaluate, feature_importance


def make_mock_model(preds):
    model = MagicMock()
    model.predict.return_value = np.array(preds)
    model.feature_importances_ = np.array([0.5, 0.3, 0.2])
    return model


def test_evaluate_returns_dict():
    model = make_mock_model([10000, 15000, 20000])
    X_test = pd.DataFrame({"a": [1, 2, 3]})
    y_test = pd.Series([10000, 15000, 20000])
    result = evaluate(model, X_test, y_test)
    assert "MAE" in result
    assert "R2" in result
    assert result["R2"] == pytest.approx(1.0)


def test_feature_importance_top_n():
    model = make_mock_model([1, 2, 3])
    model.feature_importances_ = np.array([0.5, 0.3, 0.2])
    df = feature_importance(model, ["age", "odometer", "brand"], top_n=2)
    assert len(df) == 2
    assert df.iloc[0]["feature"] == "age"


import pytest
