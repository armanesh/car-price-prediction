"""Model training: Random Forest and XGBoost with cross-validation."""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "models"


def train_random_forest(X_train, y_train, params: dict = None):
    """Train a Random Forest regressor."""
    default_params = {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_leaf": 5,
        "n_jobs": -1,
        "random_state": 42,
    }
    if params:
        default_params.update(params)

    model = RandomForestRegressor(**default_params)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
    print(f"RF CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, params: dict = None):
    """Train an XGBoost regressor."""
    default_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "tree_method": "hist",
    }
    if params:
        default_params.update(params)

    model = XGBRegressor(**default_params)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
    print(f"XGB CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    model.fit(X_train, y_train, verbose=False)
    return model


def save_model(model, name: str):
    """Persist model to disk."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    print(f"Saved: {path}")
    return path


def load_model(name: str):
    """Load a persisted model."""
    return joblib.load(OUTPUT_DIR / f"{name}.joblib")
