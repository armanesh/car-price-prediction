"""Evaluation utilities: metrics and feature importance."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(model, X_test, y_test, name: str = "Model") -> dict:
    """Compute and print regression metrics."""
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

    results = {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}
    print(f"\n{name} Test Results:")
    print(f"  MAE:  ${mae:,.0f}")
    print(f"  RMSE: ${rmse:,.0f}")
    print(f"  R2:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    return results


def feature_importance(model, feature_names: list, top_n: int = 15) -> pd.DataFrame:
    """Extract and return top feature importances."""
    importances = model.feature_importances_
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
    return df
