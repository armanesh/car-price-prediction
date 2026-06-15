"""
End-to-end pipeline: load -> preprocess -> encode -> train -> evaluate.
Run with: python -m src.pipeline
"""
from src.data.load import load_raw
from src.data.preprocess import run as preprocess
from src.features.encode import encode_and_split
from src.models.train import train_random_forest, train_xgboost, save_model
from src.evaluation.metrics import evaluate, feature_importance
import pandas as pd


def main():
    print("=== Car Price Prediction Pipeline ===\n")

    # 1. Load
    raw_df = load_raw()

    # 2. Preprocess
    clean_df = preprocess(raw_df)

    # 3. Encode + split
    X_train, X_test, y_train, y_test, encoder = encode_and_split(clean_df)

    # 4. Train
    print("\n--- Training Random Forest ---")
    rf = train_random_forest(X_train, y_train)

    print("\n--- Training XGBoost ---")
    xgb = train_xgboost(X_train, y_train)

    # 5. Evaluate
    results = []
    results.append(evaluate(rf, X_test, y_test, name="RandomForest"))
    results.append(evaluate(xgb, X_test, y_test, name="XGBoost"))

    print("\n--- Model Comparison ---")
    print(pd.DataFrame(results).set_index("model").to_string())

    # 6. Feature importance (best model)
    print("\n--- Top Features (XGBoost) ---")
    fi = feature_importance(xgb, X_train.columns.tolist())
    print(fi.to_string(index=False))

    # 7. Save best
    save_model(xgb, "xgboost_car_price")
    save_model(rf, "random_forest_car_price")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
