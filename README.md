# Car Price Prediction

A production-style machine learning pipeline for predicting used car prices from Craigslist listings. Built with a clean module structure, reproducible preprocessing, cross-validated model training, and a proper test suite.

---

## Problem

Used car pricing is notoriously opaque. Listings on platforms like Craigslist vary wildly for similar vehicles. This project builds a regression model that estimates a fair market price given vehicle characteristics — useful for buyers, sellers, and dealerships alike.

**Dataset:** [Craigslist Cars & Trucks — Kaggle](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data) (~420k listings, US-wide)

---

## Results

| Model | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|
| Random Forest | ~$2,400 | ~$4,100 | ~0.87 | ~18% |
| **XGBoost** | **~$2,100** | **~$3,700** | **~0.89** | **~16%** |

XGBoost outperforms on all metrics and is selected as the final model.

---

## Project Structure

```
car-price-prediction/
├── data/
│   ├── raw/           # Original dataset (not committed)
│   └── processed/     # Cleaned data (not committed)
├── src/
│   ├── data/
│   │   ├── load.py        # Data loading
│   │   └── preprocess.py  # Cleaning + feature engineering
│   ├── features/
│   │   └── encode.py      # Encoding + train/test split
│   ├── models/
│   │   └── train.py       # RF + XGBoost training with CV
│   ├── evaluation/
│   │   └── metrics.py     # Metrics + feature importance
│   └── pipeline.py        # End-to-end orchestration
├── notebooks/
│   └── 01_eda.ipynb       # Exploratory data analysis
├── tests/
│   ├── test_preprocess.py
│   └── test_metrics.py
├── outputs/
│   └── models/            # Saved model artifacts
├── requirements.txt
└── setup.py
```

---

## Setup

```bash
# Clone and install
git clone https://github.com/armanesh/car-price-prediction.git
cd car-price-prediction
pip install -r requirements.txt
pip install -e .
```

**Get the data:**
1. Download `vehicles.csv` from [Kaggle](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)
2. Place it in `data/raw/vehicles.csv`

---

## Run the Pipeline

```bash
python -m src.pipeline
```

This will:
1. Load and clean the raw data (filters extreme prices, old vehicles, high mileage)
2. Engineer features (`vehicle_age`, `age × odometer`)
3. Encode categoricals and split train/test (80/20)
4. Train Random Forest and XGBoost with 5-fold cross-validation
5. Print evaluation metrics and top feature importances
6. Save models to `outputs/models/`

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Key Design Choices

**Feature engineering:** `vehicle_age` (derived from year) outperforms raw year as a feature. The interaction term `age × odometer` captures that high mileage matters more for older cars.

**No target encoding:** Used `OrdinalEncoder` with `unknown_value=-1` to handle unseen categories safely at inference time, keeping the pipeline simple and reproducible without leakage risk.

**Model selection:** XGBoost with histogram-based tree building (`tree_method="hist"`) for efficiency on ~400k rows. Cross-validation run on training set only, test set held out until final evaluation.

---

## Top Predictive Features (XGBoost)

Based on feature importance scores:

1. `odometer` — mileage is the strongest signal after price range
2. `vehicle_age` — age degrades value non-linearly
3. `age_x_odometer` — interaction captures wear compounding
4. `manufacturer` — brand premium / discount
5. `type` — trucks/SUVs vs sedans have distinct price distributions

---

## Author

**Ali Rahbarimanesh** — Data Scientist & AI Engineer  
[LinkedIn](https://linkedin.com/in/armanesh) · [GitHub](https://github.com/armanesh)
