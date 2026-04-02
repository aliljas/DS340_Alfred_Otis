import pandas as pd
import numpy as np
import optuna
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from pathlib import Path
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from xgboost import plot_importance

# --- CONFIG ---
BASE_DIR = Path(r"C:\Users\user\Downloads")
DATA_FILE = BASE_DIR / "na_pm25_cells_clean.csv"

# --- LOAD & OPTIMIZE MEMORY ---
print("Loading data...")
df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])
df[['lat', 'lon', 'pm25']] = df[['lat', 'lon', 'pm25']].astype(np.float32)

# Feature Engineering
# ======================
# FEATURE ENGINEERING (ENHANCED)
# ======================
print("\n--- ENHANCED FEATURE ENGINEERING ---")

# 1. Basic Time Features
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).astype(np.float32)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).astype(np.float32)

# 2. Spatial Momentum (Neighbor Lags)
# We sort by date first to ensure we are looking at the same "snapshot" of the map
# Step A: Get the neighbor's value for the current month
df = df.sort_values(["date", "lat", "lon"])
df["temp_west"] = df.groupby("date")["pm25"].shift(1)
df["temp_east"] = df.groupby("date")["pm25"].shift(-1)

# Step B: Shift those neighbor values by 1 month so they are "Lagged"
df = df.sort_values(["lat", "lon", "date"])
df["pm25_west_lag1"] = df.groupby(["lat", "lon"])["temp_west"].shift(1).astype(np.float32)
df["pm25_east_lag1"] = df.groupby(["lat", "lon"])["temp_east"].shift(1).astype(np.float32)

# Clean up temporary columns
df = df.drop(columns=["temp_west", "temp_east"])

# 3. Temporal Momentum (Lag + Rolling Mean)
df = df.sort_values(["lat", "lon", "date"])
print("Adding temporal trends...")
# Previous month
df["pm25_lag1"] = df.groupby(["lat", "lon"])["pm25"].shift(1).astype(np.float32)
# 3-month rolling average (captures seasonal persistence)
df["pm25_rolling_3mo"] = df.groupby(["lat", "lon"])["pm25"].transform(
    lambda x: x.shift(1).rolling(window=3).mean()
).astype(np.float32)

# Drop rows where we don't have enough history/neighbors
df = df.dropna()

FEATURES = [
    "lat", "lon", 
    "month_sin", "month_cos", 
    "pm25_lag1", 
    "pm25_west_lag1",  # Note the '_lag1' name change
    "pm25_east_lag1",  # Note the '_lag1' name change
    "pm25_rolling_3mo"
]

# Chronological Split
train_df = df[df["date"] < "2022-01-01"]
val_df = df[(df["date"] >= "2022-01-01") & (df["date"] < "2022-07-01")]
test_df = df[df["date"] >= "2022-07-01"]

TARGET = "pm25"

# --- OPTUNA OBJECTIVE ---
def objective(trial):
    # Search space
    param = {
        "n_estimators": 500,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "tree_method": "hist",
        "early_stopping_rounds": 50,
        "random_state": 42
    }
    
    # Use a smaller subset for tuning (e.g., 10% of training data) to save time
    tune_df = train_df.sample(n=min(1000000, len(train_df)), random_state=42)
    
    model = XGBRegressor(**param)
    model.fit(
        tune_df[FEATURES], tune_df[TARGET],
        eval_set=[(val_df[FEATURES], val_df[TARGET])],
        verbose=False
    )
    
    preds = model.predict(val_df[FEATURES])
    rmse = mean_squared_error(val_df[TARGET], preds)**0.5
    return rmse

# --- RUN THE STUDY ---
print("Starting Hyperparameter Optimization...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20) # 20 trials is a good start

print("\nBest Parameters:", study.best_params)

# --- FINAL TRAINING ---
print("\nTraining final model on full dataset with best parameters...")
final_model = XGBRegressor(
    **study.best_params, 
    n_estimators=1000, 
    tree_method="hist",
    early_stopping_rounds=50
)

final_model.fit(
    train_df[FEATURES], train_df[TARGET],
    eval_set=[(val_df[FEATURES], val_df[TARGET])],
    verbose=100
)

# --- FINAL EVALUATION ---
test_preds = final_model.predict(test_df[FEATURES])
final_rmse = mean_squared_error(test_df[TARGET], test_preds)**0.5
print(f"\nFinal Test RMSE with Optuna: {final_rmse}")

test_actuals = test_df[TARGET]
test_r2 = r2_score(test_actuals, test_preds)

print(f"Final Test R² Score: {test_r2:.4f}")

print("\n--- GENERATING FEATURE IMPORTANCE PLOT ---")
fig, ax = plt.subplots(figsize=(10, 8))
plot_importance(final_model, ax=ax, importance_type='gain') # 'gain' is best for accuracy
plt.title("Feature Importance (Information Gain)")
plt.show()

# 2. Print Numerical Scores
importance_scores = final_model.get_booster().get_score(importance_type='gain')
sorted_scores = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)

print("\nFeature Importance Rankings (Gain):")
for feature, score in sorted_scores:
    print(f"{feature}: {score:.2f}")