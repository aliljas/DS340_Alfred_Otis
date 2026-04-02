from pathlib import Path
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    median_absolute_error
)

# ======================
# PATHS
# ======================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = BASE_DIR / "data/processed/na_pm25_cells_clean.csv"
OUTPUT_FILE = BASE_DIR / "data/processed/catboost_predictions_2023.csv"

# ======================
# LOAD DATA
# ======================
print("\n--- LOADING DATA ---")
df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["lat", "lon", "date"])

print("Rows:", len(df))

# ======================
# FEATURE ENGINEERING
# ======================
print("\n--- FEATURE ENGINEERING ---")

df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year
df["time_index"] = (df["year"] - df["year"].min()) * 12 + df["month"]

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

grouped = df.groupby(["lat", "lon"])["pm25"]

# Core lags
df["pm25_lag1"] = grouped.shift(1)
df["pm25_lag2"] = grouped.shift(2)
df["pm25_lag3"] = grouped.shift(3)
df["pm25_lag6"] = grouped.shift(6)
df["pm25_lag12"] = grouped.shift(12)

# Rolling means using only past data
df["pm25_roll3_mean"] = grouped.shift(1).rolling(3).mean()
df["pm25_roll6_mean"] = grouped.shift(1).rolling(6).mean()
df["pm25_roll12_mean"] = grouped.shift(1).rolling(12).mean()

# Rolling std to capture volatility
df["pm25_roll3_std"] = grouped.shift(1).rolling(3).std()
df["pm25_roll6_std"] = grouped.shift(1).rolling(6).std()

# Recent changes / momentum
df["pm25_diff1"] = df["pm25_lag1"] - df["pm25_lag2"]
df["pm25_diff2"] = df["pm25_lag2"] - df["pm25_lag3"]
df["pm25_vs_roll3"] = df["pm25_lag1"] - df["pm25_roll3_mean"]
df["pm25_vs_roll6"] = df["pm25_lag1"] - df["pm25_roll6_mean"]

# Long-run cell baseline
df["cell_pm25_mean"] = grouped.transform("mean")

# Drop rows with missing engineered values
df = df.dropna().copy()

print("After features:", len(df))

# ======================
# SPLIT
# ======================
train = df[df["year"] <= 2020]
val   = df[df["year"] == 2021]
test  = df[df["year"] == 2022]

features = [
    "lat", "lon",
    "time_index",
    "month_sin", "month_cos",
    "pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_lag6", "pm25_lag12",
    "pm25_roll3_mean", "pm25_roll6_mean", "pm25_roll12_mean",
    "pm25_roll3_std", "pm25_roll6_std",
    "pm25_diff1", "pm25_diff2",
    "pm25_vs_roll3", "pm25_vs_roll6",
    "cell_pm25_mean"
]
target = "pm25"

X_train, y_train = train[features], train[target]
X_val, y_val     = val[features], val[target]
X_test, y_test   = test[features], test[target]

print("\nTrain:", len(X_train), "Val:", len(X_val), "Test:", len(X_test))

# ======================
# MODEL
# ======================
model = CatBoostRegressor(
    iterations=400,
    learning_rate=0.03,
    depth=6,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    verbose=100,
    use_best_model=True,
    early_stopping_rounds=40,
    allow_writing_files=False,

    # regularization
    l2_leaf_reg=10,
    rsm=0.8,
    random_strength=1,
    min_data_in_leaf=50,

    thread_count=-1
)

print("\n--- TRAINING CATBOOST ---")
model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val)
)
print("--- DONE ---")

print("Best iteration:", model.get_best_iteration())

# ======================
# EVALUATION HELPERS
# ======================
def compute_metrics(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "MedianAE": median_absolute_error(y_true, y_pred),
        "Bias": np.mean(y_pred - y_true)
    }

def print_metrics(name, metrics):
    print(f"\n{name}")
    print(f"RMSE: {metrics['RMSE']:.6f}")
    print(f"MAE: {metrics['MAE']:.6f}")
    print(f"R^2: {metrics['R2']:.6f}")
    print(f"Median AE: {metrics['MedianAE']:.6f}")
    print(f"Bias: {metrics['Bias']:.6f}")

# ======================
# EVALUATION
# ======================
val_pred = model.predict(X_val)
test_pred = model.predict(X_test)
naive_pred = X_test["pm25_lag1"].values

val_metrics = compute_metrics(y_val, val_pred)
test_metrics = compute_metrics(y_test, test_pred)
naive_metrics = compute_metrics(y_test, naive_pred)

print("\n--- RESULTS ---")
print_metrics("Validation Metrics", val_metrics)
print_metrics("Test Metrics", test_metrics)
print_metrics("Naive Test Metrics", naive_metrics)

# Optional: save metrics to CSV
metrics_df = pd.DataFrame([
    {"Dataset": "Validation", **val_metrics},
    {"Dataset": "Test", **test_metrics},
    {"Dataset": "Naive_Test", **naive_metrics}
])
metrics_output = BASE_DIR / "data/processed/catboost_eval_metrics.csv"
metrics_df.to_csv(metrics_output, index=False)

print("\nSaved evaluation metrics to:", metrics_output)

# ======================
# FORECAST 2023
# ======================
print("\n--- FORECASTING 2023 ---")

current_df = df[df["date"] == df["date"].max()].copy()

future_steps = 12
all_preds = []

for step in range(future_steps):
    current_df["date"] += pd.DateOffset(months=1)

    current_df["month"] = current_df["date"].dt.month
    current_df["year"] = current_df["date"].dt.year
    current_df["time_index"] = (
        (current_df["year"] - df["year"].min()) * 12 + current_df["month"]
    )
    current_df["month_sin"] = np.sin(2 * np.pi * current_df["month"] / 12)
    current_df["month_cos"] = np.cos(2 * np.pi * current_df["month"] / 12)

    # update lags
    old_lag1 = current_df["pm25_lag1"].copy()
    old_lag2 = current_df["pm25_lag2"].copy()
    old_lag3 = current_df["pm25_lag3"].copy()
    old_lag6 = current_df["pm25_lag6"].copy()

    current_df["pm25_lag12"] = current_df["pm25_lag6"]
    current_df["pm25_lag6"] = current_df["pm25_lag3"]
    current_df["pm25_lag3"] = current_df["pm25_lag2"]
    current_df["pm25_lag2"] = current_df["pm25_lag1"]
    current_df["pm25_lag1"] = current_df["pm25"]

    # recompute rolling / diff features approximately from updated lag history
    current_df["pm25_roll3_mean"] = (
        current_df["pm25_lag1"] + current_df["pm25_lag2"] + current_df["pm25_lag3"]
    ) / 3

    current_df["pm25_roll6_mean"] = (
        current_df["pm25_lag1"] + current_df["pm25_lag2"] + current_df["pm25_lag3"] +
        old_lag3 + old_lag2 + old_lag1
    ) / 6

    current_df["pm25_roll12_mean"] = (
        current_df["pm25_roll6_mean"] + current_df["pm25_lag12"]
    ) / 2

    current_df["pm25_roll3_std"] = np.std(
        np.vstack([
            current_df["pm25_lag1"].values,
            current_df["pm25_lag2"].values,
            current_df["pm25_lag3"].values
        ]),
        axis=0
    )

    current_df["pm25_roll6_std"] = np.std(
        np.vstack([
            current_df["pm25_lag1"].values,
            current_df["pm25_lag2"].values,
            current_df["pm25_lag3"].values,
            old_lag3.values,
            old_lag2.values,
            old_lag1.values
        ]),
        axis=0
    )

    current_df["pm25_diff1"] = current_df["pm25_lag1"] - current_df["pm25_lag2"]
    current_df["pm25_diff2"] = current_df["pm25_lag2"] - current_df["pm25_lag3"]
    current_df["pm25_vs_roll3"] = current_df["pm25_lag1"] - current_df["pm25_roll3_mean"]
    current_df["pm25_vs_roll6"] = current_df["pm25_lag1"] - current_df["pm25_roll6_mean"]

    preds = model.predict(current_df[features])
    current_df["pm25"] = preds

    all_preds.append(current_df[["lat", "lon", "date", "pm25"]].copy())

final_df = pd.concat(all_preds, ignore_index=True)
final_df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved predictions to:", OUTPUT_FILE)
