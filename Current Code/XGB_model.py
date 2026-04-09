import os
import tempfile
from pathlib import Path

#Cross-platform Matplotlib cache
MPLCONFIGDIR = Path(tempfile.gettempdir()) / "matplotlib-cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        median_absolute_error,
        r2_score,
    )
    from xgboost import XGBRegressor

    from model_feature_utils import (
        TARGET,
        add_era5_features,
        add_experimental_features,
        add_history_features,
        add_train_only_climatology,
        build_feature_sets,
        prepare_modeling_frame,
    )
except ImportError as exc:
    raise SystemExit(
        "Required packages for XGB_model.py are not installed. "
        "Install the project requirements first, then rerun this script."
    ) from exc


#Paths/config
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data/processed"
RAW_DIR = BASE_DIR / "data/raw"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_FILE = PROCESSED_DIR / "na_pm25_cells_clean.csv"

TRAIN_END = pd.Timestamp("2021-01-01")
VAL_END = pd.Timestamp("2022-01-01")
RANDOM_SEED = 42

#Quick sample option
SAMPLE_CELL_COUNT = int(os.getenv("XGB_SAMPLE_CELLS", "0")) or None
FEATURE_SET = os.getenv("XGB_FEATURE_SET", "trend_region")
TARGET_TRANSFORM = os.getenv("XGB_TARGET_TRANSFORM", "log1p")
SAVE_PLOT = os.getenv("XGB_SAVE_PLOT", "0") == "1"
USE_ERA5 = os.getenv("XGB_USE_ERA5", "1") == "1"
MODEL_NAME = "XGBoost"

XGB_PARAMS = {
    "n_estimators": 1500,
    "learning_rate": 0.05,
    "max_depth": 7,
    "min_child_weight": 12,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.2,
    "reg_lambda": 4.0,
    "gamma": 0.02,
    "tree_method": "hist",
    "objective": "reg:squarederror",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

RUN_SUFFIX = "" if SAMPLE_CELL_COUNT is None else f"_sample_{SAMPLE_CELL_COUNT}_cells"
PRED_OUTPUT_FILE = PROCESSED_DIR / f"xgb_predictions_2023{RUN_SUFFIX}.csv"
METRICS_OUTPUT_FILE = PROCESSED_DIR / f"xgb_eval_metrics{RUN_SUFFIX}.csv"
IMPORTANCE_OUTPUT_FILE = PROCESSED_DIR / f"xgb_feature_importance{RUN_SUFFIX}.csv"
PLOT_OUTPUT_FILE = PROCESSED_DIR / f"xgb_model_results{RUN_SUFFIX}.png"


#Metric helpers
def compute_metrics(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MedianAE": float(median_absolute_error(y_true, y_pred)),
        "Bias": float(np.mean(y_pred - y_true)),
    }


#Print helper
def print_metrics(name, metrics):
    print(f"\n{name}")
    print(f"RMSE:      {metrics['RMSE']:.6f}")
    print(f"MAE:       {metrics['MAE']:.6f}")
    print(f"R^2:       {metrics['R2']:.6f}")
    print(f"Median AE: {metrics['MedianAE']:.6f}")
    print(f"Bias:      {metrics['Bias']:.6f}")


#Target transform
def transform_target(values):
    if TARGET_TRANSFORM == "log1p":
        return np.log1p(values)
    if TARGET_TRANSFORM == "none":
        return values
    raise ValueError("XGB_TARGET_TRANSFORM must be 'log1p' or 'none'")


#Target inverse transform
def inverse_target(values):
    if TARGET_TRANSFORM == "log1p":
        return np.expm1(values)
    return values


#Load/features
#Build the shared modeling table before fitting XGBoost.
df, era5_feature_names = prepare_modeling_frame(
    DATA_FILE,
    RAW_DIR,
    train_end=TRAIN_END,
    sample_cell_count=SAMPLE_CELL_COUNT,
    random_seed=RANDOM_SEED,
    use_era5=USE_ERA5,
    target=TARGET,
)

#Split
#Use full 2021 for validation and full 2022 for the holdout test year.
train = df[df["date"] < TRAIN_END].copy()
val = df[(df["date"] >= TRAIN_END) & (df["date"] < VAL_END)].copy()
test = df[df["date"] >= VAL_END].copy()

feature_sets = build_feature_sets(era5_feature_names)
if FEATURE_SET not in feature_sets:
    raise ValueError(
        f"Unknown XGB_FEATURE_SET={FEATURE_SET!r}. "
        f"Choose from: {', '.join(feature_sets)}"
    )

features = feature_sets[FEATURE_SET]

X_train = train[features]
X_val = val[features]
X_test = test[features]
y_train = transform_target(train[TARGET].values)
y_val = transform_target(val[TARGET].values)
y_val_raw = val[TARGET].values
y_test = test[TARGET].values

print(f"\nFeature set: {FEATURE_SET}")
print(f"Feature count: {len(features)}")
print(f"Target transform: {TARGET_TRANSFORM}")
print(f"\nTrain: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

#Model
#XGBoost uses the same engineered panel with histogram tree building.
print(f"\n--- TRAINING {MODEL_NAME.upper()} ---")
model = XGBRegressor(
    eval_metric="rmse",
    early_stopping_rounds=75,
    **XGB_PARAMS,
)
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=100,
)
print("--- DONE ---")
print("Best iteration:", model.best_iteration)

#Evaluation
val_pred = inverse_target(model.predict(X_val, iteration_range=(0, model.best_iteration + 1)))
test_pred = inverse_target(model.predict(X_test, iteration_range=(0, model.best_iteration + 1)))
naive_pred = X_test["pm25_lag1"].values

val_metrics = compute_metrics(y_val_raw, val_pred)
test_metrics = compute_metrics(y_test, test_pred)
naive_metrics = compute_metrics(y_test, naive_pred)

print("\n--- FINAL RESULTS ---")
print_metrics(f"{MODEL_NAME} Validation Metrics", val_metrics)
print_metrics(f"{MODEL_NAME} Test Metrics", test_metrics)
print_metrics("Naive Test Metrics", naive_metrics)

metrics_df = pd.DataFrame(
    [
        {"Dataset": "Validation", "Model": MODEL_NAME, "FeatureSet": FEATURE_SET, **val_metrics},
        {"Dataset": "Test", "Model": MODEL_NAME, "FeatureSet": FEATURE_SET, **test_metrics},
        {"Dataset": "Naive_Test", "Model": "pm25_lag1", **naive_metrics},
    ]
)
metrics_df.to_csv(METRICS_OUTPUT_FILE, index=False)
print("\nSaved evaluation metrics to:", METRICS_OUTPUT_FILE)

#Feature importance
importance_df = pd.DataFrame(
    {
        "feature": features,
        "importance": model.feature_importances_,
    }
).sort_values("importance", ascending=False)
importance_df.to_csv(IMPORTANCE_OUTPUT_FILE, index=False)
print("Saved feature importance to:", IMPORTANCE_OUTPUT_FILE)
print("\nTop 15 Features:")
print(importance_df.head(15).to_string(index=False))

#Plots
if SAVE_PLOT:
    #Create a compact 3-panel summary figure
    fig = plt.figure(figsize=(18, 5))

    #Feature chart
    plt.subplot(1, 3, 1)
    top_imp = importance_df.head(15).sort_values("importance")
    plt.barh(top_imp["feature"], top_imp["importance"])
    plt.title("Top 15 Feature Importance")

    #Predicted vs actual
    plt.subplot(1, 3, 2)
    plot_sample = min(100_000, len(y_test))
    sample_idx = np.random.default_rng(RANDOM_SEED).choice(len(y_test), size=plot_sample, replace=False)
    plt.scatter(y_test[sample_idx], test_pred[sample_idx], alpha=0.25, s=5)
    lims = [
        min(float(y_test.min()), float(np.min(test_pred))),
        max(float(y_test.max()), float(np.max(test_pred))),
    ]
    plt.plot(lims, lims, "r--", linewidth=1)
    plt.xlabel("Actual PM2.5")
    plt.ylabel("Predicted PM2.5")
    plt.title("Predicted vs Actual (Test)")

    #Residual histogram
    plt.subplot(1, 3, 3)
    residuals = y_test - test_pred
    plt.hist(residuals, bins=60)
    plt.axvline(0, linewidth=1)
    plt.xlabel("Residual (Actual - Predicted)")
    plt.title("Residual Distribution")

    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_FILE, dpi=150)
    plt.close()

    print("Saved plot to:", PLOT_OUTPUT_FILE)
else:
    print("Skipped plot generation. Set XGB_SAVE_PLOT=1 to save plots.")

#Forecast 2023
#Kept commented while tuning.
#For a strict future forecast, same-month ERA5 should stay disabled unless
#you have forecast meteorology rather than reanalysis.
#
# print("\n--- FORECASTING 2023 ---")
# #Start from the fully observed history table
# history = df[["lat", "lon", "date", TARGET]].copy()
# future_preds = []
# latest_date = history["date"].max()
#
# for _ in range(12):
#     #Advance one month at a time
#     next_date = latest_date + pd.DateOffset(months=1)
#
#     #Create one placeholder row per grid cell for the next month
#     base = history[history["date"] == latest_date][["lat", "lon"]].copy()
#     base["date"] = next_date
#
#     #Rebuild the full causal feature table with the new placeholder month
#     temp = pd.concat([history, base.assign(pm25=np.nan)], ignore_index=True)
#     temp = temp.sort_values(["lat", "lon", "date"]).reset_index(drop=True)
#     temp = add_history_features(temp, target=TARGET)
#     temp = add_train_only_climatology(temp, train_end=TRAIN_END, target=TARGET)
#     temp = add_experimental_features(temp)
#     #Disable same-month ERA5 for strict forecasting mode
#     temp, future_era5_feature_names = add_era5_features(
#         temp,
#         raw_dir=RAW_DIR,
#         train_end=TRAIN_END,
#         use_era5=False,
#     )
#
#     #Select the same feature set used during model training
#     future_feature_sets = build_feature_sets(future_era5_feature_names)
#     future_features = future_feature_sets[FEATURE_SET]
#     future_rows = temp[temp["date"] == next_date].copy()
#     future_rows = future_rows.dropna(subset=future_features)
#
#     #Predict the next month and invert the target transform
#     preds = inverse_target(
#         model.predict(
#             future_rows[future_features],
#             iteration_range=(0, model.best_iteration + 1),
#         )
#     )
#     future_rows[TARGET] = preds
#
#     #Append predictions back into history for the next recursive step
#     history = pd.concat(
#         [history, future_rows[["lat", "lon", "date", TARGET]]],
#         ignore_index=True,
#     )
#     future_preds.append(future_rows[["lat", "lon", "date", TARGET]].copy())
#     latest_date = next_date
#
# #Combine all recursive forecast months into one output file
# final_df = pd.concat(future_preds, ignore_index=True)
# final_df.to_csv(PRED_OUTPUT_FILE, index=False)
# print("\nSaved predictions to:", PRED_OUTPUT_FILE)
