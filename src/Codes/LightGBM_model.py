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
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        median_absolute_error,
        r2_score,
    )

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
        "Required packages for LightGBM_model.py are not installed. "
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
SAMPLE_CELL_COUNT = int(os.getenv("LIGHTGBM_SAMPLE_CELLS", "0")) or None
FEATURE_SET = os.getenv("LIGHTGBM_FEATURE_SET", "trend_region_era5plus")
SAVE_PLOT = os.getenv("LIGHTGBM_SAVE_PLOT", "0") == "1"
SAVE_COMPARISON_PLOT = os.getenv("LIGHTGBM_SAVE_COMPARISON_PLOT", "1") == "1"
COMPARE_ERA5 = os.getenv("LIGHTGBM_COMPARE_ERA5", "1") == "1"
MODEL_NAME = "LightGBM"
TARGET_TRANSFORM = os.getenv("LIGHTGBM_TARGET_TRANSFORM", "log1p")
CATEGORICAL_FEATURES = ["month", "region_lat_bin", "region_lon_bin"]
USE_ERA5 = os.getenv("LIGHTGBM_USE_ERA5", "1") == "1"

LIGHTGBM_PARAMS = {
    "n_estimators": 1200,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 200,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.2,
    "reg_lambda": 4.0,
    "min_split_gain": 0.02,
    "max_bin": 127,
    "force_col_wise": True,
    "random_state": RANDOM_SEED,
    "objective": "regression",
    "n_jobs": -1,
    "verbosity": -1,
}

RUN_SUFFIX = "" if SAMPLE_CELL_COUNT is None else f"_sample_{SAMPLE_CELL_COUNT}_cells"
PRED_OUTPUT_FILE = PROCESSED_DIR / f"lightgbm_predictions_2023{RUN_SUFFIX}.csv"
METRICS_OUTPUT_FILE = PROCESSED_DIR / f"lightgbm_eval_metrics{RUN_SUFFIX}.csv"
IMPORTANCE_OUTPUT_FILE = PROCESSED_DIR / f"lightgbm_feature_importance{RUN_SUFFIX}.csv"
PLOT_OUTPUT_FILE = PROCESSED_DIR / f"lightgbm_model_results{RUN_SUFFIX}.png"
COMPARISON_METRICS_OUTPUT_FILE = PROCESSED_DIR / f"lightgbm_era5_comparison_metrics{RUN_SUFFIX}.csv"
COMPARISON_PLOT_OUTPUT_FILE = PROCESSED_DIR / f"lightgbm_era5_comparison{RUN_SUFFIX}.png"


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
    raise ValueError("LIGHTGBM_TARGET_TRANSFORM must be 'log1p' or 'none'")


#Target inverse transform
def inverse_target(values):
    if TARGET_TRANSFORM == "log1p":
        return np.expm1(values)
    return values


#Category casting
def prepare_model_frame(frame, features):
    prepared = frame[features].copy()
    for col in CATEGORICAL_FEATURES:
        if col in prepared.columns:
            prepared[col] = prepared[col].astype("category")
    return prepared


#Fit one scenario
def fit_scenario(label, features, train, val, test):
    X_train = prepare_model_frame(train, features)
    X_val = prepare_model_frame(val, features)
    X_test = prepare_model_frame(test, features)

    y_train = transform_target(train[TARGET].values)
    y_val = transform_target(val[TARGET].values)
    y_val_raw = val[TARGET].values
    y_test = test[TARGET].values

    print(f"\n--- TRAINING {MODEL_NAME.upper()} ({label}) ---")
    model = LGBMRegressor(**LIGHTGBM_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        categorical_feature=[col for col in CATEGORICAL_FEATURES if col in features],
        callbacks=[
            early_stopping(
                stopping_rounds=75,
                first_metric_only=True,
                verbose=True,
                min_delta=1e-4,
            ),
            log_evaluation(period=100),
        ],
    )
    print("--- DONE ---")
    print("Best iteration:", model.best_iteration_)

    val_pred = inverse_target(model.predict(X_val, num_iteration=model.best_iteration_))
    test_pred = inverse_target(model.predict(X_test, num_iteration=model.best_iteration_))

    return {
        "label": label,
        "model": model,
        "features": features,
        "X_test": X_test,
        "y_test": y_test,
        "val_pred": val_pred,
        "test_pred": test_pred,
        "val_metrics": compute_metrics(y_val_raw, val_pred),
        "test_metrics": compute_metrics(y_test, test_pred),
        "importance_df": pd.DataFrame(
            {
                "feature": features,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False),
    }


#Main plot
def save_main_plot(result):
    if not SAVE_PLOT:
        print("Skipped plot generation. Set LIGHTGBM_SAVE_PLOT=1 to save plots.")
        return

    y_test = result["y_test"]
    test_pred = result["test_pred"]
    importance_df = result["importance_df"]

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


#Before/after ERA5 plot
def save_comparison_plot(baseline_result, enhanced_result):
    if not SAVE_COMPARISON_PLOT:
        print("Skipped comparison plot generation. Set LIGHTGBM_SAVE_COMPARISON_PLOT=1 to save it.")
        return

    baseline_metrics = baseline_result["test_metrics"]
    enhanced_metrics = enhanced_result["test_metrics"]
    baseline_val_metrics = baseline_result["val_metrics"]
    enhanced_val_metrics = enhanced_result["val_metrics"]

    #Create a 2x2 paper-style comparison figure
    fig = plt.figure(figsize=(16, 10))

    #R^2 comparison
    plt.subplot(2, 2, 1)
    metric_names = ["Validation", "Test"]
    baseline_r2 = [baseline_val_metrics["R2"], baseline_metrics["R2"]]
    enhanced_r2 = [enhanced_val_metrics["R2"], enhanced_metrics["R2"]]
    x = np.arange(len(metric_names))
    width = 0.35
    plt.bar(x - width / 2, baseline_r2, width=width, label="Without ERA5")
    plt.bar(x + width / 2, enhanced_r2, width=width, label="With ERA5")
    plt.xticks(x, metric_names)
    plt.ylabel("R^2")
    plt.title("LightGBM R^2 Before vs After ERA5")
    plt.legend()

    #RMSE comparison
    plt.subplot(2, 2, 2)
    baseline_rmse = [baseline_val_metrics["RMSE"], baseline_metrics["RMSE"]]
    enhanced_rmse = [enhanced_val_metrics["RMSE"], enhanced_metrics["RMSE"]]
    plt.bar(x - width / 2, baseline_rmse, width=width, label="Without ERA5")
    plt.bar(x + width / 2, enhanced_rmse, width=width, label="With ERA5")
    plt.xticks(x, metric_names)
    plt.ylabel("RMSE")
    plt.title("LightGBM RMSE Before vs After ERA5")
    plt.legend()

    #Use the same test sample in both scatter panels
    plot_sample = min(100_000, len(enhanced_result["y_test"]))
    sample_idx = np.random.default_rng(RANDOM_SEED).choice(
        len(enhanced_result["y_test"]),
        size=plot_sample,
        replace=False,
    )

    #Predicted vs actual without ERA5
    plt.subplot(2, 2, 3)
    plt.scatter(
        baseline_result["y_test"][sample_idx],
        baseline_result["test_pred"][sample_idx],
        alpha=0.2,
        s=5,
    )
    lims = [
        min(
            float(enhanced_result["y_test"].min()),
            float(np.min(baseline_result["test_pred"])),
            float(np.min(enhanced_result["test_pred"])),
        ),
        max(
            float(enhanced_result["y_test"].max()),
            float(np.max(baseline_result["test_pred"])),
            float(np.max(enhanced_result["test_pred"])),
        ),
    ]
    plt.plot(lims, lims, "r--", linewidth=1)
    plt.xlabel("Actual PM2.5")
    plt.ylabel("Predicted PM2.5")
    plt.title("Without ERA5")

    #Predicted vs actual with ERA5
    plt.subplot(2, 2, 4)
    plt.scatter(
        enhanced_result["y_test"][sample_idx],
        enhanced_result["test_pred"][sample_idx],
        alpha=0.2,
        s=5,
    )
    plt.plot(lims, lims, "r--", linewidth=1)
    plt.xlabel("Actual PM2.5")
    plt.ylabel("Predicted PM2.5")
    plt.title("With ERA5")

    plt.tight_layout()
    plt.savefig(COMPARISON_PLOT_OUTPUT_FILE, dpi=150)
    plt.close()

    print("Saved ERA5 comparison plot to:", COMPARISON_PLOT_OUTPUT_FILE)


#Load/features
#Build the shared modeling table once, then train scenarios on top of it.
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
#Use full 2021 as validation and full 2022 as the holdout test year.
train = df[df["date"] < TRAIN_END].copy()
val = df[(df["date"] >= TRAIN_END) & (df["date"] < VAL_END)].copy()
test = df[df["date"] >= VAL_END].copy()

feature_sets_no_era5 = build_feature_sets([])
feature_sets_with_era5 = build_feature_sets(era5_feature_names)

if FEATURE_SET not in feature_sets_with_era5:
    raise ValueError(
        f"Unknown LIGHTGBM_FEATURE_SET={FEATURE_SET!r}. "
        f"Choose from: {', '.join(feature_sets_with_era5)}"
    )

baseline_features = feature_sets_no_era5[FEATURE_SET]
enhanced_features = feature_sets_with_era5[FEATURE_SET]
active_features = enhanced_features if era5_feature_names else baseline_features

print(f"\nFeature set: {FEATURE_SET}")
print(f"Feature count: {len(active_features)}")
print(f"Target transform: {TARGET_TRANSFORM}")
print(f"\nTrain: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

#Optional baseline
#Train the no-ERA5 scenario first when we want a paper-ready comparison.
baseline_result = None
if COMPARE_ERA5 and era5_feature_names:
    baseline_result = fit_scenario("Without ERA5", baseline_features, train, val, test)

enhanced_label = "With ERA5" if era5_feature_names else "No ERA5 Available"
enhanced_result = fit_scenario(enhanced_label, active_features, train, val, test)

naive_pred = enhanced_result["X_test"]["pm25_lag1"].values
naive_metrics = compute_metrics(enhanced_result["y_test"], naive_pred)

print("\n--- FINAL RESULTS ---")
print_metrics(f"{MODEL_NAME} Validation Metrics", enhanced_result["val_metrics"])
print_metrics(f"{MODEL_NAME} Test Metrics", enhanced_result["test_metrics"])
print_metrics("Naive Test Metrics", naive_metrics)

metrics_df = pd.DataFrame(
    [
        {"Dataset": "Validation", "Model": MODEL_NAME, "FeatureSet": FEATURE_SET, "Scenario": enhanced_label, **enhanced_result["val_metrics"]},
        {"Dataset": "Test", "Model": MODEL_NAME, "FeatureSet": FEATURE_SET, "Scenario": enhanced_label, **enhanced_result["test_metrics"]},
        {"Dataset": "Naive_Test", "Model": "pm25_lag1", "Scenario": "lag1", **naive_metrics},
    ]
)
metrics_df.to_csv(METRICS_OUTPUT_FILE, index=False)
print("\nSaved evaluation metrics to:", METRICS_OUTPUT_FILE)

enhanced_result["importance_df"].to_csv(IMPORTANCE_OUTPUT_FILE, index=False)
print("Saved feature importance to:", IMPORTANCE_OUTPUT_FILE)
print("\nTop 15 Features:")
print(enhanced_result["importance_df"].head(15).to_string(index=False))

if baseline_result is not None:
    comparison_df = pd.DataFrame(
        [
            {"Scenario": "Without_ERA5", "Dataset": "Validation", "FeatureSet": FEATURE_SET, **baseline_result["val_metrics"]},
            {"Scenario": "Without_ERA5", "Dataset": "Test", "FeatureSet": FEATURE_SET, **baseline_result["test_metrics"]},
            {"Scenario": "With_ERA5", "Dataset": "Validation", "FeatureSet": FEATURE_SET, **enhanced_result["val_metrics"]},
            {"Scenario": "With_ERA5", "Dataset": "Test", "FeatureSet": FEATURE_SET, **enhanced_result["test_metrics"]},
        ]
    )
    comparison_df.to_csv(COMPARISON_METRICS_OUTPUT_FILE, index=False)
    print("Saved ERA5 comparison metrics to:", COMPARISON_METRICS_OUTPUT_FILE)
    save_comparison_plot(baseline_result, enhanced_result)
else:
    print("Skipped ERA5 comparison outputs because the baseline scenario was not run.")

save_main_plot(enhanced_result)

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
#     X_future = prepare_model_frame(future_rows, future_features)
#     preds = inverse_target(
#         enhanced_result["model"].predict(
#             X_future,
#             num_iteration=enhanced_result["model"].best_iteration_,
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
