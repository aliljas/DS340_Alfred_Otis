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

FEATURE_SET = os.getenv("XGB_FEATURE_SET", "trend_region")
TARGET_TRANSFORM = os.getenv("XGB_TARGET_TRANSFORM", "log1p")
SAVE_PLOT = os.getenv("XGB_SAVE_PLOT", "0") == "1"
SAVE_COMPARISON_PLOT = os.getenv("XGB_SAVE_COMPARISON_PLOT", "1") == "1"
COMPARE_ERA5 = os.getenv("XGB_COMPARE_ERA5", "1") == "1"
USE_ERA5 = os.getenv("XGB_USE_ERA5", "1") == "1"
ERA5_FEATURE_LEVEL = os.getenv("XGB_ERA5_FEATURE_LEVEL", "core")
RUN_FORECAST = os.getenv("XGB_RUN_FORECAST", "1") == "1"
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

PRED_OUTPUT_FILE = PROCESSED_DIR / "xgb_predictions_2023.csv"
METRICS_OUTPUT_FILE = PROCESSED_DIR / "xgb_eval_metrics.csv"
IMPORTANCE_OUTPUT_FILE = PROCESSED_DIR / "xgb_feature_importance.csv"
PLOT_OUTPUT_FILE = PROCESSED_DIR / "xgb_model_results.png"
COMPARISON_METRICS_OUTPUT_FILE = PROCESSED_DIR / "xgb_era5_comparison_metrics.csv"
COMPARISON_PLOT_OUTPUT_FILE = PROCESSED_DIR / "xgb_era5_comparison.png"


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


#One XGBoost scenario
def fit_scenario(label, feature_list, train_frame, val_frame, test_frame):
    X_train = train_frame[feature_list]
    X_val = val_frame[feature_list]
    X_test = test_frame[feature_list]
    y_train = transform_target(train_frame[TARGET].values)
    y_val = transform_target(val_frame[TARGET].values)
    y_val_raw = val_frame[TARGET].values
    y_test = test_frame[TARGET].values

    print(f"\n--- TRAINING {MODEL_NAME.upper()} ({label}) ---")
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

    val_pred = inverse_target(model.predict(X_val, iteration_range=(0, model.best_iteration + 1)))
    test_pred = inverse_target(model.predict(X_test, iteration_range=(0, model.best_iteration + 1)))
    importance_df = pd.DataFrame(
        {
            "feature": feature_list,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return {
        "label": label,
        "features": feature_list,
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "test_pred": test_pred,
        "val_metrics": compute_metrics(y_val_raw, val_pred),
        "test_metrics": compute_metrics(y_test, test_pred),
        "importance_df": importance_df,
    }


#Main summary plot
def save_main_plot(result):
    if not SAVE_PLOT:
        print("Skipped plot generation. Set XGB_SAVE_PLOT=1 to save plots.")
        return

    #Create a compact 3-panel summary figure
    fig = plt.figure(figsize=(18, 5))

    #Feature chart
    plt.subplot(1, 3, 1)
    top_imp = result["importance_df"].head(15).sort_values("importance")
    plt.barh(top_imp["feature"], top_imp["importance"])
    plt.title("Top 15 Feature Importance")

    #Predicted vs actual
    plt.subplot(1, 3, 2)
    plot_sample = min(100_000, len(result["y_test"]))
    sample_idx = np.random.default_rng(RANDOM_SEED).choice(
        len(result["y_test"]),
        size=plot_sample,
        replace=False,
    )
    plt.scatter(result["y_test"][sample_idx], result["test_pred"][sample_idx], alpha=0.25, s=5)
    lims = [
        min(float(result["y_test"].min()), float(np.min(result["test_pred"]))),
        max(float(result["y_test"].max()), float(np.max(result["test_pred"]))),
    ]
    plt.plot(lims, lims, "r--", linewidth=1)
    plt.xlabel("Actual PM2.5")
    plt.ylabel("Predicted PM2.5")
    plt.title("Predicted vs Actual (Test)")

    #Residual histogram
    plt.subplot(1, 3, 3)
    residuals = result["y_test"] - result["test_pred"]
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
        print("Skipped comparison plot generation. Set XGB_SAVE_COMPARISON_PLOT=1 to save it.")
        return

    #Create a 2x2 paper-style comparison figure
    fig = plt.figure(figsize=(12, 10))

    #R^2 comparison bars
    plt.subplot(2, 2, 1)
    #Set shared labels
    labels = ["Validation", "Test"]
    #Set bar positions
    x = np.arange(len(labels))
    #Set bar width
    width = 0.35
    #Collect baseline R^2 values
    baseline_r2 = [baseline_result["val_metrics"]["R2"], baseline_result["test_metrics"]["R2"]]
    #Collect ERA R^2 values
    enhanced_r2 = [enhanced_result["val_metrics"]["R2"], enhanced_result["test_metrics"]["R2"]]
    #Draw baseline bars
    plt.bar(x - width / 2, baseline_r2, width=width, label="Without ERA5")
    #Draw ERA bars
    plt.bar(x + width / 2, enhanced_r2, width=width, label="With ERA5")
    #Set x ticks
    plt.xticks(x, labels)
    #Set y label
    plt.ylabel("R^2")
    #Set panel title
    plt.title("XGBoost R^2 Before vs After ERA5")
    #Show legend
    plt.legend()

    #RMSE comparison bars
    plt.subplot(2, 2, 2)
    #Collect baseline RMSE values
    baseline_rmse = [baseline_result["val_metrics"]["RMSE"], baseline_result["test_metrics"]["RMSE"]]
    #Collect ERA RMSE values
    enhanced_rmse = [enhanced_result["val_metrics"]["RMSE"], enhanced_result["test_metrics"]["RMSE"]]
    #Draw baseline bars
    plt.bar(x - width / 2, baseline_rmse, width=width, label="Without ERA5")
    #Draw ERA bars
    plt.bar(x + width / 2, enhanced_rmse, width=width, label="With ERA5")
    #Set x ticks
    plt.xticks(x, labels)
    #Set y label
    plt.ylabel("RMSE")
    #Set panel title
    plt.title("XGBoost RMSE Before vs After ERA5")
    #Show legend
    plt.legend()

    #Predicted vs actual without ERA5
    plt.subplot(2, 2, 3)
    plot_sample = min(75_000, len(baseline_result["y_test"]))
    sample_idx = np.random.default_rng(RANDOM_SEED).choice(
        len(baseline_result["y_test"]),
        size=plot_sample,
        replace=False,
    )
    plt.scatter(
        baseline_result["y_test"][sample_idx],
        baseline_result["test_pred"][sample_idx],
        alpha=0.2,
        s=5,
    )
    lims = [
        min(float(baseline_result["y_test"].min()), float(np.min(baseline_result["test_pred"]))),
        max(float(baseline_result["y_test"].max()), float(np.max(baseline_result["test_pred"]))),
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
    lims = [
        min(float(enhanced_result["y_test"].min()), float(np.min(enhanced_result["test_pred"]))),
        max(float(enhanced_result["y_test"].max()), float(np.max(enhanced_result["test_pred"]))),
    ]
    plt.plot(lims, lims, "r--", linewidth=1)
    plt.xlabel("Actual PM2.5")
    plt.ylabel("Predicted PM2.5")
    plt.title("With ERA5")

    #Tighten layout
    plt.tight_layout()
    #Save figure
    plt.savefig(COMPARISON_PLOT_OUTPUT_FILE, dpi=150)
    #Close figure to save memory
    plt.close()

    print("Saved ERA5 comparison plot to:", COMPARISON_PLOT_OUTPUT_FILE)


#Load/features
#Build the shared modeling table before fitting XGBoost.
df, era5_feature_names = prepare_modeling_frame(
    DATA_FILE,
    RAW_DIR,
    train_end=TRAIN_END,
    sample_cell_count=None,
    random_seed=RANDOM_SEED,
    use_era5=USE_ERA5,
    era5_feature_level=ERA5_FEATURE_LEVEL,
    target=TARGET,
)

#Split
#Use full 2021 for validation and full 2022 for the holdout test year.
train = df[df["date"] < TRAIN_END].copy()
val = df[(df["date"] >= TRAIN_END) & (df["date"] < VAL_END)].copy()
test = df[df["date"] >= VAL_END].copy()

feature_sets_no_era5 = build_feature_sets([])
feature_sets_with_era5 = build_feature_sets(era5_feature_names)
if FEATURE_SET not in feature_sets_with_era5:
    raise ValueError(
        f"Unknown XGB_FEATURE_SET={FEATURE_SET!r}. "
        f"Choose from: {', '.join(feature_sets_with_era5)}"
    )

baseline_features = feature_sets_no_era5[FEATURE_SET]
enhanced_features = feature_sets_with_era5[FEATURE_SET]
active_features = enhanced_features if era5_feature_names else baseline_features

print(f"\nFeature set: {FEATURE_SET}")
print(f"Feature count: {len(active_features)}")
print(f"Target transform: {TARGET_TRANSFORM}")
print(f"ERA5 feature level: {ERA5_FEATURE_LEVEL}")
print(f"\nTrain: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

#Optional baseline
#Train the no-ERA5 scenario first when we want a paper-ready comparison.
baseline_result = None
if COMPARE_ERA5 and era5_feature_names:
    baseline_result = fit_scenario("Without ERA5", baseline_features, train, val, test)

#Set the active scenario label
enhanced_label = "With ERA5" if era5_feature_names else "No ERA5 Available"

#Fit the active scenario
enhanced_result = fit_scenario(enhanced_label, active_features, train, val, test)

#Build the lag-1 benchmark
naive_pred = enhanced_result["X_test"]["pm25_lag1"].values

#Score the lag-1 benchmark
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

#Feature importance
enhanced_result["importance_df"].to_csv(IMPORTANCE_OUTPUT_FILE, index=False)
print("Saved feature importance to:", IMPORTANCE_OUTPUT_FILE)
print("\nTop 15 Features:")
print(enhanced_result["importance_df"].head(15).to_string(index=False))

if baseline_result is not None:
    #Build comparison table
    comparison_df = pd.DataFrame(
        [
            {"Scenario": "Without_ERA5", "Dataset": "Validation", "FeatureSet": FEATURE_SET, **baseline_result["val_metrics"]},
            {"Scenario": "Without_ERA5", "Dataset": "Test", "FeatureSet": FEATURE_SET, **baseline_result["test_metrics"]},
            {"Scenario": "With_ERA5", "Dataset": "Validation", "FeatureSet": FEATURE_SET, **enhanced_result["val_metrics"]},
            {"Scenario": "With_ERA5", "Dataset": "Test", "FeatureSet": FEATURE_SET, **enhanced_result["test_metrics"]},
        ]
    )
    #Save comparison CSV
    comparison_df.to_csv(COMPARISON_METRICS_OUTPUT_FILE, index=False)
    #Print save path
    print("Saved ERA5 comparison metrics to:", COMPARISON_METRICS_OUTPUT_FILE)
    #Save comparison figure
    save_comparison_plot(baseline_result, enhanced_result)
else:
    #Explain skipped comparison
    print("Skipped ERA5 comparison outputs because the baseline scenario was not run.")

#Save optional main figure
save_main_plot(enhanced_result)

#Run baseline forecast only
if RUN_FORECAST and not USE_ERA5 and not COMPARE_ERA5:
    #Start forecast section
    print("\n--- FORECASTING 2023 ---")

    #Keep the observed history
    history = df[["lat", "lon", "date", TARGET]].copy()

    #Store each forecast month
    future_preds = []

    #Track the latest known month
    latest_date = history["date"].max()

    #Reuse training medians for fill
    train_feature_fill_values = train[active_features].median(numeric_only=True)

    #Forecast 12 months
    for step in range(12):
        #Advance one month
        next_date = latest_date + pd.DateOffset(months=1)

        #Copy the latest grid
        base = history[history["date"] == latest_date][["lat", "lon"]].copy()

        #Set the future month
        base["date"] = next_date

        #Append blank target rows
        temp = pd.concat([history, base.assign(pm25=np.nan)], ignore_index=True)

        #Sort for lag building
        temp = temp.sort_values(["lat", "lon", "date"]).reset_index(drop=True)

        #Add lag features
        temp = add_history_features(temp, target=TARGET)

        #Add climatology
        temp = add_train_only_climatology(temp, train_end=TRAIN_END, target=TARGET)

        #Add regional features
        temp = add_experimental_features(temp)

        #Keep forecast causality clean
        temp, future_era5_feature_names = add_era5_features(
            temp,
            raw_dir=RAW_DIR,
            train_end=TRAIN_END,
            use_era5=False,
        )

        #Rebuild the feature set
        future_feature_sets = build_feature_sets(future_era5_feature_names)

        #Match the trained feature set
        future_features = future_feature_sets[FEATURE_SET]

        #Keep just the next-month rows
        future_rows = temp[temp["date"] == next_date].copy()

        #Build the forecast matrix
        X_future = future_rows.loc[:, future_features].copy()

        #Fill from training medians
        X_future = X_future.fillna(train_feature_fill_values)

        #Fill any leftovers
        X_future = X_future.fillna(0.0)

        #Predict next month
        preds = inverse_target(
            enhanced_result["model"].predict(
                X_future,
                iteration_range=(0, enhanced_result["model"].best_iteration + 1),
            )
        )

        #Clip to train range
        preds = np.clip(
            preds,
            float(train[TARGET].min()),
            float(train[TARGET].max()),
        )

        #Write forecast target
        future_rows[TARGET] = preds

        #Append to history
        history = pd.concat(
            [history, future_rows[["lat", "lon", "date", TARGET]]],
            ignore_index=True,
        )

        #Save this month
        future_preds.append(future_rows[["lat", "lon", "date", TARGET]].copy())

        #Advance pointer
        latest_date = next_date

    #Combine all months
    final_df = pd.concat(future_preds, ignore_index=True)

    #Save forecast CSV
    final_df.to_csv(PRED_OUTPUT_FILE, index=False)

    #Print forecast path
    print("\nSaved predictions to:", PRED_OUTPUT_FILE)

#Skip forecast in ERA or comparison mode
elif RUN_FORECAST:
    #Explain skipped forecast
    print(
        "\nSkipped forecasting because this run used "
        "ERA5 or comparison mode. The baseline no-ERA run remains the strict "
        "forecast-ready XGBoost path."
    )
