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
    from catboost import CatBoostRegressor
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
        "Required packages for CatBoost_model.py are not installed. "
        "Install the project requirements first, then rerun this script."
    ) from exc


#Paths/config
#Use project-relative paths
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data/processed"
RAW_DIR = BASE_DIR / "data/raw"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_FILE = PROCESSED_DIR / "na_pm25_cells_clean.csv"

#Train/val/test cutoffs
TRAIN_END = pd.Timestamp("2021-01-01")
VAL_END = pd.Timestamp("2022-01-01")
RANDOM_SEED = 42

#Choose feature bundle
FEATURE_SET = os.getenv("CATBOOST_FEATURE_SET", "base")
#Stabilize spikes
TARGET_TRANSFORM = os.getenv("CATBOOST_TARGET_TRANSFORM", "log1p")
#Optional plots
SAVE_PLOT = os.getenv("CATBOOST_SAVE_PLOT", "0") == "1"
SAVE_COMPARISON_PLOT = os.getenv("CATBOOST_SAVE_COMPARISON_PLOT", "1") == "1"
#Optional ERA comparison
COMPARE_ERA5 = os.getenv("CATBOOST_COMPARE_ERA5", "0") == "1"
#ERA is additive, not default
USE_ERA5 = os.getenv("CATBOOST_USE_ERA5", "0") == "1"
#Lean ERA feature batch
ERA5_FEATURE_LEVEL = os.getenv("CATBOOST_ERA5_FEATURE_LEVEL", "core")
#Speed up early stopping
EVAL_SAMPLE_ROWS = int(os.getenv("CATBOOST_EVAL_SAMPLE_ROWS", "400000")) or None
#Forecast on baseline runs
RUN_FORECAST = os.getenv("CATBOOST_RUN_FORECAST", "1") == "1"
#Forecast horizon
FORECAST_MONTHS = int(os.getenv("CATBOOST_FORECAST_MONTHS", "12"))
MODEL_NAME = "CatBoost"

#Faster stable CatBoost settings
CATBOOST_PARAMS = {
    "iterations": 900,
    "learning_rate": 0.06,
    "depth": 6,
    "l2_leaf_reg": 28,
    "random_strength": 2.0,
    "min_data_in_leaf": 80,
    "rsm": 0.75,
    "bagging_temperature": 1.0,
    "border_count": 128,
}

#Saved outputs
PRED_OUTPUT_FILE = PROCESSED_DIR / "catboost_predictions_2023.csv"
METRICS_OUTPUT_FILE = PROCESSED_DIR / "catboost_eval_metrics.csv"
IMPORTANCE_OUTPUT_FILE = PROCESSED_DIR / "catboost_feature_importance.csv"
PLOT_OUTPUT_FILE = PROCESSED_DIR / "catboost_model_results.png"
COMPARISON_METRICS_OUTPUT_FILE = PROCESSED_DIR / "catboost_era5_comparison_metrics.csv"
COMPARISON_PLOT_OUTPUT_FILE = PROCESSED_DIR / "catboost_era5_comparison.png"


#Metric helpers
#Compute paper metrics
def compute_metrics(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MedianAE": float(median_absolute_error(y_true, y_pred)),
        "Bias": float(np.mean(y_pred - y_true)),
    }


#Print helper
#Print metrics consistently
def print_metrics(name, metrics):
    print(f"\n{name}")
    print(f"RMSE:      {metrics['RMSE']:.6f}")
    print(f"MAE:       {metrics['MAE']:.6f}")
    print(f"R^2:       {metrics['R2']:.6f}")
    print(f"Median AE: {metrics['MedianAE']:.6f}")
    print(f"Bias:      {metrics['Bias']:.6f}")


#Target transform
#Train in transformed space
def transform_target(values):
    if TARGET_TRANSFORM == "log1p":
        return np.log1p(values)
    if TARGET_TRANSFORM == "none":
        return values
    raise ValueError("CATBOOST_TARGET_TRANSFORM must be 'log1p' or 'none'")


#Target inverse transform
#Return to PM2.5 scale
def inverse_target(values):
    if TARGET_TRANSFORM == "log1p":
        return np.expm1(values)
    return values

#Create helper that selects needed columns only
def prepare_model_frame(frame, feature_list, categorical_cols):
    #Avoid extra copies
    return frame.loc[:, feature_list]


#One CatBoost scenario
#Fit one scenario
def fit_scenario(label, feature_list, train_frame, val_frame, test_frame):
    #Categorical split columns
    categorical_cols = [col for col in ["month", "region_lat_bin", "region_lon_bin"] if col in feature_list]

    #Feature matrices
    X_train = prepare_model_frame(train_frame, feature_list, categorical_cols)
    X_val = prepare_model_frame(val_frame, feature_list, categorical_cols)
    X_test = prepare_model_frame(test_frame, feature_list, categorical_cols)

    #Targets
    y_train = transform_target(train_frame[TARGET].values)
    y_val = transform_target(val_frame[TARGET].values)
    y_val_raw = val_frame[TARGET].values
    y_test = test_frame[TARGET].values

    #Optional eval subset
    X_val_eval = X_val
    y_val_eval = y_val
    if EVAL_SAMPLE_ROWS is not None and len(X_val) > EVAL_SAMPLE_ROWS:
        rng = np.random.default_rng(RANDOM_SEED)
        eval_idx = rng.choice(len(X_val), size=EVAL_SAMPLE_ROWS, replace=False)
        X_val_eval = X_val.iloc[eval_idx]
        y_val_eval = y_val[eval_idx]

    print(f"\n--- TRAINING {MODEL_NAME.upper()} ({label}) ---")
    if len(X_val_eval) != len(X_val):
        print(f"Using {len(X_val_eval):,} validation rows for early stopping out of {len(X_val):,}.")

    #Build the model
    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=RANDOM_SEED,
        verbose=25,
        use_best_model=True,
        early_stopping_rounds=50,
        allow_writing_files=False,
        thread_count=-1,
        **CATBOOST_PARAMS,
    )
    #Fit with early stopping
    model.fit(
        X_train,
        y_train,
        eval_set=(X_val_eval, y_val_eval),
        cat_features=categorical_cols,
    )
    print("--- DONE ---")
    print("Best iteration:", model.get_best_iteration())

    #Predictions on original scale
    val_pred = inverse_target(model.predict(X_val))
    test_pred = inverse_target(model.predict(X_test))
    #Feature importance table
    importance_df = pd.DataFrame(
        {
            "feature": feature_list,
        "importance": model.get_feature_importance(),
        }
    ).sort_values("importance", ascending=False)

    #Return everything we need
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


#Full-data forecast refit
#Refit on all observed history
def fit_forecast_model(feature_list, observed_frame, best_iteration):
    #Reuse categorical columns
    categorical_cols = [col for col in ["month", "region_lat_bin", "region_lon_bin"] if col in feature_list]
    #Use all observed history
    X_full = prepare_model_frame(observed_frame, feature_list, categorical_cols)
    y_full = transform_target(observed_frame[TARGET].values)

    #Freeze tree count
    forecast_params = dict(CATBOOST_PARAMS)
    forecast_params["iterations"] = max(1, int(best_iteration) + 1)

    print(f"\n--- RETRAINING {MODEL_NAME.upper()} FOR FORECAST ---")
    print(f"Using {forecast_params['iterations']} trees on all observed history.")

    #Final forecast model
    forecast_model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=RANDOM_SEED,
        verbose=25,
        allow_writing_files=False,
        thread_count=-1,
        **forecast_params,
    )
    forecast_model.fit(
        X_full,
        y_full,
        cat_features=categorical_cols,
    )
    return forecast_model


#Recursive future forecast
#Forecast one month at a time
def run_recursive_forecast(forecast_model, history_frame, feature_list):
    print("\n--- FORECASTING 2023 ---")

    #Observed history
    history = history_frame[["lat", "lon", "date", TARGET]].copy()
    history["date"] = pd.to_datetime(history["date"])
    history = history.sort_values(["lat", "lon", "date"]).reset_index(drop=True)
    future_preds = []
    latest_date = history["date"].max()

    #Reuse the same grid cells
    cell_template = history.loc[history["date"] == latest_date, ["lat", "lon"]].copy()

    for step in range(FORECAST_MONTHS):
        #Next month
        next_date = latest_date + pd.DateOffset(months=1)

        #Placeholder rows
        placeholder = cell_template.copy()
        placeholder["date"] = next_date
        placeholder[TARGET] = np.nan

        #Rebuild causal features
        temp = pd.concat([history, placeholder], ignore_index=True)
        temp = temp.sort_values(["lat", "lon", "date"]).reset_index(drop=True)
        temp = add_history_features(temp, target=TARGET)
        temp = add_train_only_climatology(temp, train_end=TRAIN_END, target=TARGET)
        temp = add_experimental_features(temp)

        #Keep same-month ERA off
        temp, future_era5_feature_names = add_era5_features(
            temp,
            raw_dir=RAW_DIR,
            train_end=TRAIN_END,
            use_era5=False,
            feature_level=ERA5_FEATURE_LEVEL,
        )

        #Future feature rows
        future_feature_sets = build_feature_sets(future_era5_feature_names)
        future_features = future_feature_sets[FEATURE_SET]
        future_rows = temp.loc[temp["date"] == next_date].copy()
        future_rows = future_rows.dropna(subset=future_features)

        #Stop if no rows are scoreable
        if future_rows.empty:
            raise SystemExit(
                "Recursive forecasting could not build valid future rows. "
                "Check that the selected feature set can be computed causally."
            )

        #Score the next month
        X_future = prepare_model_frame(future_rows, future_features, [])
        preds = inverse_target(forecast_model.predict(X_future)).astype(np.float32)
        future_rows[TARGET] = preds

        #Append predictions to history
        next_month_output = future_rows[["lat", "lon", "date", TARGET]].copy()
        history = pd.concat([history, next_month_output], ignore_index=True)
        future_preds.append(next_month_output)
        latest_date = next_date

        print(f"Forecasted month {step + 1}/{FORECAST_MONTHS}: {next_date:%Y-%m}")

    #Save the 12-month forecast
    final_df = pd.concat(future_preds, ignore_index=True)
    final_df.to_csv(PRED_OUTPUT_FILE, index=False)
    print("Saved predictions to:", PRED_OUTPUT_FILE)


#Main summary plot
#Save the main summary figure
def save_main_plot(result):
    if not SAVE_PLOT:
        print("Skipped plot generation. Set CATBOOST_SAVE_PLOT=1 to save plots.")
        return

    #Create a 3-panel figure
    fig = plt.figure(figsize=(18, 5))

    #Feature chart
    #First panel
    plt.subplot(1, 3, 1)
    #Top 15 features
    top_imp = result["importance_df"].head(15).sort_values("importance")
    #Plot importance bars
    plt.barh(top_imp["feature"], top_imp["importance"])
    #Panel title
    plt.title("Top 15 Feature Importance")

    #Predicted vs actual
    #Second panel
    plt.subplot(1, 3, 2)
    #Plot sample size
    plot_sample = min(100_000, len(result["y_test"]))
    #Sample test rows
    sample_idx = np.random.default_rng(RANDOM_SEED).choice(
        len(result["y_test"]),
        size=plot_sample,
        replace=False,
    )
    #Scatter actual vs predicted
    plt.scatter(result["y_test"][sample_idx], result["test_pred"][sample_idx], alpha=0.25, s=5)
    #Axis limits
    lims = [
        min(float(result["y_test"].min()), float(np.min(result["test_pred"]))),
        max(float(result["y_test"].max()), float(np.max(result["test_pred"]))),
    ]
    #Perfect-fit line
    plt.plot(lims, lims, "r--", linewidth=1)
    #X label
    plt.xlabel("Actual PM2.5")
    #Y label
    plt.ylabel("Predicted PM2.5")
    #Panel title
    plt.title("Predicted vs Actual (Test)")

    #Residual histogram
    #Third panel
    plt.subplot(1, 3, 3)
    #Residuals
    residuals = result["y_test"] - result["test_pred"]
    #Residual histogram
    plt.hist(residuals, bins=60)
    #Zero line
    plt.axvline(0, linewidth=1)
    #X label
    plt.xlabel("Residual (Actual - Predicted)")
    #Panel title
    plt.title("Residual Distribution")

    #Tight layout
    plt.tight_layout()
    #Save figure
    plt.savefig(PLOT_OUTPUT_FILE, dpi=150)
    #Close figure to save memory
    plt.close()

    #Print save path
    print("Saved plot to:", PLOT_OUTPUT_FILE)


#Before/after ERA5 plot
#Save the ERA comparison figure
def save_comparison_plot(baseline_result, enhanced_result):
    if not SAVE_COMPARISON_PLOT:
        print("Skipped comparison plot generation. Set CATBOOST_SAVE_COMPARISON_PLOT=1 to save it.")
        return

    #Create a 2x2 figure
    fig = plt.figure(figsize=(12, 10))

    #Create R^2 comparison plot
    plt.subplot(2, 2, 1)
    #Shared labels
    labels = ["Validation", "Test"]
    #Bar positions
    x = np.arange(len(labels))
    #Set bar width
    width = 0.35
    #Baseline R^2
    baseline_r2 = [baseline_result["val_metrics"]["R2"], baseline_result["test_metrics"]["R2"]]
    #ERA R^2
    enhanced_r2 = [enhanced_result["val_metrics"]["R2"], enhanced_result["test_metrics"]["R2"]]
    #Baseline bars
    plt.bar(x - width / 2, baseline_r2, width=width, label="Without ERA5")
    #ERA bars
    plt.bar(x + width / 2, enhanced_r2, width=width, label="With ERA5")
    #X labels
    plt.xticks(x, labels)
    #Y label
    plt.ylabel("R^2")
    #Panel title
    plt.title("CatBoost R^2 Before vs After ERA5")
    #Legend
    plt.legend()

    #RMSE comparison bars
    #Second panel
    plt.subplot(2, 2, 2)
    #Baseline RMSE
    baseline_rmse = [baseline_result["val_metrics"]["RMSE"], baseline_result["test_metrics"]["RMSE"]]
    #ERA RMSE
    enhanced_rmse = [enhanced_result["val_metrics"]["RMSE"], enhanced_result["test_metrics"]["RMSE"]]
    #Baseline bars
    plt.bar(x - width / 2, baseline_rmse, width=width, label="Without ERA5")
    #ERA bars
    plt.bar(x + width / 2, enhanced_rmse, width=width, label="With ERA5")
    #X labels
    plt.xticks(x, labels)
    #Y label
    plt.ylabel("RMSE")
    #Panel title
    plt.title("CatBoost RMSE Before vs After ERA5")
    #Legend
    plt.legend()

    #Predicted vs actual without ERA5
    #Third panel
    plt.subplot(2, 2, 3)
    #Scatter sample size
    plot_sample = min(75_000, len(baseline_result["y_test"]))
    #Shared sampled rows
    sample_idx = np.random.default_rng(RANDOM_SEED).choice(
        len(baseline_result["y_test"]),
        size=plot_sample,
        replace=False,
    )
    #Scatter baseline predictions
    plt.scatter(
        baseline_result["y_test"][sample_idx],
        baseline_result["test_pred"][sample_idx],
        alpha=0.2,
        s=5,
    )
    #Axis limits
    lims = [
        min(float(baseline_result["y_test"].min()), float(np.min(baseline_result["test_pred"]))),
        max(float(baseline_result["y_test"].max()), float(np.max(baseline_result["test_pred"]))),
    ]
    #Perfect-fit line
    plt.plot(lims, lims, "r--", linewidth=1)
    #X label
    plt.xlabel("Actual PM2.5")
    #Y label
    plt.ylabel("Predicted PM2.5")
    #Panel title
    plt.title("Without ERA5")

    #Predicted vs actual with ERA5
    #Fourth panel
    plt.subplot(2, 2, 4)
    #Scatter ERA predictions
    plt.scatter(
        enhanced_result["y_test"][sample_idx],
        enhanced_result["test_pred"][sample_idx],
        alpha=0.2,
        s=5,
    )
    #Axis limits
    lims = [
        min(float(enhanced_result["y_test"].min()), float(np.min(enhanced_result["test_pred"]))),
        max(float(enhanced_result["y_test"].max()), float(np.max(enhanced_result["test_pred"]))),
    ]
    #Perfect-fit line
    plt.plot(lims, lims, "r--", linewidth=1)
    #X label
    plt.xlabel("Actual PM2.5")
    #Y label
    plt.ylabel("Predicted PM2.5")
    #Panel title
    plt.title("With ERA5")

    #Tight layout
    plt.tight_layout()
    #Save figure
    plt.savefig(COMPARISON_PLOT_OUTPUT_FILE, dpi=150)
    #Close figure to save memory
    plt.close()

    #Print save path
    print("Saved ERA5 comparison plot to:", COMPARISON_PLOT_OUTPUT_FILE)


#Load/features
#Build the modeling table
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

feature_sets_no_era5 = build_feature_sets([])
feature_sets_with_era5 = build_feature_sets(era5_feature_names)
if FEATURE_SET not in feature_sets_with_era5:
    raise ValueError(
        f"Unknown CATBOOST_FEATURE_SET={FEATURE_SET!r}. "
        f"Choose from: {', '.join(feature_sets_with_era5)}"
    )

baseline_features = feature_sets_no_era5[FEATURE_SET]
enhanced_features = feature_sets_with_era5[FEATURE_SET]
active_features = enhanced_features if era5_feature_names else baseline_features

#Trim to needed columns
#Drop unused columns
required_features = list(active_features)
if COMPARE_ERA5 and era5_feature_names:
    required_features = sorted(set(required_features).union(baseline_features))
required_columns = ["date", TARGET] + required_features
df = df.loc[:, required_columns].copy()

#Split
#Split by date
train = df.loc[df["date"] < TRAIN_END]
val = df.loc[(df["date"] >= TRAIN_END) & (df["date"] < VAL_END)]
test = df.loc[df["date"] >= VAL_END]
observed_history = pd.concat([train, val, test], ignore_index=True)
del df

print(f"\nFeature set: {FEATURE_SET}")
print(f"Feature count: {len(active_features)}")
print(f"Target transform: {TARGET_TRANSFORM}")
print(f"ERA5 feature level: {ERA5_FEATURE_LEVEL}")
print(f"\nTrain: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

#Optional baseline
#Run baseline first for comparisons
baseline_result = None
if COMPARE_ERA5 and era5_feature_names:
    baseline_result = fit_scenario("Without ERA5", baseline_features, train, val, test)

enhanced_label = "With ERA5" if era5_feature_names else "Without ERA5"
enhanced_result = fit_scenario(enhanced_label, active_features, train, val, test)

#Naive lag-1 baseline
naive_pred = enhanced_result["X_test"]["pm25_lag1"].astype(float).values
naive_metrics = compute_metrics(enhanced_result["y_test"], naive_pred)

#Print final metrics
print("\n--- FINAL RESULTS ---")
print_metrics(f"{MODEL_NAME} Validation Metrics", enhanced_result["val_metrics"])
print_metrics(f"{MODEL_NAME} Test Metrics", enhanced_result["test_metrics"])
print_metrics("Naive Test Metrics", naive_metrics)

#Metrics table
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
#Save importance table
enhanced_result["importance_df"].to_csv(IMPORTANCE_OUTPUT_FILE, index=False)
print("Saved feature importance to:", IMPORTANCE_OUTPUT_FILE)
print("\nTop 15 Features:")
print(enhanced_result["importance_df"].head(15).to_string(index=False))

#Save ERA comparison only when both scenarios were run
if baseline_result is not None:
    #Comparison table
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
    print("Saved ERA5 comparison metrics to:", COMPARISON_METRICS_OUTPUT_FILE)
    #Save comparison figure
    save_comparison_plot(baseline_result, enhanced_result)
else:
    #Skip comparison when only one scenario was trained
    print("Skipped ERA5 comparison outputs because the baseline scenario was not run.")

#Optional main figure
save_main_plot(enhanced_result)

#Forecast 2023
#Only forecast on the baseline path
#Run recursive forecast only for one no-ERA scenario
if RUN_FORECAST and not USE_ERA5 and not COMPARE_ERA5:
    #Refit on all observed history
    forecast_model = fit_forecast_model(
        active_features,
        observed_history,
        enhanced_result["model"].get_best_iteration(),
    )
    #Generate monthly recursive predictions
    run_recursive_forecast(forecast_model, observed_history, active_features)
#Explain why forecast was skipped in other modes
elif RUN_FORECAST:
    print(
        "Skipped recursive forecast because the current CatBoost run is using "
        "ERA5 or comparison mode. The baseline no-ERA run remains the strict "
        "forecasting path."
    )
