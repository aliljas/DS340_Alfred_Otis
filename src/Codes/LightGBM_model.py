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
#Use project-relative paths
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data/processed"
RAW_DIR = BASE_DIR / "data/raw"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_FILE = PROCESSED_DIR / "na_pm25_cells_clean.csv"

TRAIN_END = pd.Timestamp("2021-01-01")
VAL_END = pd.Timestamp("2022-01-01")
RANDOM_SEED = 42

FEATURE_SET = os.getenv("LIGHTGBM_FEATURE_SET", "trend_region_era5plus")
SAVE_PLOT = os.getenv("LIGHTGBM_SAVE_PLOT", "0") == "1"
SAVE_COMPARISON_PLOT = os.getenv("LIGHTGBM_SAVE_COMPARISON_PLOT", "1") == "1"
COMPARE_ERA5 = os.getenv("LIGHTGBM_COMPARE_ERA5", "1") == "1"
MODEL_NAME = "LightGBM"
TARGET_TRANSFORM = os.getenv("LIGHTGBM_TARGET_TRANSFORM", "log1p")
CATEGORICAL_FEATURES = ["month", "region_lat_bin", "region_lon_bin"]
USE_ERA5 = os.getenv("LIGHTGBM_USE_ERA5", "1") == "1"
ERA5_FEATURE_LEVEL = os.getenv("LIGHTGBM_ERA5_FEATURE_LEVEL", "extended")
RUN_FORECAST = os.getenv("LIGHTGBM_RUN_FORECAST", "1") == "1"

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

PRED_OUTPUT_FILE = PROCESSED_DIR / "lightgbm_predictions_2023.csv"
METRICS_OUTPUT_FILE = PROCESSED_DIR / "lightgbm_eval_metrics.csv"
IMPORTANCE_OUTPUT_FILE = PROCESSED_DIR / "lightgbm_feature_importance.csv"
PLOT_OUTPUT_FILE = PROCESSED_DIR / "lightgbm_model_results.png"
COMPARISON_METRICS_OUTPUT_FILE = PROCESSED_DIR / "lightgbm_era5_comparison_metrics.csv"
COMPARISON_PLOT_OUTPUT_FILE = PROCESSED_DIR / "lightgbm_era5_comparison.png"


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
    raise ValueError("LIGHTGBM_TARGET_TRANSFORM must be 'log1p' or 'none'")


#Target inverse transform
#Return to PM2.5 scale
def inverse_target(values):
    if TARGET_TRANSFORM == "log1p":
        return np.expm1(values)
    return values


#Category casting
#Mark categorical columns
def prepare_model_frame(frame, features):
    prepared = frame[features].copy()
    for col in CATEGORICAL_FEATURES:
        if col in prepared.columns:
            prepared[col] = prepared[col].astype("category")
    return prepared


#Fit one scenario
#Fit one LightGBM run
def fit_scenario(label, features, train, val, test):
    #Feature matrices
    X_train = prepare_model_frame(train, features)
    X_val = prepare_model_frame(val, features)
    X_test = prepare_model_frame(test, features)

    #Targets
    y_train = transform_target(train[TARGET].values)
    y_val = transform_target(val[TARGET].values)
    y_val_raw = val[TARGET].values
    y_test = test[TARGET].values

    print(f"\n--- TRAINING {MODEL_NAME.upper()} ({label}) ---")
    #Build model
    model = LGBMRegressor(**LIGHTGBM_PARAMS)
    #Fit with early stopping
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

    #Predictions on original scale
    val_pred = inverse_target(model.predict(X_val, num_iteration=model.best_iteration_))
    test_pred = inverse_target(model.predict(X_test, num_iteration=model.best_iteration_))

    #Return everything we need
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
#Save the main summary figure
def save_main_plot(result):
    if not SAVE_PLOT:
        print("Skipped plot generation. Set LIGHTGBM_SAVE_PLOT=1 to save plots.")
        return

    #Unpack result pieces
    y_test = result["y_test"]
    test_pred = result["test_pred"]
    importance_df = result["importance_df"]

    #Create a 3-panel figure
    fig = plt.figure(figsize=(18, 5))

    #First panel
    plt.subplot(1, 3, 1)
    #Top 15 features
    top_imp = importance_df.head(15).sort_values("importance")
    #Plot importance bars
    plt.barh(top_imp["feature"], top_imp["importance"])
    #Panel title
    plt.title("Top 15 Feature Importance")

    #Second panel
    plt.subplot(1, 3, 2)
    #Plot sample size
    plot_sample = min(100_000, len(y_test))
    #Sample test rows
    sample_idx = np.random.default_rng(RANDOM_SEED).choice(len(y_test), size=plot_sample, replace=False)
    #Scatter actual vs predicted
    plt.scatter(y_test[sample_idx], test_pred[sample_idx], alpha=0.25, s=5)
    #Axis limits
    lims = [
        min(float(y_test.min()), float(np.min(test_pred))),
        max(float(y_test.max()), float(np.max(test_pred))),
    ]
    #Perfect-fit line
    plt.plot(lims, lims, "r--", linewidth=1)
    #X label
    plt.xlabel("Actual PM2.5")
    #Y label
    plt.ylabel("Predicted PM2.5")
    #Panel title
    plt.title("Predicted vs Actual (Test)")

    #Third panel
    plt.subplot(1, 3, 3)
    #Residuals
    residuals = y_test - test_pred
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
        print("Skipped comparison plot generation. Set LIGHTGBM_SAVE_COMPARISON_PLOT=1 to save it.")
        return

    #Metric blocks
    baseline_metrics = baseline_result["test_metrics"]
    enhanced_metrics = enhanced_result["test_metrics"]
    baseline_val_metrics = baseline_result["val_metrics"]
    enhanced_val_metrics = enhanced_result["val_metrics"]

    #Create a 2x2 figure
    fig = plt.figure(figsize=(16, 10))

    #First panel
    plt.subplot(2, 2, 1)
    #Shared labels
    metric_names = ["Validation", "Test"]
    #Baseline R^2
    baseline_r2 = [baseline_val_metrics["R2"], baseline_metrics["R2"]]
    #ERA R^2
    enhanced_r2 = [enhanced_val_metrics["R2"], enhanced_metrics["R2"]]
    #Bar positions
    x = np.arange(len(metric_names))
    #Bar width
    width = 0.35
    #Baseline bars
    plt.bar(x - width / 2, baseline_r2, width=width, label="Without ERA5")
    #ERA bars
    plt.bar(x + width / 2, enhanced_r2, width=width, label="With ERA5")
    #X labels
    plt.xticks(x, metric_names)
    #Y label
    plt.ylabel("R^2")
    #Panel title
    plt.title("LightGBM R^2 Before vs After ERA5")
    #Legend
    plt.legend()

    #Second panel
    plt.subplot(2, 2, 2)
    #Baseline RMSE
    baseline_rmse = [baseline_val_metrics["RMSE"], baseline_metrics["RMSE"]]
    #ERA RMSE
    enhanced_rmse = [enhanced_val_metrics["RMSE"], enhanced_metrics["RMSE"]]
    #Baseline bars
    plt.bar(x - width / 2, baseline_rmse, width=width, label="Without ERA5")
    #ERA bars
    plt.bar(x + width / 2, enhanced_rmse, width=width, label="With ERA5")
    #X labels
    plt.xticks(x, metric_names)
    #Y label
    plt.ylabel("RMSE")
    #Panel title
    plt.title("LightGBM RMSE Before vs After ERA5")
    #Legend
    plt.legend()

    #Shared scatter sample
    plot_sample = min(100_000, len(enhanced_result["y_test"]))
    #Sample test rows
    sample_idx = np.random.default_rng(RANDOM_SEED).choice(
        len(enhanced_result["y_test"]),
        size=plot_sample,
        replace=False,
    )

    #Third panel
    plt.subplot(2, 2, 3)
    #Scatter baseline predictions
    plt.scatter(
        baseline_result["y_test"][sample_idx],
        baseline_result["test_pred"][sample_idx],
        alpha=0.2,
        s=5,
    )
    #Axis limits
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
    #Perfect-fit line
    plt.plot(lims, lims, "r--", linewidth=1)
    #X label
    plt.xlabel("Actual PM2.5")
    #Y label
    plt.ylabel("Predicted PM2.5")
    #Panel title
    plt.title("Without ERA5")

    #Fourth panel
    plt.subplot(2, 2, 4)
    #Scatter ERA predictions
    plt.scatter(
        enhanced_result["y_test"][sample_idx],
        enhanced_result["test_pred"][sample_idx],
        alpha=0.2,
        s=5,
    )
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

#Split
#Split by date
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
print(f"ERA5 feature level: {ERA5_FEATURE_LEVEL}")
print(f"\nTrain: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

#Optional baseline
#Run baseline first for comparisons
baseline_result = None
if COMPARE_ERA5 and era5_feature_names:
    baseline_result = fit_scenario("Without ERA5", baseline_features, train, val, test)

enhanced_label = "With ERA5" if era5_feature_names else "No ERA5 Available"
enhanced_result = fit_scenario(enhanced_label, active_features, train, val, test)

#Naive lag-1 baseline
naive_pred = enhanced_result["X_test"]["pm25_lag1"].values
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
enhanced_result["importance_df"].to_csv(IMPORTANCE_OUTPUT_FILE, index=False)
print("Saved feature importance to:", IMPORTANCE_OUTPUT_FILE)
print("\nTop 15 Features:")
print(enhanced_result["importance_df"].head(15).to_string(index=False))

#Save ERA comparison only when both scenarios were run
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
    #Skip comparison when only one scenario was trained
    print("Skipped ERA5 comparison outputs because the baseline scenario was not run.")

#Optional main figure
save_main_plot(enhanced_result)

#Store training medians for forecast fill
train_feature_fill_values = train[active_features].median(numeric_only=True)

#Fill any all-missing medians
train_feature_fill_values = train_feature_fill_values.fillna(0.0)

#Run baseline forecast only
if RUN_FORECAST and not USE_ERA5 and not COMPARE_ERA5:
    #Start forecasting
    print("\n--- FORECASTING 2023 ---")

    #Build full history
    history = df[["lat", "lon", "date", TARGET]].copy()

    #Store forecast outputs
    future_preds = []

    #Find last known date
    latest_date = history["date"].max()

    #Forecast 12 months
    for step in range(12):
        #Advance one month
        next_date = latest_date + pd.DateOffset(months=1)

        #Copy latest grid
        base = history[history["date"] == latest_date][["lat", "lon"]].copy()

        #Stop if latest grid is empty
        if base.empty:
            raise ValueError(
                f"No grid cells found for latest_date={latest_date:%Y-%m-%d}. "
                "History does not contain a usable latest month."
            )

        #Set future date
        base["date"] = next_date

        #Append empty future target
        temp = pd.concat([history, base.assign(pm25=np.nan)], ignore_index=True)

        #Sort for lag features
        temp = temp.sort_values(["lat", "lon", "date"]).reset_index(drop=True)

        #Add history features
        temp = add_history_features(temp, target=TARGET)

        #Add climatology features
        temp = add_train_only_climatology(temp, train_end=TRAIN_END, target=TARGET)

        #Add engineered features
        temp = add_experimental_features(temp)

        #Add forecast-safe ERA5 columns
        temp, future_era5_feature_names = add_era5_features(
            temp,
            raw_dir=RAW_DIR,
            train_end=TRAIN_END,
            use_era5=False,
        )

        #Build forecast feature sets
        future_feature_sets = build_feature_sets(future_era5_feature_names)

        #Validate forecast feature set
        if FEATURE_SET not in future_feature_sets:
            raise ValueError(f"Unknown forecast feature set: {FEATURE_SET}")

        #Keep next-month rows
        future_rows = temp[temp["date"] == next_date].copy()

        #Fallback if exact date slice is empty
        if future_rows.empty:
            print(f"\nForecast step {step + 1}/12: {next_date.strftime('%Y-%m-%d')}")
            print("No rows found for next month after feature building.")
            print("Falling back to latest grid with filled features.")
            future_rows = base.copy()

            #Rebuild temp using fallback rows
            temp = pd.concat([history, future_rows.assign(pm25=np.nan)], ignore_index=True)
            temp = temp.sort_values(["lat", "lon", "date"]).reset_index(drop=True)
            temp = add_history_features(temp, target=TARGET)
            temp = add_train_only_climatology(temp, train_end=TRAIN_END, target=TARGET)
            temp = add_experimental_features(temp)
            temp, future_era5_feature_names = add_era5_features(
                temp,
                raw_dir=RAW_DIR,
                train_end=TRAIN_END,
                use_era5=False,
            )
            future_rows = temp[temp["date"] == next_date].copy()

        #Stop if still empty
        if future_rows.empty:
            raise ValueError(
                f"Forecast rows are empty for {next_date:%Y-%m-%d} even after fallback."
            )

        #Check for missing columns
        missing_feature_columns = [col for col in active_features if col not in future_rows.columns]

        #Stop on missing columns
        if missing_feature_columns:
            raise ValueError(
                "Forecast feature mismatch. Missing columns: "
                + ", ".join(missing_feature_columns)
            )

        #Print step header
        print(f"\nForecast step {step + 1}/12: {next_date.strftime('%Y-%m-%d')}")

        #Count missing values
        missing_counts = future_rows[active_features].isna().sum()
        missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

        #Print missing values
        if not missing_counts.empty:
            print("Missing values before fill:")
            print(missing_counts.to_string())

        #Build future matrix
        X_future = future_rows.loc[:, active_features].copy()

        #Fill from training medians
        X_future = X_future.fillna(train_feature_fill_values)

        #Fill remaining values
        X_future = X_future.fillna(0.0)

        #Stop if empty
        if X_future.empty or len(X_future) == 0:
            raise ValueError(
                f"Forecast matrix is empty for {next_date:%Y-%m-%d} after feature filling."
            )

        #Stop if NaNs remain
        if X_future.isna().any().any():
            remaining_missing = X_future.isna().sum()
            remaining_missing = remaining_missing[remaining_missing > 0]
            raise ValueError(
                "Forecast matrix still contains NaNs after fill:\n"
                + remaining_missing.to_string()
            )

        #Prepare future matrix
        X_future = prepare_model_frame(X_future, active_features)

        #Stop if prepared matrix is empty
        if X_future.empty or len(X_future) == 0:
            raise ValueError(
                f"Prepared forecast matrix is empty for {next_date:%Y-%m-%d}."
            )

        #Predict future PM2.5
        preds = inverse_target(
            enhanced_result["model"].predict(
                X_future,
                num_iteration=enhanced_result["model"].best_iteration_,
            )
        )

        #Clip predictions
        preds = np.clip(
            preds,
            float(train[TARGET].min()),
            float(train[TARGET].max()),
        )

        #Write predictions
        future_rows[TARGET] = preds

        #Append to history
        history = pd.concat(
            [history, future_rows[["lat", "lon", "date", TARGET]]],
            ignore_index=True,
        )

        #Save monthly forecast
        future_preds.append(future_rows[["lat", "lon", "date", TARGET]].copy())

        #Advance pointer
        latest_date = next_date

    #Combine forecasts
    final_df = pd.concat(future_preds, ignore_index=True)

    #Save predictions
    final_df.to_csv(PRED_OUTPUT_FILE, index=False)

    #Print save path
    print("\nSaved predictions to:", PRED_OUTPUT_FILE)

#Skip forecast in ERA or comparison mode
elif RUN_FORECAST:
    print(
        "\nSkipped forecasting because this run used "
        "ERA5 or comparison mode. The baseline no-ERA run remains the strict "
        "forecast-ready LightGBM path."
    )
