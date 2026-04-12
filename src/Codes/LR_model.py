#Import OS helpers
import os

#Import temp helpers
import tempfile

#Import path tools
from pathlib import Path

#Set matplotlib cache path
MPLCONFIGDIR = Path(tempfile.gettempdir()) / "matplotlib-cache"

#Create cache folder
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

#Set matplotlib cache env var
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

#Import required libraries
try:
    import matplotlib

    #Use non-GUI backend
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Ridge
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        median_absolute_error,
        r2_score,
    )
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from model_feature_utils import (
        TARGET,
        add_era5_features,
        add_experimental_features,
        add_history_features,
        add_train_only_climatology,
        prepare_modeling_frame,
    )

#Stop if imports fail
except ImportError as exc:
    #Raise clear error
    raise SystemExit(
        "Required packages for LR_model.py are not installed. "
        "Install the project requirements first, then rerun this script."
    ) from exc


#Set base directory
BASE_DIR = Path(__file__).resolve().parents[2]

#Set processed directory
PROCESSED_DIR = BASE_DIR / "data/processed"

#Set raw directory
RAW_DIR = BASE_DIR / "data/raw"

#Make processed directory
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

#Make raw directory
RAW_DIR.mkdir(parents=True, exist_ok=True)

#Set cleaned data file
DATA_FILE = PROCESSED_DIR / "na_pm25_cells_clean.csv"

#Set train end date
TRAIN_END = pd.Timestamp("2021-01-01")

#Set validation end date
VAL_END = pd.Timestamp("2022-01-01")

#Set random seed
RANDOM_SEED = 42


#Set feature set
FEATURE_SET = os.getenv("LR_FEATURE_SET", "linear_core")

#Set target transform
TARGET_TRANSFORM = os.getenv("LR_TARGET_TRANSFORM", "none")

#Set main plot
SAVE_PLOT = os.getenv("LR_SAVE_PLOT", "0") == "1"

#Set comparison plot
SAVE_COMPARISON_PLOT = os.getenv("LR_SAVE_COMPARISON_PLOT", "1") == "1"

#Set ERA5 comparison
COMPARE_ERA5 = os.getenv("LR_COMPARE_ERA5", "0") == "1"

#Set ERA5 usage
USE_ERA5 = os.getenv("LR_USE_ERA5", "0") == "1"

#Set ERA5 feature level
ERA5_FEATURE_LEVEL = os.getenv("LR_ERA5_FEATURE_LEVEL", "core")

#Set forecast usage
RUN_FORECAST = os.getenv("LR_RUN_FORECAST", "1") == "1"

#Set model name
MODEL_NAME = "RidgeRegression"


#Set prediction output file
PRED_OUTPUT_FILE = PROCESSED_DIR / "lr_predictions_2023.csv"

#Set metrics output file
METRICS_OUTPUT_FILE = PROCESSED_DIR / "lr_eval_metrics.csv"

#Set coefficients output file
COEF_OUTPUT_FILE = PROCESSED_DIR / "lr_coefficients.csv"

#Set plot output file
PLOT_OUTPUT_FILE = PROCESSED_DIR / "lr_model_results.png"

#Set comparison metrics output file
COMPARISON_METRICS_OUTPUT_FILE = PROCESSED_DIR / "lr_era5_comparison_metrics.csv"

#Set comparison plot output file
COMPARISON_PLOT_OUTPUT_FILE = PROCESSED_DIR / "lr_era5_comparison.png"


#Compute regression metrics
def compute_metrics(y_true, y_pred):
    #Return metric dictionary
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MedianAE": float(median_absolute_error(y_true, y_pred)),
        "Bias": float(np.mean(y_pred - y_true)),
    }


#Print metrics nicely
def print_metrics(name, metrics):
    #Print section name
    print(f"\n{name}")

    #Print RMSE
    print(f"RMSE:      {metrics['RMSE']:.6f}")

    #Print MAE
    print(f"MAE:       {metrics['MAE']:.6f}")

    #Print R-squared
    print(f"R^2:       {metrics['R2']:.6f}")

    #Print median AE
    print(f"Median AE: {metrics['MedianAE']:.6f}")

    #Print bias
    print(f"Bias:      {metrics['Bias']:.6f}")


#Transform target
def transform_target(values):
    #Use log1p transform
    if TARGET_TRANSFORM == "log1p":
        return np.log1p(values)

    #Use raw target
    if TARGET_TRANSFORM == "none":
        return values

    #Reject bad option
    raise ValueError("LR_TARGET_TRANSFORM must be 'log1p' or 'none'")


#Inverse target transform
def inverse_target(values):
    #Undo log1p
    if TARGET_TRANSFORM == "log1p":
        return np.expm1(values)

    #Return unchanged
    return values


#Find best blend weight
def find_blend_weight(y_true, model_pred, naive_pred):
    #Compute difference
    diff = model_pred - naive_pred

    #Compute denominator
    denom = float(np.dot(diff, diff))

    #Handle zero denominator
    if denom <= 0:
        return 0.0

    #Compute numerator
    numer = float(np.dot(diff, y_true - naive_pred))

    #Clamp blend weight
    return float(np.clip(numer / denom, 0.0, 1.0))


#Build LR feature sets
def build_linear_feature_sets(era5_feature_names=None):
    #Normalize ERA5 feature list
    era5_feature_names = list(era5_feature_names or [])

    #Define base features
    base_features = [
        "lat",
        "lon",
        "month_sin",
        "month_cos",
        "doy_sin",
        "doy_cos",
        "time_index",
        "pm25_lag1",
        "pm25_lag12",
        "pm25_lag24",
        "pm25_deviation_30",
        "pm25_yoy_change",
        "cell_month_mean_train",
        "cell_month_std_train",
        "cell_month_median_train",
        "region_month_std_train",
        "cell_vs_region_month",
        "pm25_region_zscore",
    ]

    #Return feature set mapping
    return {
        "linear_core": base_features + era5_feature_names,
    }


#Train one scenario
def fit_scenario(label, feature_list, train_frame, val_frame, test_frame):
    #Build train features
    X_train = train_frame[feature_list]

    #Build validation features
    X_val = val_frame[feature_list]

    #Build test features
    X_test = test_frame[feature_list]

    #Build train target
    y_train = transform_target(train_frame[TARGET].values)

    #Build validation target
    y_val_raw = val_frame[TARGET].values

    #Build test target
    y_test = test_frame[TARGET].values

    #Build naive validation baseline
    naive_val = val_frame["pm25_lag1"].astype(float).values

    #Build naive test baseline
    naive_test = test_frame["pm25_lag1"].astype(float).values

    #Get train minimum
    y_train_raw_min = float(np.min(train_frame[TARGET].values))

    #Get train maximum
    y_train_raw_max = float(np.max(train_frame[TARGET].values))

    #Print training header
    print(f"\n--- TRAINING {MODEL_NAME.upper()} ({label}) ---")

    #Build ridge pipeline
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=20.0, random_state=RANDOM_SEED),
    )

    #Fit model
    model.fit(X_train, y_train)

    #Print done message
    print("--- DONE ---")

    #Predict validation transformed
    val_pred_trans = model.predict(X_val)

    #Predict test transformed
    test_pred_trans = model.predict(X_test)

    #Invert validation predictions
    val_pred = inverse_target(val_pred_trans)

    #Invert test predictions
    test_pred = inverse_target(test_pred_trans)

    #Clip validation predictions
    val_pred = np.clip(val_pred, y_train_raw_min, y_train_raw_max)

    #Clip test predictions
    test_pred = np.clip(test_pred, y_train_raw_min, y_train_raw_max)

    #Tune blend on validation
    blend_weight = find_blend_weight(y_val_raw, val_pred, naive_val)

    #Blend validation predictions
    val_pred = blend_weight * val_pred + (1.0 - blend_weight) * naive_val

    #Blend test predictions
    test_pred = blend_weight * test_pred + (1.0 - blend_weight) * naive_test

    #Extract coefficients
    coef_values = model.named_steps["ridge"].coef_

    #Build coefficient table
    coef_df = pd.DataFrame(
        {
            "feature": feature_list,
            "coefficient": coef_values,
            "abs_coefficient": np.abs(coef_values),
        }
    ).sort_values("abs_coefficient", ascending=False)

    #Return scenario outputs
    return {
        "label": label,
        "features": feature_list,
        "model": model,
        "blend_weight": blend_weight,
        "X_test": X_test,
        "y_test": y_test,
        "test_pred": test_pred,
        "val_metrics": compute_metrics(y_val_raw, val_pred),
        "test_metrics": compute_metrics(y_test, test_pred),
        "coef_df": coef_df,
    }


#Save main plot
def save_main_plot(result):
    #Skip if disabled
    if not SAVE_PLOT:
        print("Skipped plot generation. Set LR_SAVE_PLOT=1 to save plots.")
        return

    #Create figure
    fig = plt.figure(figsize=(18, 5))

    #Go to first panel
    plt.subplot(1, 3, 1)

    #Select top coefficients
    top_coef = result["coef_df"].head(15).sort_values("abs_coefficient")

    #Draw bar chart
    plt.barh(top_coef["feature"], top_coef["abs_coefficient"])

    #Set panel title
    plt.title("Top 15 Absolute Coefficients")

    #Go to second panel
    plt.subplot(1, 3, 2)

    #Set sample size
    plot_sample = min(100_000, len(result["y_test"]))

    #Sample test rows
    sample_idx = np.random.default_rng(RANDOM_SEED).choice(
        len(result["y_test"]),
        size=plot_sample,
        replace=False,
    )

    #Draw scatter
    plt.scatter(
        result["y_test"][sample_idx],
        result["test_pred"][sample_idx],
        alpha=0.25,
        s=5,
    )

    #Compute axis limits
    lims = [
        min(float(result["y_test"].min()), float(np.min(result["test_pred"]))),
        max(float(result["y_test"].max()), float(np.max(result["test_pred"]))),
    ]

    #Draw perfect-fit line
    plt.plot(lims, lims, "r--", linewidth=1)

    #Set x label
    plt.xlabel("Actual PM2.5")

    #Set y label
    plt.ylabel("Predicted PM2.5")

    #Set panel title
    plt.title("Predicted vs Actual (Test)")

    #Go to third panel
    plt.subplot(1, 3, 3)

    #Compute residuals
    residuals = result["y_test"] - result["test_pred"]

    #Draw residual histogram
    plt.hist(residuals, bins=60)

    #Draw zero line
    plt.axvline(0, linewidth=1)

    #Set x label
    plt.xlabel("Residual (Actual - Predicted)")

    #Set panel title
    plt.title("Residual Distribution")

    #Tighten layout
    plt.tight_layout()

    #Save figure
    plt.savefig(PLOT_OUTPUT_FILE, dpi=150)

    #Close figure
    plt.close()

    #Print save path
    print("Saved plot to:", PLOT_OUTPUT_FILE)


#Save comparison plot
def save_comparison_plot(baseline_result, enhanced_result):
    #Skip if disabled
    if not SAVE_COMPARISON_PLOT:
        print("Skipped comparison plot generation. Set LR_SAVE_COMPARISON_PLOT=1 to save it.")
        return

    #Create figure
    fig = plt.figure(figsize=(12, 10))

    #Go to first panel
    plt.subplot(2, 2, 1)

    #Set labels
    labels = ["Validation", "Test"]

    #Set x positions
    x = np.arange(len(labels))

    #Set bar width
    width = 0.35

    #Collect baseline R2
    baseline_r2 = [baseline_result["val_metrics"]["R2"], baseline_result["test_metrics"]["R2"]]

    #Collect enhanced R2
    enhanced_r2 = [enhanced_result["val_metrics"]["R2"], enhanced_result["test_metrics"]["R2"]]

    #Draw baseline bars
    plt.bar(x - width / 2, baseline_r2, width=width, label="Without ERA5")

    #Draw enhanced bars
    plt.bar(x + width / 2, enhanced_r2, width=width, label="With ERA5")

    #Set x ticks
    plt.xticks(x, labels)

    #Set y label
    plt.ylabel("R^2")

    #Set title
    plt.title("Ridge Regression R^2 Before vs After ERA5")

    #Show legend
    plt.legend()

    #Go to second panel
    plt.subplot(2, 2, 2)

    #Collect baseline RMSE
    baseline_rmse = [baseline_result["val_metrics"]["RMSE"], baseline_result["test_metrics"]["RMSE"]]

    #Collect enhanced RMSE
    enhanced_rmse = [enhanced_result["val_metrics"]["RMSE"], enhanced_result["test_metrics"]["RMSE"]]

    #Draw baseline bars
    plt.bar(x - width / 2, baseline_rmse, width=width, label="Without ERA5")

    #Draw enhanced bars
    plt.bar(x + width / 2, enhanced_rmse, width=width, label="With ERA5")

    #Set x ticks
    plt.xticks(x, labels)

    #Set y label
    plt.ylabel("RMSE")

    #Set title
    plt.title("Ridge Regression RMSE Before vs After ERA5")

    #Show legend
    plt.legend()

    #Go to third panel
    plt.subplot(2, 2, 3)

    #Set scatter sample size
    plot_sample = min(75_000, len(baseline_result["y_test"]))

    #Sample common rows
    sample_idx = np.random.default_rng(RANDOM_SEED).choice(
        len(baseline_result["y_test"]),
        size=plot_sample,
        replace=False,
    )

    #Draw baseline scatter
    plt.scatter(
        baseline_result["y_test"][sample_idx],
        baseline_result["test_pred"][sample_idx],
        alpha=0.2,
        s=5,
    )

    #Compute limits
    lims = [
        min(float(baseline_result["y_test"].min()), float(np.min(baseline_result["test_pred"]))),
        max(float(baseline_result["y_test"].max()), float(np.max(baseline_result["test_pred"]))),
    ]

    #Draw perfect-fit line
    plt.plot(lims, lims, "r--", linewidth=1)

    #Set x label
    plt.xlabel("Actual PM2.5")

    #Set y label
    plt.ylabel("Predicted PM2.5")

    #Set title
    plt.title("Without ERA5")

    #Go to fourth panel
    plt.subplot(2, 2, 4)

    #Draw enhanced scatter
    plt.scatter(
        enhanced_result["y_test"][sample_idx],
        enhanced_result["test_pred"][sample_idx],
        alpha=0.2,
        s=5,
    )

    #Compute limits
    lims = [
        min(float(enhanced_result["y_test"].min()), float(np.min(enhanced_result["test_pred"]))),
        max(float(enhanced_result["y_test"].max()), float(np.max(enhanced_result["test_pred"]))),
    ]

    #Draw perfect-fit line
    plt.plot(lims, lims, "r--", linewidth=1)

    #Set x label
    plt.xlabel("Actual PM2.5")

    #Set y label
    plt.ylabel("Predicted PM2.5")

    #Set title
    plt.title("With ERA5")

    #Tighten layout
    plt.tight_layout()

    #Save figure
    plt.savefig(COMPARISON_PLOT_OUTPUT_FILE, dpi=150)

    #Close figure
    plt.close()

    #Print save path
    print("Saved ERA5 comparison plot to:", COMPARISON_PLOT_OUTPUT_FILE)


#Build modeling frame
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

#Build non-ERA5 feature sets
feature_sets_no_era5 = build_linear_feature_sets([])

#Build ERA5 feature sets
feature_sets_with_era5 = build_linear_feature_sets(era5_feature_names)

#Validate selected feature set
if FEATURE_SET not in feature_sets_with_era5:
    raise ValueError(
        f"Unknown LR_FEATURE_SET={FEATURE_SET!r}. "
        f"Choose from: {', '.join(feature_sets_with_era5)}"
    )

#Get baseline features
baseline_features = feature_sets_no_era5[FEATURE_SET]

#Get enhanced features
enhanced_features = feature_sets_with_era5[FEATURE_SET]

#Select active features
active_features = enhanced_features if era5_feature_names else baseline_features

#Start required feature list
required_features = list(active_features)

#Include baseline features for comparison
if COMPARE_ERA5 and era5_feature_names:
    required_features = sorted(set(required_features).union(baseline_features))

#Set required columns
required_columns = ["date", TARGET] + required_features

#Trim dataframe
df = df.loc[:, required_columns].copy()

#Split training data
train = df[df["date"] < TRAIN_END].copy()

#Split validation data
val = df[(df["date"] >= TRAIN_END) & (df["date"] < VAL_END)].copy()

#Split test data
test = df[df["date"] >= VAL_END].copy()

#Free memory
del df

#Build fill values from training data
train_feature_fill_values = train[active_features].median(numeric_only=True)

#Fill any remaining missing medians with zero
train_feature_fill_values = train_feature_fill_values.fillna(0.0)

#Print selected feature set
print(f"\nFeature set: {FEATURE_SET}")

#Print feature count
print(f"Feature count: {len(active_features)}")

#Print target transform
print(f"Target transform: {TARGET_TRANSFORM}")

#Print ERA5 level
print(f"ERA5 feature level: {ERA5_FEATURE_LEVEL}")

#Print split sizes
print(f"\nTrain: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

#Initialize baseline result
baseline_result = None

#Fit baseline if comparing ERA5
if COMPARE_ERA5 and era5_feature_names:
    baseline_result = fit_scenario("Without ERA5", baseline_features, train, val, test)

#Set enhanced label
enhanced_label = "With ERA5" if era5_feature_names else "Without ERA5"

#Fit active scenario
enhanced_result = fit_scenario(enhanced_label, active_features, train, val, test)

#Store fitted model
model = enhanced_result["model"]

#Build naive test baseline
naive_pred = enhanced_result["X_test"]["pm25_lag1"].astype(float).values

#Score naive baseline
naive_metrics = compute_metrics(enhanced_result["y_test"], naive_pred)

#Print final results header
print("\n--- FINAL RESULTS ---")

#Print blend weight
print(f"Validation-tuned naive blend weight: {enhanced_result['blend_weight']:.4f}")

#Print validation metrics
print_metrics(f"{MODEL_NAME} Validation Metrics", enhanced_result["val_metrics"])

#Print test metrics
print_metrics(f"{MODEL_NAME} Test Metrics", enhanced_result["test_metrics"])

#Print naive metrics
print_metrics("Naive Test Metrics", naive_metrics)

#Build metrics table
metrics_df = pd.DataFrame(
    [
        {"Dataset": "Validation", "Model": MODEL_NAME, "FeatureSet": FEATURE_SET, "Scenario": enhanced_label, **enhanced_result["val_metrics"]},
        {"Dataset": "Test", "Model": MODEL_NAME, "FeatureSet": FEATURE_SET, "Scenario": enhanced_label, **enhanced_result["test_metrics"]},
        {"Dataset": "Naive_Test", "Model": "pm25_lag1", "Scenario": "lag1", **naive_metrics},
    ]
)

#Save metrics table
metrics_df.to_csv(METRICS_OUTPUT_FILE, index=False)

#Print metrics path
print("\nSaved evaluation metrics to:", METRICS_OUTPUT_FILE)

#Save coefficients
enhanced_result["coef_df"].to_csv(COEF_OUTPUT_FILE, index=False)

#Print coefficient path
print("Saved coefficients to:", COEF_OUTPUT_FILE)

#Print coefficient header
print("\nTop 15 Coefficients:")

#Print top coefficients
print(enhanced_result["coef_df"].head(15).to_string(index=False))

#Save comparison outputs
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

    #Print comparison path
    print("Saved ERA5 comparison metrics to:", COMPARISON_METRICS_OUTPUT_FILE)

    #Save comparison plot
    save_comparison_plot(baseline_result, enhanced_result)

#Explain why comparison skipped
else:
    print("Skipped ERA5 comparison outputs because the baseline scenario was not run.")

#Save optional main plot
save_main_plot(enhanced_result)

#Run baseline forecast only
if RUN_FORECAST and not USE_ERA5 and not COMPARE_ERA5:
    #Start 2023 forecasting
    print("\n--- FORECASTING 2023 ---")

    #Build full history
    history = pd.concat([train, val, test], ignore_index=True)[["lat", "lon", "date", TARGET]].copy()

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

        #Set future date
        base["date"] = next_date

        #Append empty future target
        temp = pd.concat([history, base.assign(pm25=np.nan)], ignore_index=True)

        #Sort for lag features
        temp = temp.sort_values(["lat", "lon", "date"]).reset_index(drop=True)

        #Add history features
        temp = add_history_features(temp, target=TARGET)

        #Add train-only climatology
        temp = add_train_only_climatology(temp, train_end=TRAIN_END, target=TARGET)

        #Add engineered features
        temp = add_experimental_features(temp)

        #Add future-safe ERA5 columns
        temp, future_era5_feature_names = add_era5_features(
            temp,
            raw_dir=RAW_DIR,
            train_end=TRAIN_END,
            use_era5=False,
        )

        #Build LR feature sets
        future_feature_sets = build_linear_feature_sets(future_era5_feature_names)

        #Get matching feature set
        future_features = future_feature_sets[FEATURE_SET]

        #Keep next-month rows
        future_rows = temp[temp["date"] == next_date].copy()

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

        #Count missing values before fill
        missing_counts = future_rows[active_features].isna().sum()
        missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

        #Print missing features if any
        if not missing_counts.empty:
            print("Missing values before fill:")
            print(missing_counts.to_string())

        #Build future matrix in training order
        X_future = future_rows.loc[:, active_features].copy()

        #Fill missing values from training medians
        X_future = X_future.fillna(train_feature_fill_values)

        #Fill anything still missing with zero
        X_future = X_future.fillna(0.0)

        #Stop if no rows remain
        if X_future.empty:
            raise ValueError("Forecast matrix is empty after feature construction.")

        #Stop if NaNs still remain
        if X_future.isna().any().any():
            remaining_missing = X_future.isna().sum()
            remaining_missing = remaining_missing[remaining_missing > 0]
            raise ValueError(
                "Forecast matrix still contains NaNs after fill:\n"
                + remaining_missing.to_string()
            )

        #Predict future PM2.5
        preds = inverse_target(model.predict(X_future))

        #Clip to train range
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
        "forecast-ready ridge path."
    )
