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
        build_feature_sets,
        prepare_modeling_frame,
    )
except ImportError as exc:
    raise SystemExit(
        "Required packages for LR_model.py are not installed. "
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

#Model options
FEATURE_SET = os.getenv("LR_FEATURE_SET", "linear_core")
TARGET_TRANSFORM = os.getenv("LR_TARGET_TRANSFORM", "none")
SAVE_PLOT = os.getenv("LR_SAVE_PLOT", "0") == "1"
SAVE_COMPARISON_PLOT = os.getenv("LR_SAVE_COMPARISON_PLOT", "1") == "1"
COMPARE_ERA5 = os.getenv("LR_COMPARE_ERA5", "0") == "1"
USE_ERA5 = os.getenv("LR_USE_ERA5", "0") == "1"
ERA5_FEATURE_LEVEL = os.getenv("LR_ERA5_FEATURE_LEVEL", "core")
MODEL_NAME = "RidgeRegression"

#Saved outputs
PRED_OUTPUT_FILE = PROCESSED_DIR / "lr_predictions_2023.csv"
METRICS_OUTPUT_FILE = PROCESSED_DIR / "lr_eval_metrics.csv"
COEF_OUTPUT_FILE = PROCESSED_DIR / "lr_coefficients.csv"
PLOT_OUTPUT_FILE = PROCESSED_DIR / "lr_model_results.png"
COMPARISON_METRICS_OUTPUT_FILE = PROCESSED_DIR / "lr_era5_comparison_metrics.csv"
COMPARISON_PLOT_OUTPUT_FILE = PROCESSED_DIR / "lr_era5_comparison.png"


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
    raise ValueError("LR_TARGET_TRANSFORM must be 'log1p' or 'none'")


#Target inverse transform
def inverse_target(values):
    if TARGET_TRANSFORM == "log1p":
        return np.expm1(values)
    return values


#Validation blend helper
def find_blend_weight(y_true, model_pred, naive_pred):
    diff = model_pred - naive_pred
    denom = float(np.dot(diff, diff))
    if denom <= 0:
        return 0.0
    numer = float(np.dot(diff, y_true - naive_pred))
    return float(np.clip(numer / denom, 0.0, 1.0))


#Linear feature list
def build_linear_feature_sets(era5_feature_names=None):
    era5_feature_names = list(era5_feature_names or [])
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

    return {
        "linear_core": base_features + list(era5_feature_names),
    }


#One linear regression scenario
def fit_scenario(label, feature_list, train_frame, val_frame, test_frame):
    #Feature matrices
    X_train = train_frame[feature_list]
    X_val = val_frame[feature_list]
    X_test = test_frame[feature_list]

    #Targets
    y_train = transform_target(train_frame[TARGET].values)
    y_val_raw = val_frame[TARGET].values
    y_test = test_frame[TARGET].values
    #Naive references
    naive_val = val_frame["pm25_lag1"].astype(float).values
    naive_test = test_frame["pm25_lag1"].astype(float).values
    #Raw bounds
    y_train_raw_min = float(np.min(train_frame[TARGET].values))
    y_train_raw_max = float(np.max(train_frame[TARGET].values))

    print(f"\n--- TRAINING {MODEL_NAME.upper()} ({label}) ---")
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=20.0, random_state=RANDOM_SEED),
    )
    model.fit(X_train, y_train)
    print("--- DONE ---")

    #Predictions on transformed scale
    val_pred_trans = model.predict(X_val)
    test_pred_trans = model.predict(X_test)
    #Predictions on original scale
    val_pred = inverse_target(val_pred_trans)
    test_pred = inverse_target(test_pred_trans)
    #Clip on PM2.5 scale
    val_pred = np.clip(val_pred, y_train_raw_min, y_train_raw_max)
    test_pred = np.clip(test_pred, y_train_raw_min, y_train_raw_max)
    #Validation-tuned blend weight
    blend_weight = find_blend_weight(y_val_raw, val_pred, naive_val)
    #Shrink linear predictions toward naive
    val_pred = blend_weight * val_pred + (1.0 - blend_weight) * naive_val
    test_pred = blend_weight * test_pred + (1.0 - blend_weight) * naive_test
    #Coefficient table
    coef_values = model.named_steps["ridge"].coef_
    coef_df = pd.DataFrame(
        {
            "feature": feature_list,
            "coefficient": coef_values,
            "abs_coefficient": np.abs(coef_values),
        }
    ).sort_values("abs_coefficient", ascending=False)

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


#Main summary plot
def save_main_plot(result):
    if not SAVE_PLOT:
        print("Skipped plot generation. Set LR_SAVE_PLOT=1 to save plots.")
        return

    #Create a 3-panel figure
    fig = plt.figure(figsize=(18, 5))

    #First panel
    plt.subplot(1, 3, 1)
    #Top 15 coefficients
    top_coef = result["coef_df"].head(15).sort_values("abs_coefficient")
    #Plot coefficient bars
    plt.barh(top_coef["feature"], top_coef["abs_coefficient"])
    #Panel title
    plt.title("Top 15 Absolute Coefficients")

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
def save_comparison_plot(baseline_result, enhanced_result):
    if not SAVE_COMPARISON_PLOT:
        print("Skipped comparison plot generation. Set LR_SAVE_COMPARISON_PLOT=1 to save it.")
        return

    #Create a 2x2 figure
    fig = plt.figure(figsize=(12, 10))

    #First panel
    plt.subplot(2, 2, 1)
    #Shared labels
    labels = ["Validation", "Test"]
    #Bar positions
    x = np.arange(len(labels))
    #Bar width
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
    plt.title("Ridge Regression R^2 Before vs After ERA5")
    #Legend
    plt.legend()

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
    plt.title("Ridge Regression RMSE Before vs After ERA5")
    #Legend
    plt.legend()

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
#Build the shared modeling table
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

feature_sets_no_era5 = build_linear_feature_sets([])
feature_sets_with_era5 = build_linear_feature_sets(era5_feature_names)
if FEATURE_SET not in feature_sets_with_era5:
    raise ValueError(
        f"Unknown LR_FEATURE_SET={FEATURE_SET!r}. "
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
train = df[df["date"] < TRAIN_END].copy()
val = df[(df["date"] >= TRAIN_END) & (df["date"] < VAL_END)].copy()
test = df[df["date"] >= VAL_END].copy()
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
model = enhanced_result["model"]

#Naive lag-1 baseline
naive_pred = enhanced_result["X_test"]["pm25_lag1"].astype(float).values
naive_metrics = compute_metrics(enhanced_result["y_test"], naive_pred)

#Print final metrics
print("\n--- FINAL RESULTS ---")
print(f"Validation-tuned naive blend weight: {enhanced_result['blend_weight']:.4f}")
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

#Coefficients
enhanced_result["coef_df"].to_csv(COEF_OUTPUT_FILE, index=False)
print("Saved coefficients to:", COEF_OUTPUT_FILE)
print("\nTop 15 Coefficients:")
print(enhanced_result["coef_df"].head(15).to_string(index=False))

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
print("\n--- FORECASTING 2023 ---")
history = pd.concat([train, val, test], ignore_index=True)[["lat", "lon", "date", TARGET]].copy()
future_preds = []
latest_date = history["date"].max()

for _ in range(12):
    #Advance one month
    next_date = latest_date + pd.DateOffset(months=1)

    #Placeholder rows
    base = history[history["date"] == latest_date][["lat", "lon"]].copy()
    base["date"] = next_date

    #Rebuild causal features
    temp = pd.concat([history, base.assign(pm25=np.nan)], ignore_index=True)
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

    #Future feature rows
    future_feature_sets = build_feature_sets(future_era5_feature_names)
    future_features = future_feature_sets[FEATURE_SET]
    future_rows = temp[temp["date"] == next_date].copy()
    future_rows = future_rows.dropna(subset=future_features)

    #Predict next month
    preds = inverse_target(model.predict(future_rows[future_features]))
    future_rows[TARGET] = preds

    #Append predictions to history
    history = pd.concat(
        [history, future_rows[["lat", "lon", "date", TARGET]]],
        ignore_index=True,
    )
    future_preds.append(future_rows[["lat", "lon", "date", TARGET]].copy())
    latest_date = next_date

#Save full forecast
final_df = pd.concat(future_preds, ignore_index=True)
final_df.to_csv(PRED_OUTPUT_FILE, index=False)
print("\nSaved predictions to:", PRED_OUTPUT_FILE)
