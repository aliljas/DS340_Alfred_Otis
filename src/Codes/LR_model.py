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

    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from common_model_utils import (
        build_comparison_table,
        build_metrics_table,
        compute_metrics,
        inverse_target as shared_inverse_target,
        print_metrics,
        print_run_configuration,
        run_recursive_forecast,
        save_era5_comparison_plot,
        save_main_results_plot,
        split_train_val_test,
        transform_target as shared_transform_target,
    )
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
PROCESSED_DIR = Path(os.getenv("LR_OUTPUT_DIR", str(BASE_DIR / "data/processed")))

#Set raw directory
RAW_DIR = BASE_DIR / "data/raw"

#Make processed directory
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

#Make raw directory
RAW_DIR.mkdir(parents=True, exist_ok=True)

#Set cleaned data file
DATA_FILE = Path(os.getenv("LR_DATA_FILE", str(PROCESSED_DIR / "na_pm25_cells_clean.csv")))

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

#Set forecast horizon
FORECAST_MONTHS = int(os.getenv("LR_FORECAST_MONTHS", "12"))

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


def transform_target(values):
    return shared_transform_target(values, TARGET_TRANSFORM, "LR_TARGET_TRANSFORM")


def inverse_target(values):
    return shared_inverse_target(values, TARGET_TRANSFORM)


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
train, val, test = split_train_val_test(df, TRAIN_END, VAL_END)

#Free memory
del df

#Build fill values from training data
train_feature_fill_values = train[active_features].median(numeric_only=True)

#Fill any remaining missing medians with zero
train_feature_fill_values = train_feature_fill_values.fillna(0.0)

print_run_configuration(FEATURE_SET, active_features, TARGET_TRANSFORM, ERA5_FEATURE_LEVEL, train, val, test)

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
metrics_df = build_metrics_table(
    MODEL_NAME,
    FEATURE_SET,
    enhanced_label,
    enhanced_result["val_metrics"],
    enhanced_result["test_metrics"],
    naive_metrics,
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
    comparison_df = build_comparison_table(FEATURE_SET, baseline_result, enhanced_result)

    #Save comparison CSV
    comparison_df.to_csv(COMPARISON_METRICS_OUTPUT_FILE, index=False)

    #Print comparison path
    print("Saved ERA5 comparison metrics to:", COMPARISON_METRICS_OUTPUT_FILE)

    #Save comparison plot
    save_era5_comparison_plot(
        baseline_result,
        enhanced_result,
        COMPARISON_PLOT_OUTPUT_FILE,
        SAVE_COMPARISON_PLOT,
        "Skipped comparison plot generation. Set LR_SAVE_COMPARISON_PLOT=1 to save it.",
        "Ridge Regression",
        RANDOM_SEED,
    )

#Explain why comparison skipped
else:
    print("Skipped ERA5 comparison outputs because the baseline scenario was not run.")

#Save optional main plot
save_main_results_plot(
    enhanced_result,
    PLOT_OUTPUT_FILE,
    SAVE_PLOT,
    "Skipped plot generation. Set LR_SAVE_PLOT=1 to save plots.",
    "coef_df",
    "abs_coefficient",
    "Top 15 Absolute Coefficients",
    RANDOM_SEED,
    sort_column="abs_coefficient",
)

#Run baseline forecast only
if RUN_FORECAST and not USE_ERA5 and not COMPARE_ERA5:
    run_recursive_forecast(
        model=model,
        history_frame=pd.concat([train, val, test], ignore_index=True),
        train_frame=train,
        active_features=active_features,
        feature_set_name=FEATURE_SET,
        output_file=PRED_OUTPUT_FILE,
        target=TARGET,
        train_end=TRAIN_END,
        raw_dir=RAW_DIR,
        build_feature_sets_fn=build_linear_feature_sets,
        add_history_features_fn=add_history_features,
        add_train_only_climatology_fn=add_train_only_climatology,
        add_experimental_features_fn=add_experimental_features,
        add_era5_features_fn=add_era5_features,
        predict_fn=lambda fitted_model, X_future: fitted_model.predict(X_future),
        inverse_transform_fn=inverse_target,
        fill_values=train_feature_fill_values,
        forecast_months=FORECAST_MONTHS,
        era5_feature_level=ERA5_FEATURE_LEVEL,
    )

#Skip forecast in ERA or comparison mode
elif RUN_FORECAST:
    print(
        "\nSkipped forecasting because this run used "
        "ERA5 or comparison mode. The baseline no-ERA run remains the strict "
        "forecast-ready ridge path."
    )
