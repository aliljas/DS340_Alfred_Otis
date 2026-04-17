import os
import tempfile
from pathlib import Path

# Use a writable Matplotlib cache so plots work on classroom machines too.
MPLCONFIGDIR = Path(tempfile.gettempdir()) / "matplotlib-cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

try:
    # Use a non-interactive backend because this script saves figures to disk.
    import matplotlib
    matplotlib.use("Agg")

    # Core numerical and table libraries.
    import numpy as np
    import pandas as pd

    # XGBoost model.
    from xgboost import XGBRegressor

    # Shared project helpers so this script follows the same pipeline as the
    # rest of the repository.
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
        build_feature_sets,
        prepare_modeling_frame,
    )
except ImportError as exc:
    raise SystemExit(
        "Required packages for XGB_model.py are not installed. "
        "Install the project requirements first, then rerun this script."
    ) from exc


# ---------------------------------------------------------------------------
# PATHS AND GLOBAL CONFIG
# ---------------------------------------------------------------------------

# Anchor the script to the project root so paths stay portable.
BASE_DIR = Path(__file__).resolve().parents[2]

# Allow the class demo to redirect outputs into its own folder.
PROCESSED_DIR = Path(os.getenv("XGB_OUTPUT_DIR", str(BASE_DIR / "data/processed")))
RAW_DIR = BASE_DIR / "data/raw"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Allow the class demo to hand this script a sampled CSV instead of the full
# cleaned modeling table.
DATA_FILE = Path(os.getenv("XGB_DATA_FILE", str(PROCESSED_DIR / "na_pm25_cells_clean.csv")))

# Keep the same train / validation / test boundaries as the rest of the project.
TRAIN_END = pd.Timestamp("2021-01-01")
VAL_END = pd.Timestamp("2022-01-01")
RANDOM_SEED = 42

# Feature-set choice and run behavior.
FEATURE_SET = os.getenv("XGB_FEATURE_SET", "trend_region")
TARGET_TRANSFORM = os.getenv("XGB_TARGET_TRANSFORM", "log1p")
SAVE_PLOT = os.getenv("XGB_SAVE_PLOT", "0") == "1"
SAVE_COMPARISON_PLOT = os.getenv("XGB_SAVE_COMPARISON_PLOT", "1") == "1"
COMPARE_ERA5 = os.getenv("XGB_COMPARE_ERA5", "0") == "1"
USE_ERA5 = os.getenv("XGB_USE_ERA5", "0") == "1"
ERA5_FEATURE_LEVEL = os.getenv("XGB_ERA5_FEATURE_LEVEL", "core")
RUN_FORECAST = os.getenv("XGB_RUN_FORECAST", "1") == "1"
FORECAST_MONTHS = int(os.getenv("XGB_FORECAST_MONTHS", "12"))

# When the classroom demo calls this script on sampled data, it can flip this
# flag on so we skip any slower tuning workflow and use a fixed parameter set.
SKIP_OPTUNA = os.getenv("XGB_SKIP_OPTUNA", "0") == "1"

MODEL_NAME = "XGBoost"

# Stable project-style XGBoost settings. These are used directly in demo mode
# and also act as the main default configuration for regular runs.
XGB_PARAMS = {
    "n_estimators": 900,
    "learning_rate": 0.06,
    "max_depth": 6,
    "min_child_weight": 10,
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

# Keep the output filenames aligned with the rest of the repository.
PRED_OUTPUT_FILE = PROCESSED_DIR / "xgb_predictions_2023.csv"
METRICS_OUTPUT_FILE = PROCESSED_DIR / "xgb_eval_metrics.csv"
IMPORTANCE_OUTPUT_FILE = PROCESSED_DIR / "xgb_feature_importance.csv"
PLOT_OUTPUT_FILE = PROCESSED_DIR / "xgb_model_results.png"
COMPARISON_METRICS_OUTPUT_FILE = PROCESSED_DIR / "xgb_era5_comparison_metrics.csv"
COMPARISON_PLOT_OUTPUT_FILE = PROCESSED_DIR / "xgb_era5_comparison.png"


# ---------------------------------------------------------------------------
# TARGET TRANSFORMS
# ---------------------------------------------------------------------------

def transform_target(values):
    # Reuse the shared transform helper so this matches the other model scripts.
    return shared_transform_target(values, TARGET_TRANSFORM, "XGB_TARGET_TRANSFORM")


def inverse_target(values):
    # Reuse the shared inverse-transform helper so predictions land back on the
    # original PM2.5 scale.
    return shared_inverse_target(values, TARGET_TRANSFORM)


# ---------------------------------------------------------------------------
# MODEL FITTING
# ---------------------------------------------------------------------------

def fit_scenario(label, features, train, val, test):
    # Build the feature matrices for train, validation, and test.
    X_train = train[features]
    X_val = val[features]
    X_test = test[features]

    # Train in transformed space but score in the original PM2.5 units.
    y_train = transform_target(train[TARGET].values)
    y_val = transform_target(val[TARGET].values)
    y_val_raw = val[TARGET].values
    y_test = test[TARGET].values

    print(f"\n--- TRAINING {MODEL_NAME.upper()} ({label}) ---")

    # In sampled classroom runs, skip any expensive tuning logic and just fit
    # the fixed project configuration. This keeps the live demo fast.
    if SKIP_OPTUNA:
        print("Skipping Optuna/tuning because XGB_SKIP_OPTUNA=1.")

    model = XGBRegressor(
        eval_metric="rmse",
        early_stopping_rounds=50,
        **XGB_PARAMS,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )
    print("--- DONE ---")

    # XGBoost returns a zero-based best iteration, so add 1 to get the number
    # of boosting rounds we actually want to use for prediction.
    best_iteration = model.best_iteration + 1 if model.best_iteration is not None else XGB_PARAMS["n_estimators"]

    # Predict validation and test on the original PM2.5 scale.
    val_pred = inverse_target(model.predict(X_val, iteration_range=(0, best_iteration)))
    test_pred = inverse_target(model.predict(X_test, iteration_range=(0, best_iteration)))

    # Save feature importance in a project-friendly tabular form.
    importance_df = pd.DataFrame(
        {
            "feature": features,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    # Return everything the rest of the script needs for metrics, plots, and
    # forecasting.
    return {
        "label": label,
        "features": features,
        "model": model,
        "best_iteration": best_iteration,
        "X_test": X_test,
        "y_test": y_test,
        "test_pred": test_pred,
        "val_metrics": compute_metrics(y_val_raw, val_pred),
        "test_metrics": compute_metrics(y_test, test_pred),
        "importance_df": importance_df,
    }


def fit_forecast_model(observed_history, features, best_iteration):
    # Refit XGBoost on all observed data before generating the recursive 2023
    # forecast path.
    X_full = observed_history[features]
    y_full = transform_target(observed_history[TARGET].values)

    print(f"\n--- RETRAINING {MODEL_NAME.upper()} FOR FORECAST ---")

    forecast_model = XGBRegressor(
        eval_metric="rmse",
        n_estimators=max(1, int(best_iteration)),
        **{key: value for key, value in XGB_PARAMS.items() if key != "n_estimators"},
    )
    forecast_model.fit(X_full, y_full, verbose=False)
    return forecast_model


# ---------------------------------------------------------------------------
# DATA LOADING AND SHARED FEATURE PIPELINE
# ---------------------------------------------------------------------------

# Build the modeling table through the shared project pipeline so XGBoost uses
# the same cleaned data and feature logic as the other models.
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

# Split by time so the evaluation remains forecast-style rather than random.
train, val, test = split_train_val_test(df, TRAIN_END, VAL_END)

# Build the no-ERA and ERA-aware feature-set dictionaries from the shared
# project utility.
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

print_run_configuration(FEATURE_SET, active_features, TARGET_TRANSFORM, ERA5_FEATURE_LEVEL, train, val, test)


# ---------------------------------------------------------------------------
# TRAIN, SCORE, AND SAVE OUTPUTS
# ---------------------------------------------------------------------------

baseline_result = None
if COMPARE_ERA5 and era5_feature_names:
    # Train a no-ERA baseline only when we explicitly want a before/after ERA5
    # comparison table or figure.
    baseline_result = fit_scenario("Without ERA5", baseline_features, train, val, test)

enhanced_label = "With ERA5" if era5_feature_names else "Without ERA5"
enhanced_result = fit_scenario(enhanced_label, active_features, train, val, test)

# Keep the lag-1 naive benchmark for context in the metrics table.
naive_pred = enhanced_result["X_test"]["pm25_lag1"].values
naive_metrics = compute_metrics(enhanced_result["y_test"], naive_pred)

print("\n--- FINAL RESULTS ---")
print_metrics(f"{MODEL_NAME} Validation Metrics", enhanced_result["val_metrics"])
print_metrics(f"{MODEL_NAME} Test Metrics", enhanced_result["test_metrics"])
print_metrics("Naive Test Metrics", naive_metrics)

metrics_df = build_metrics_table(
    MODEL_NAME,
    FEATURE_SET,
    enhanced_label,
    enhanced_result["val_metrics"],
    enhanced_result["test_metrics"],
    naive_metrics,
)
metrics_df.to_csv(METRICS_OUTPUT_FILE, index=False)
print("\nSaved evaluation metrics to:", METRICS_OUTPUT_FILE)

enhanced_result["importance_df"].to_csv(IMPORTANCE_OUTPUT_FILE, index=False)
print("Saved feature importance to:", IMPORTANCE_OUTPUT_FILE)
print("\nTop 15 Features:")
print(enhanced_result["importance_df"].head(15).to_string(index=False))

if baseline_result is not None:
    comparison_df = build_comparison_table(FEATURE_SET, baseline_result, enhanced_result)
    comparison_df.to_csv(COMPARISON_METRICS_OUTPUT_FILE, index=False)
    print("Saved ERA5 comparison metrics to:", COMPARISON_METRICS_OUTPUT_FILE)
    save_era5_comparison_plot(
        baseline_result,
        enhanced_result,
        COMPARISON_PLOT_OUTPUT_FILE,
        SAVE_COMPARISON_PLOT,
        "Skipped comparison plot generation. Set XGB_SAVE_COMPARISON_PLOT=1 to save it.",
        "XGBoost",
        RANDOM_SEED,
    )
else:
    print("Skipped ERA5 comparison outputs because the baseline scenario was not run.")

save_main_results_plot(
    enhanced_result,
    PLOT_OUTPUT_FILE,
    SAVE_PLOT,
    "Skipped plot generation. Set XGB_SAVE_PLOT=1 to save plots.",
    "importance_df",
    "importance",
    "Top 15 Feature Importance",
    RANDOM_SEED,
)


# ---------------------------------------------------------------------------
# STRICT NO-ERA FORECAST PATH
# ---------------------------------------------------------------------------

if RUN_FORECAST and not USE_ERA5 and not COMPARE_ERA5:
    # Refit the final XGBoost model on all observed history before forecasting.
    observed_history = pd.concat([train, val, test], ignore_index=True)
    forecast_model = fit_forecast_model(
        observed_history,
        active_features,
        enhanced_result["best_iteration"],
    )

    # Fill any remaining forecast-time feature gaps with training medians.
    train_feature_fill_values = train[active_features].median(numeric_only=True).fillna(0.0)

    # Forecast month-by-month using the shared recursive forecast helper.
    run_recursive_forecast(
        model=forecast_model,
        history_frame=observed_history,
        train_frame=train,
        active_features=active_features,
        feature_set_name=FEATURE_SET,
        output_file=PRED_OUTPUT_FILE,
        target=TARGET,
        train_end=TRAIN_END,
        raw_dir=RAW_DIR,
        build_feature_sets_fn=build_feature_sets,
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
elif RUN_FORECAST:
    print(
        "\nSkipped forecasting because this run used "
        "ERA5 or comparison mode. The baseline no-ERA run remains the strict "
        "forecast-ready XGBoost path."
    )
