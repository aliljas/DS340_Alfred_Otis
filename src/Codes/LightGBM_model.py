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
    import numpy as np
    import pandas as pd
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
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
        "Required packages for LightGBM_model.py are not installed. "
        "Install the project requirements first, then rerun this script."
    ) from exc


#Paths/config
#Use project-relative paths
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data/processed"
RAW_DIR = BASE_DIR / "data/raw"
PROCESSED_DIR = Path(os.getenv("LIGHTGBM_OUTPUT_DIR", str(PROCESSED_DIR)))
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_FILE = Path(os.getenv("LIGHTGBM_DATA_FILE", str(PROCESSED_DIR / "na_pm25_cells_clean.csv")))

TRAIN_END = pd.Timestamp("2021-01-01")
VAL_END = pd.Timestamp("2022-01-01")
RANDOM_SEED = 42

FEATURE_SET = os.getenv("LIGHTGBM_FEATURE_SET", "trend_region")
SAVE_PLOT = os.getenv("LIGHTGBM_SAVE_PLOT", "0") == "1"
SAVE_COMPARISON_PLOT = os.getenv("LIGHTGBM_SAVE_COMPARISON_PLOT", "1") == "1"
COMPARE_ERA5 = os.getenv("LIGHTGBM_COMPARE_ERA5", "0") == "1"
MODEL_NAME = "LightGBM"
TARGET_TRANSFORM = os.getenv("LIGHTGBM_TARGET_TRANSFORM", "log1p")
CATEGORICAL_FEATURES = ["month", "region_lat_bin", "region_lon_bin"]
USE_ERA5 = os.getenv("LIGHTGBM_USE_ERA5", "0") == "1"
ERA5_FEATURE_LEVEL = os.getenv("LIGHTGBM_ERA5_FEATURE_LEVEL", "extended")
RUN_FORECAST = os.getenv("LIGHTGBM_RUN_FORECAST", "1") == "1"
FORECAST_MONTHS = int(os.getenv("LIGHTGBM_FORECAST_MONTHS", "12"))

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


def transform_target(values):
    return shared_transform_target(values, TARGET_TRANSFORM, "LIGHTGBM_TARGET_TRANSFORM")


def inverse_target(values):
    return shared_inverse_target(values, TARGET_TRANSFORM)


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
train, val, test = split_train_val_test(df, TRAIN_END, VAL_END)

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

print_run_configuration(FEATURE_SET, active_features, TARGET_TRANSFORM, ERA5_FEATURE_LEVEL, train, val, test)

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

#Feature importance
enhanced_result["importance_df"].to_csv(IMPORTANCE_OUTPUT_FILE, index=False)
print("Saved feature importance to:", IMPORTANCE_OUTPUT_FILE)
print("\nTop 15 Features:")
print(enhanced_result["importance_df"].head(15).to_string(index=False))

#Save ERA comparison only when both scenarios were run
if baseline_result is not None:
    comparison_df = build_comparison_table(FEATURE_SET, baseline_result, enhanced_result)
    comparison_df.to_csv(COMPARISON_METRICS_OUTPUT_FILE, index=False)
    print("Saved ERA5 comparison metrics to:", COMPARISON_METRICS_OUTPUT_FILE)
    save_era5_comparison_plot(
        baseline_result,
        enhanced_result,
        COMPARISON_PLOT_OUTPUT_FILE,
        SAVE_COMPARISON_PLOT,
        "Skipped comparison plot generation. Set LIGHTGBM_SAVE_COMPARISON_PLOT=1 to save it.",
        "LightGBM",
        RANDOM_SEED,
        sample_size=100_000,
    )
else:
    #Skip comparison when only one scenario was trained
    print("Skipped ERA5 comparison outputs because the baseline scenario was not run.")

#Optional main figure
save_main_results_plot(
    enhanced_result,
    PLOT_OUTPUT_FILE,
    SAVE_PLOT,
    "Skipped plot generation. Set LIGHTGBM_SAVE_PLOT=1 to save plots.",
    "importance_df",
    "importance",
    "Top 15 Feature Importance",
    RANDOM_SEED,
)

#Store training medians for forecast fill
train_feature_fill_values = train[active_features].median(numeric_only=True)

#Fill any all-missing medians
train_feature_fill_values = train_feature_fill_values.fillna(0.0)

#Run baseline forecast only
if RUN_FORECAST and not USE_ERA5 and not COMPARE_ERA5:
    run_recursive_forecast(
        model=enhanced_result["model"],
        history_frame=df,
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
        predict_fn=lambda fitted_model, X_future: fitted_model.predict(
            X_future,
            num_iteration=fitted_model.best_iteration_,
        ),
        inverse_transform_fn=inverse_target,
        fill_values=train_feature_fill_values,
        forecast_months=FORECAST_MONTHS,
        era5_feature_level=ERA5_FEATURE_LEVEL,
        prepare_model_frame_fn=prepare_model_frame,
    )

#Skip forecast in ERA or comparison mode
elif RUN_FORECAST:
    print(
        "\nSkipped forecasting because this run used "
        "ERA5 or comparison mode. The baseline no-ERA run remains the strict "
        "forecast-ready LightGBM path."
    )
