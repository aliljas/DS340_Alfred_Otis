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
    from catboost import CatBoostRegressor
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
        "Required packages for CatBoost_model.py are not installed. "
        "Install the project requirements first, then rerun this script."
    ) from exc


#Paths/config
#Use project-relative paths
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data/processed"
RAW_DIR = BASE_DIR / "data/raw"
PROCESSED_DIR = Path(os.getenv("CATBOOST_OUTPUT_DIR", str(PROCESSED_DIR)))
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_FILE = Path(os.getenv("CATBOOST_DATA_FILE", str(PROCESSED_DIR / "na_pm25_cells_clean.csv")))

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


def transform_target(values):
    return shared_transform_target(values, TARGET_TRANSFORM, "CATBOOST_TARGET_TRANSFORM")


def inverse_target(values):
    return shared_inverse_target(values, TARGET_TRANSFORM)

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
train, val, test = split_train_val_test(df, TRAIN_END, VAL_END)
observed_history = pd.concat([train, val, test], ignore_index=True)
del df

print_run_configuration(FEATURE_SET, active_features, TARGET_TRANSFORM, ERA5_FEATURE_LEVEL, train, val, test)

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

#Store training medians for forecast fill
train_feature_fill_values = train[active_features].median(numeric_only=True)

#Fill any all-missing medians
train_feature_fill_values = train_feature_fill_values.fillna(0.0)

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
#Save importance table
enhanced_result["importance_df"].to_csv(IMPORTANCE_OUTPUT_FILE, index=False)
print("Saved feature importance to:", IMPORTANCE_OUTPUT_FILE)
print("\nTop 15 Features:")
print(enhanced_result["importance_df"].head(15).to_string(index=False))

#Save ERA comparison only when both scenarios were run
if baseline_result is not None:
    #Comparison table
    comparison_df = build_comparison_table(FEATURE_SET, baseline_result, enhanced_result)
    #Save comparison CSV
    comparison_df.to_csv(COMPARISON_METRICS_OUTPUT_FILE, index=False)
    print("Saved ERA5 comparison metrics to:", COMPARISON_METRICS_OUTPUT_FILE)
    #Save comparison figure
    save_era5_comparison_plot(
        baseline_result,
        enhanced_result,
        COMPARISON_PLOT_OUTPUT_FILE,
        SAVE_COMPARISON_PLOT,
        "Skipped comparison plot generation. Set CATBOOST_SAVE_COMPARISON_PLOT=1 to save it.",
        "CatBoost",
        RANDOM_SEED,
    )
else:
    #Skip comparison when only one scenario was trained
    print("Skipped ERA5 comparison outputs because the baseline scenario was not run.")

#Optional main figure
save_main_results_plot(
    enhanced_result,
    PLOT_OUTPUT_FILE,
    SAVE_PLOT,
    "Skipped plot generation. Set CATBOOST_SAVE_PLOT=1 to save plots.",
    "importance_df",
    "importance",
    "Top 15 Feature Importance",
    RANDOM_SEED,
)

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
        prepare_model_frame_fn=lambda frame, feature_list: prepare_model_frame(
            frame,
            feature_list,
            [col for col in ["month", "region_lat_bin", "region_lon_bin"] if col in feature_list],
        ),
    )
#Explain why forecast was skipped in other modes
elif RUN_FORECAST:
    print(
        "Skipped recursive forecast because the current CatBoost run is using "
        "ERA5 or comparison mode. The baseline no-ERA run remains the strict "
        "forecasting path."
    )
