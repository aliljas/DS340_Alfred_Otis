"""
Joint technical demo script for the DS 340W project.

This script is designed for a fresh classroom computer.
Students can use the prebuilt cleaned modeling table, run this one file,
and then explain the code section by section.
"""

# Import libraries.
import argparse
# Import temporary-directory helpers.
import tempfile
# Import ZIP extraction helpers.
import zipfile
# Import path helpers.
from pathlib import Path

# Import operating-system helpers.
import os

# Set the matplotlib cache folder so plotting libraries behave on lab machines.
MPLCONFIGDIR = Path(tempfile.gettempdir()) / "matplotlib-cache"
# Create the matplotlib cache folder if it does not exist yet.
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
# Tell matplotlib where to store its cache files.
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

# Import NumPy for numerical work.
import numpy as np
# Import Pandas for table handling.
import pandas as pd
# Import CatBoost for one of the tree models.
from catboost import CatBoostRegressor
# Import LightGBM for one of the tree models.
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
# Import Ridge regression for the linear baseline model.
from sklearn.linear_model import Ridge
# Import a pipeline wrapper for Ridge.
from sklearn.pipeline import make_pipeline
# Import a scaler so Ridge sees standardized inputs.
from sklearn.preprocessing import StandardScaler
# Import XGBoost for the boosted-tree model.
from xgboost import XGBRegressor

# Import shared metric-printing and forecast helpers from the project utilities.
from common_model_utils import compute_metrics, print_metrics, run_recursive_forecast
# Import the shared project feature-engineering utilities.
from model_feature_utils import (
    TARGET,
    add_era5_features,
    add_experimental_features,
    add_history_features,
    add_train_only_climatology,
    build_feature_sets,
)
# Set the project root directory.
BASE_DIR = Path(__file__).resolve().parents[2]
# Set the raw-data folder.
RAW_DIR = BASE_DIR / "data/raw"
# Set the processed-data folder.
PROCESSED_DIR = BASE_DIR / "data/processed"
# Set the classroom-demo output folder.
DEMO_DIR = PROCESSED_DIR / "technical_demo"
# Set the cleaned monthly modeling table path.
DATA_FILE = PROCESSED_DIR / "na_pm25_cells_clean.csv"

# Keep the same train cutoff used in the main project scripts.
TRAIN_END = pd.Timestamp("2021-01-01")
# Keep the same validation cutoff used in the main project scripts.
VAL_END = pd.Timestamp("2022-01-01")
# Keep one fixed random seed for reproducibility.
RANDOM_SEED = 42

# Store the likely cleaned-data ZIP locations the class computer may receive.
CLEANED_DATA_ZIP_CANDIDATES = [
    PROCESSED_DIR / "na_pm25_cells_clean.csv.zip",
    RAW_DIR / "na_pm25_cells_clean.csv.zip",
    RAW_DIR / "na_pm25_cells_clean.zip",
]


# Parse command-line options for the classroom demo.
def parse_args():
    # Create the argument parser.
    parser = argparse.ArgumentParser(
        description="Run the DS 340W classroom technical demo on a smaller sample."
    )
    # Let the user control how many spatial cells are sampled.
    parser.add_argument(
        "--sample-cells",
        type=int,
        default=1000,
        help="Number of spatial cells to sample for the class demo.",
    )
    # Let the user control how many future months are forecasted.
    parser.add_argument(
        "--forecast-months",
        type=int,
        default=12,
        help="How many future months to forecast after fitting each model.",
    )
    # Let the user customize the output filename prefix.
    parser.add_argument(
        "--output-prefix",
        default="joint_technical_demo",
        help="Prefix used for the saved demo output files.",
    )
    # Return the parsed arguments.
    return parser.parse_args()


# Return values unchanged for models trained on the original scale.
def transform_none(values):
    # Return the original values.
    return values


# Return values unchanged after prediction for models trained on the original scale.
def inverse_none(values):
    # Return the original values.
    return values


# Apply a log1p transform for models trained in log space.
def transform_log1p(values):
    # Return the log-transformed values.
    return np.log1p(values)


# Undo the log1p transform after prediction.
def inverse_log1p(values):
    # Return the inverse-transformed values.
    return np.expm1(values)


# Extract the prebuilt cleaned-data ZIP when it is available.
def extract_cleaned_dataset_zip():
    # Loop through each likely cleaned-data ZIP path.
    for zip_path in CLEANED_DATA_ZIP_CANDIDATES:
        # Skip ZIP paths that do not exist.
        if not zip_path.exists():
            continue
        # Tell the user which cleaned-data ZIP is being extracted.
        print(f"Extracting cleaned dataset from: {zip_path}")
        # Open the ZIP archive.
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Extract the cleaned CSV into data/processed.
            zf.extractall(PROCESSED_DIR)
        # Return immediately after a successful extraction.
        return


# Make sure the class demo can find the prebuilt cleaned modeling table.
def ensure_cleaned_dataset_ready():
    # Skip the setup work when the cleaned CSV already exists.
    if DATA_FILE.exists():
        print("Found existing cleaned dataset:", DATA_FILE)
        return
    # Try extracting the prebuilt cleaned-data ZIP before failing.
    extract_cleaned_dataset_zip()
    # Stop with a clear message if the cleaned CSV is still missing.
    if not DATA_FILE.exists():
        raise SystemExit(
            "The class demo expects data/processed/na_pm25_cells_clean.csv "
            "or a ZIP copy of that file. Place the cleaned CSV or "
            "na_pm25_cells_clean.csv.zip in data/processed or data/raw and rerun. "
            "If you are starting from the original raw NetCDF files instead, "
            "run src/Codes/Extract_Monthly_nc_values.py separately first."
        )
    # Print the path after the ZIP extraction succeeds.
    print("Extracted cleaned dataset to:", DATA_FILE)


# Sample spatial cells from the large monthly panel.
def sample_cells_from_csv(data_file, sample_cell_count, random_seed):
    # Create a reproducible random generator.
    rng = np.random.default_rng(random_seed)
    # Start an empty set of unique cells.
    unique_cells = set()
    # Read the CSV in chunks so the script stays light on memory.
    for chunk in pd.read_csv(data_file, usecols=["lat", "lon"], chunksize=250_000):
        # Add each rounded cell coordinate pair to the unique set.
        unique_cells.update(zip(chunk["lat"].round(5), chunk["lon"].round(5)))
    # Convert the unique cells into a NumPy array.
    all_cells = np.array(sorted(unique_cells), dtype=np.float32)
    # Cap the requested sample size at the total number of cells available.
    sample_n = min(sample_cell_count, len(all_cells))
    # Draw a random sample of cell indices.
    sample_idx = rng.choice(len(all_cells), size=sample_n, replace=False)
    # Print the total number of available cells.
    print(f"Unique cells available: {len(all_cells):,}")
    # Print the number of sampled cells.
    print(f"Sampled cells for this demo: {sample_n:,}")
    # Return the sampled cell coordinate tuples as a set.
    return {tuple(cell) for cell in all_cells[sample_idx]}


# Load only the sampled cells from the large processed panel.
def load_sampled_frame(data_file, sampled_cells):
    # Start an empty list to store matching chunks.
    filtered_chunks = []
    # Read the CSV in chunks so the script is class-friendly on memory.
    for chunk in pd.read_csv(data_file, chunksize=250_000):
        # Build rounded coordinate pairs for the current chunk.
        cell_keys = list(zip(chunk["lat"].round(5), chunk["lon"].round(5)))
        # Mark rows whose cell belongs to the sampled set.
        keep_mask = [key in sampled_cells for key in cell_keys]
        # Keep only the sampled rows from this chunk.
        kept = chunk.loc[keep_mask]
        # Save the chunk if it contains at least one sampled row.
        if not kept.empty:
            filtered_chunks.append(kept)
    # Stop if no sampled rows were found at all.
    if not filtered_chunks:
        raise SystemExit("No sampled rows were loaded from the cleaned monthly dataset.")
    # Combine all sampled chunks into one DataFrame.
    frame = pd.concat(filtered_chunks, ignore_index=True)
    # Convert the date column to datetimes.
    frame["date"] = pd.to_datetime(frame["date"])
    # Sort the sampled panel by cell and time.
    frame = frame.sort_values(["lat", "lon", "date"]).reset_index(drop=True)
    # Cast the key numeric columns to float32.
    frame[["lat", "lon", TARGET]] = frame[["lat", "lon", TARGET]].astype(np.float32)
    # Print the number of sampled rows loaded.
    print(f"Loaded sampled rows: {len(frame):,}")
    # Return the sampled panel.
    return frame


# Build the sampled feature table used by every demo model.
def prepare_demo_frame(sample_cell_count):
    # Sample a subset of cells from the big monthly CSV.
    sampled_cells = sample_cells_from_csv(DATA_FILE, sample_cell_count, RANDOM_SEED)
    # Load only those sampled rows.
    frame = load_sampled_frame(DATA_FILE, sampled_cells)
    # Print a feature-engineering header.
    print("\n--- FEATURE ENGINEERING ---")
    # Add PM2.5 lag and rolling-history features.
    frame = add_history_features(frame, target=TARGET)
    # Add train-only climatology features.
    frame = add_train_only_climatology(frame, train_end=TRAIN_END, target=TARGET)
    # Add regional anomaly features.
    frame = add_experimental_features(frame)
    # Add a no-ERA placeholder pass so the shared forecast code stays consistent.
    frame, _ = add_era5_features(
        frame,
        raw_dir=RAW_DIR,
        train_end=TRAIN_END,
        use_era5=False,
        feature_level="core",
    )
    # Drop rows that still contain missing values after feature construction.
    frame = frame.dropna().copy()
    # Print the number of rows left after feature engineering.
    print(f"Rows after feature engineering: {len(frame):,}")
    # Return the final sampled feature table.
    return frame


# Split the sampled table into train, validation, and test segments.
def split_frame(frame):
    # Keep rows before 2021 for training.
    train = frame.loc[frame["date"] < TRAIN_END].copy()
    # Keep rows from 2021 for validation.
    val = frame.loc[(frame["date"] >= TRAIN_END) & (frame["date"] < VAL_END)].copy()
    # Keep rows from 2022 onward for testing.
    test = frame.loc[frame["date"] >= VAL_END].copy()
    # Stop if the sample is too small to populate all splits.
    if train.empty or val.empty or test.empty:
        raise SystemExit("The sampled dataset is too small for the train/validation/test split.")
    # Print a split-size header.
    print("\n--- SPLIT SIZES ---")
    # Print the training-row count.
    print(f"Train rows: {len(train):,}")
    # Print the validation-row count.
    print(f"Validation rows: {len(val):,}")
    # Print the test-row count.
    print(f"Test rows: {len(test):,}")
    # Return the three splits.
    return train, val, test


# Define the feature sets used in the classroom demo.
def build_joint_feature_sets():
    # Build the project’s normal tree-model feature-set dictionary without ERA features.
    base_tree_sets = build_feature_sets([])
    # Return the feature list for each demo model.
    return {
        "Ridge": [
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
        ],
        "CatBoost": base_tree_sets["base"],
        "XGBoost": base_tree_sets["trend_region"],
        "LightGBM": base_tree_sets["trend_region"],
    }


# Fit the Ridge regression demo model.
def fit_ridge(train, val, test, features):
    # Build the Ridge training matrix.
    X_train = train[features]
    # Build the Ridge validation matrix.
    X_val = val[features]
    # Build the Ridge test matrix.
    X_test = test[features]
    # Collect the Ridge training target.
    y_train = train[TARGET].to_numpy()
    # Collect the Ridge validation target.
    y_val = val[TARGET].to_numpy()
    # Collect the Ridge test target.
    y_test = test[TARGET].to_numpy()
    # Build the lag-1 validation baseline.
    naive_val = val["pm25_lag1"].to_numpy(dtype=float)
    # Build the lag-1 test baseline.
    naive_test = test["pm25_lag1"].to_numpy(dtype=float)
    # Print a Ridge training header.
    print("\n--- TRAINING RIDGE ---")
    # Build the Ridge pipeline.
    model = make_pipeline(StandardScaler(), Ridge(alpha=20.0, random_state=RANDOM_SEED))
    # Fit the Ridge pipeline.
    model.fit(X_train, y_train)
    # Record the minimum training target value for clipping.
    pred_floor = float(np.min(y_train))
    # Record the maximum training target value for clipping.
    pred_ceiling = float(np.max(y_train))
    # Predict the validation rows.
    val_pred = np.clip(model.predict(X_val), pred_floor, pred_ceiling)
    # Predict the test rows.
    test_pred = np.clip(model.predict(X_test), pred_floor, pred_ceiling)
    # Compute the difference between Ridge and the naive baseline on validation.
    diff = val_pred - naive_val
    # Compute the blend denominator.
    denom = float(np.dot(diff, diff))
    # Start with a zero blend weight.
    blend_weight = 0.0
    # Learn a validation-based blend weight when the denominator is usable.
    if denom > 0:
        # Compute the blend numerator.
        numer = float(np.dot(diff, y_val - naive_val))
        # Clip the blend weight to the interval [0, 1].
        blend_weight = float(np.clip(numer / denom, 0.0, 1.0))
    # Blend Ridge with the naive baseline on validation.
    val_pred = blend_weight * val_pred + (1.0 - blend_weight) * naive_val
    # Blend Ridge with the naive baseline on test.
    test_pred = blend_weight * test_pred + (1.0 - blend_weight) * naive_test
    # Return the Ridge results dictionary.
    return {
        "model_name": "Ridge",
        "feature_set": "linear_core",
        "features": features,
        "model": model,
        "val_metrics": compute_metrics(y_val, val_pred),
        "test_metrics": compute_metrics(y_test, test_pred),
        "test_pred": test_pred,
    }


# Fit the CatBoost demo model.
def fit_catboost(train, val, test, features):
    # Mark the categorical columns used by CatBoost.
    categorical_cols = [col for col in ["month", "region_lat_bin", "region_lon_bin"] if col in features]
    # Build the CatBoost training matrix.
    X_train = train[features]
    # Build the CatBoost validation matrix.
    X_val = val[features]
    # Build the CatBoost test matrix.
    X_test = test[features]
    # Collect the log-transformed CatBoost training target.
    y_train = transform_log1p(train[TARGET].to_numpy())
    # Collect the log-transformed CatBoost validation target.
    y_val_train_scale = transform_log1p(val[TARGET].to_numpy())
    # Keep the raw validation target for metric reporting.
    y_val_raw = val[TARGET].to_numpy()
    # Keep the raw test target for metric reporting.
    y_test = test[TARGET].to_numpy()
    # Print a CatBoost training header.
    print("\n--- TRAINING CATBOOST ---")
    # Create the CatBoost regressor.
    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=220,
        learning_rate=0.06,
        depth=6,
        l2_leaf_reg=28,
        random_strength=2.0,
        min_data_in_leaf=80,
        rsm=0.75,
        bagging_temperature=1.0,
        border_count=128,
        random_seed=RANDOM_SEED,
        verbose=False,
        use_best_model=True,
        early_stopping_rounds=25,
        allow_writing_files=False,
        thread_count=-1,
    )
    # Fit the CatBoost model.
    model.fit(X_train, y_train, eval_set=(X_val, y_val_train_scale), cat_features=categorical_cols)
    # Predict the validation rows on the original scale.
    val_pred = inverse_log1p(model.predict(X_val))
    # Predict the test rows on the original scale.
    test_pred = inverse_log1p(model.predict(X_test))
    # Return the CatBoost results dictionary.
    return {
        "model_name": "CatBoost",
        "feature_set": "base",
        "features": features,
        "model": model,
        "best_iteration": model.get_best_iteration(),
        "val_metrics": compute_metrics(y_val_raw, val_pred),
        "test_metrics": compute_metrics(y_test, test_pred),
        "test_pred": test_pred,
    }


# Fit the XGBoost demo model.
def fit_xgboost(train, val, test, features):
    # Build the XGBoost training matrix.
    X_train = train[features]
    # Build the XGBoost validation matrix.
    X_val = val[features]
    # Build the XGBoost test matrix.
    X_test = test[features]
    # Collect the log-transformed XGBoost training target.
    y_train = transform_log1p(train[TARGET].to_numpy())
    # Collect the log-transformed XGBoost validation target.
    y_val_train_scale = transform_log1p(val[TARGET].to_numpy())
    # Keep the raw validation target for metric reporting.
    y_val_raw = val[TARGET].to_numpy()
    # Keep the raw test target for metric reporting.
    y_test = test[TARGET].to_numpy()
    # Print an XGBoost training header.
    print("\n--- TRAINING XGBOOST ---")
    # Create the XGBoost regressor.
    model = XGBRegressor(
        n_estimators=240,
        learning_rate=0.06,
        max_depth=6,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=4.0,
        gamma=0.02,
        tree_method="hist",
        objective="reg:squarederror",
        eval_metric="rmse",
        early_stopping_rounds=25,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    # Fit the XGBoost model.
    model.fit(X_train, y_train, eval_set=[(X_val, y_val_train_scale)], verbose=False)
    # Determine the best XGBoost iteration count.
    best_iteration = model.best_iteration + 1 if model.best_iteration is not None else 240
    # Predict the validation rows on the original scale.
    val_pred = inverse_log1p(model.predict(X_val, iteration_range=(0, best_iteration)))
    # Predict the test rows on the original scale.
    test_pred = inverse_log1p(model.predict(X_test, iteration_range=(0, best_iteration)))
    # Return the XGBoost results dictionary.
    return {
        "model_name": "XGBoost",
        "feature_set": "trend_region",
        "features": features,
        "model": model,
        "best_iteration": best_iteration,
        "val_metrics": compute_metrics(y_val_raw, val_pred),
        "test_metrics": compute_metrics(y_test, test_pred),
        "test_pred": test_pred,
    }


# Cast LightGBM categorical columns before fitting or forecasting.
def prepare_lightgbm_frame(frame, feature_list):
    # Copy the requested feature columns.
    prepared = frame.loc[:, feature_list].copy()
    # Convert the LightGBM categorical columns to pandas category dtype when present.
    for col in ["month", "region_lat_bin", "region_lon_bin"]:
        if col in prepared.columns:
            prepared[col] = prepared[col].astype("category")
    # Return the prepared LightGBM frame.
    return prepared


# Fit the LightGBM demo model.
def fit_lightgbm(train, val, test, features):
    # Build the LightGBM training matrix.
    X_train = prepare_lightgbm_frame(train, features)
    # Build the LightGBM validation matrix.
    X_val = prepare_lightgbm_frame(val, features)
    # Build the LightGBM test matrix.
    X_test = prepare_lightgbm_frame(test, features)
    # Collect the log-transformed LightGBM training target.
    y_train = transform_log1p(train[TARGET].to_numpy())
    # Collect the log-transformed LightGBM validation target.
    y_val_train_scale = transform_log1p(val[TARGET].to_numpy())
    # Keep the raw validation target for metric reporting.
    y_val_raw = val[TARGET].to_numpy()
    # Keep the raw test target for metric reporting.
    y_test = test[TARGET].to_numpy()
    # Mark the LightGBM categorical columns that are present.
    categorical_cols = [col for col in ["month", "region_lat_bin", "region_lon_bin"] if col in features]
    # Print a LightGBM training header.
    print("\n--- TRAINING LIGHTGBM ---")
    # Create the LightGBM regressor.
    model = LGBMRegressor(
        n_estimators=240,
        learning_rate=0.06,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=4.0,
        min_split_gain=0.02,
        max_bin=127,
        force_col_wise=True,
        random_state=RANDOM_SEED,
        objective="regression",
        n_jobs=-1,
        verbosity=-1,
    )
    # Fit the LightGBM model.
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val_train_scale)],
        eval_metric="rmse",
        categorical_feature=categorical_cols,
        callbacks=[
            early_stopping(stopping_rounds=25, first_metric_only=True, verbose=False),
            log_evaluation(period=0),
        ],
    )
    # Determine the best LightGBM iteration count.
    best_iteration = model.best_iteration_ or 240
    # Predict the validation rows on the original scale.
    val_pred = inverse_log1p(model.predict(X_val, num_iteration=best_iteration))
    # Predict the test rows on the original scale.
    test_pred = inverse_log1p(model.predict(X_test, num_iteration=best_iteration))
    # Return the LightGBM results dictionary.
    return {
        "model_name": "LightGBM",
        "feature_set": "trend_region",
        "features": features,
        "model": model,
        "best_iteration": best_iteration,
        "val_metrics": compute_metrics(y_val_raw, val_pred),
        "test_metrics": compute_metrics(y_test, test_pred),
        "test_pred": test_pred,
    }


# Combine train, validation, and test rows into one observed-history table.
def build_observed_history(train, val, test):
    # Return one table containing every observed month used before forecasting.
    return pd.concat([train, val, test], ignore_index=True)


# Refit Ridge on all observed history before forecasting.
def refit_ridge_for_forecast(observed_history, features):
    # Build the full Ridge feature matrix.
    X_full = observed_history[features]
    # Build the full Ridge target vector on the original scale.
    y_full = observed_history[TARGET].to_numpy()
    # Print a Ridge refit header.
    print("\n--- RETRAINING RIDGE FOR FORECAST ---")
    # Create the Ridge refit pipeline.
    forecast_model = make_pipeline(StandardScaler(), Ridge(alpha=20.0, random_state=RANDOM_SEED))
    # Fit the Ridge refit pipeline.
    forecast_model.fit(X_full, y_full)
    # Return the refit Ridge model.
    return forecast_model


# Refit CatBoost on all observed history before forecasting.
def refit_catboost_for_forecast(observed_history, features, best_iteration):
    # Build the full CatBoost feature matrix.
    X_full = observed_history[features]
    # Build the full CatBoost target vector in log space.
    y_full = transform_log1p(observed_history[TARGET].to_numpy())
    # Mark the CatBoost categorical columns.
    categorical_cols = [col for col in ["month", "region_lat_bin", "region_lon_bin"] if col in features]
    # Print a CatBoost refit header.
    print("\n--- RETRAINING CATBOOST FOR FORECAST ---")
    # Create the CatBoost refit model.
    forecast_model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=max(1, int(best_iteration)),
        learning_rate=0.06,
        depth=6,
        l2_leaf_reg=28,
        random_strength=2.0,
        min_data_in_leaf=80,
        rsm=0.75,
        bagging_temperature=1.0,
        border_count=128,
        random_seed=RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )
    # Fit the CatBoost refit model.
    forecast_model.fit(X_full, y_full, cat_features=categorical_cols)
    # Return the refit CatBoost model.
    return forecast_model


# Refit XGBoost on all observed history before forecasting.
def refit_xgboost_for_forecast(observed_history, features, best_iteration):
    # Build the full XGBoost feature matrix.
    X_full = observed_history[features]
    # Build the full XGBoost target vector in log space.
    y_full = transform_log1p(observed_history[TARGET].to_numpy())
    # Print an XGBoost refit header.
    print("\n--- RETRAINING XGBOOST FOR FORECAST ---")
    # Create the XGBoost refit model.
    forecast_model = XGBRegressor(
        n_estimators=max(1, int(best_iteration)),
        learning_rate=0.06,
        max_depth=6,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=4.0,
        gamma=0.02,
        tree_method="hist",
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    # Fit the XGBoost refit model.
    forecast_model.fit(X_full, y_full, verbose=False)
    # Return the refit XGBoost model.
    return forecast_model


# Refit LightGBM on all observed history before forecasting.
def refit_lightgbm_for_forecast(observed_history, features, best_iteration):
    # Build the full LightGBM feature matrix.
    X_full = prepare_lightgbm_frame(observed_history, features)
    # Build the full LightGBM target vector in log space.
    y_full = transform_log1p(observed_history[TARGET].to_numpy())
    # Mark the LightGBM categorical columns that are present.
    categorical_cols = [col for col in ["month", "region_lat_bin", "region_lon_bin"] if col in features]
    # Print a LightGBM refit header.
    print("\n--- RETRAINING LIGHTGBM FOR FORECAST ---")
    # Create the LightGBM refit model.
    forecast_model = LGBMRegressor(
        n_estimators=max(1, int(best_iteration)),
        learning_rate=0.06,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=4.0,
        min_split_gain=0.02,
        max_bin=127,
        force_col_wise=True,
        random_state=RANDOM_SEED,
        objective="regression",
        n_jobs=-1,
        verbosity=-1,
    )
    # Fit the LightGBM refit model.
    forecast_model.fit(X_full, y_full, categorical_feature=categorical_cols)
    # Return the refit LightGBM model.
    return forecast_model


# Save one combined evaluation-metrics table for the classroom demo.
def save_metrics_table(results, train, val, test, output_path, sample_cells):
    # Start an empty list of output rows.
    rows = []
    # Loop through each fitted model result.
    for result in results:
        # Build the base summary row for this model.
        row = {
            "model": result["model_name"],
            "feature_set": result["feature_set"],
            "sample_cells": sample_cells,
            "train_rows": len(train),
            "val_rows": len(val),
            "test_rows": len(test),
        }
        # Add the validation metrics to this row.
        row.update({f"val_{key.lower()}": value for key, value in result["val_metrics"].items()})
        # Add the test metrics to this row.
        row.update({f"test_{key.lower()}": value for key, value in result["test_metrics"].items()})
        # Append the row to the list.
        rows.append(row)
    # Convert the metric rows into a DataFrame and sort by test RMSE.
    metrics_df = pd.DataFrame(rows).sort_values("test_rmse").reset_index(drop=True)
    # Save the combined metrics file.
    metrics_df.to_csv(output_path, index=False)
    # Print the metrics-file path.
    print("\nSaved combined evaluation metrics to:", output_path)
    # Print a short leaderboard for the demo.
    print(metrics_df.loc[:, ["model", "test_rmse", "test_mae", "test_r2"]].to_string(index=False))


# Merge all forecast CSVs into one side-by-side comparison table.
def merge_forecasts(forecast_paths, output_path):
    # Start with no merged table.
    merged = None
    # Loop through each saved forecast file.
    for model_name, path in forecast_paths.items():
        # Load the current model’s forecast file.
        df = pd.read_csv(path)
        # Rename the prediction column to include the model name.
        df = df.rename(columns={TARGET: f"{model_name.lower()}_forecast_pm25"})
        # Initialize the merged table with the first forecast file.
        if merged is None:
            merged = df
        # Merge later forecast files onto the existing merged table.
        else:
            merged = merged.merge(df, on=["lat", "lon", "date"], how="inner")
    # Save the combined forecast comparison table.
    merged.to_csv(output_path, index=False)
    # Print the merged-forecast path.
    print("Saved combined forecast table to:", output_path)


# Run the full classroom demo workflow.
def main():
    # Parse the command-line arguments.
    args = parse_args()
    # Make sure the raw-data folder exists in case the user also stores the cleaned-data ZIP there.
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    # Make sure the processed-data folder exists.
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    # Make sure the classroom-demo output folder exists.
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    # Print a script header.
    print("\n=== DS 340W JOINT TECHNICAL DEMO ===")
    # Print the project root folder.
    print("Project root:", BASE_DIR)
    # Print the cleaned-data CSV path.
    print("Expected cleaned dataset:", DATA_FILE)
    # Print the demo output folder.
    print("Processed output folder:", DEMO_DIR)
    # Make sure the prebuilt cleaned modeling table is available for the demo.
    ensure_cleaned_dataset_ready()
    # Build the shared sampled feature table.
    frame = prepare_demo_frame(args.sample_cells)
    # Split the sampled feature table into train, validation, and test data.
    train, val, test = split_frame(frame)
    # Build the model-specific feature lists used in the classroom demo.
    feature_sets = build_joint_feature_sets()
    # Print a feature-count header.
    print("\n--- FEATURE COUNTS ---")
    # Print the number of features used by each model.
    for model_name, feature_list in feature_sets.items():
        print(f"{model_name}: {len(feature_list)} features")
    # Fit the Ridge demo model.
    ridge_result = fit_ridge(train, val, test, feature_sets["Ridge"])
    # Fit the CatBoost demo model.
    catboost_result = fit_catboost(train, val, test, feature_sets["CatBoost"])
    # Fit the XGBoost demo model.
    xgb_result = fit_xgboost(train, val, test, feature_sets["XGBoost"])
    # Fit the LightGBM demo model.
    lightgbm_result = fit_lightgbm(train, val, test, feature_sets["LightGBM"])
    # Collect all fitted-model results in one list.
    results = [ridge_result, catboost_result, xgb_result, lightgbm_result]
    # Print an evaluation-metrics header.
    print("\n--- EVALUATION METRICS ---")
    # Print the validation and test metrics for each model.
    for result in results:
        print_metrics(f"{result['model_name']} validation metrics", result["val_metrics"])
        print_metrics(f"{result['model_name']} test metrics", result["test_metrics"])
    # Build the combined metrics output path.
    metrics_path = DEMO_DIR / f"{args.output_prefix}_eval_metrics.csv"
    # Save the combined metrics file.
    save_metrics_table(results, train, val, test, metrics_path, args.sample_cells)
    # Combine the observed history for recursive forecasting.
    observed_history = build_observed_history(train, val, test)
    # Build the XGBoost fill values used during forecasting.
    xgb_fill_values = observed_history[feature_sets["XGBoost"]].median(numeric_only=True).fillna(0.0)
    # Build the Ridge fill values used during forecasting.
    ridge_fill_values = observed_history[feature_sets["Ridge"]].median(numeric_only=True).fillna(0.0)
    # Build the CatBoost fill values used during forecasting.
    catboost_fill_values = observed_history[feature_sets["CatBoost"]].median(numeric_only=True).fillna(0.0)
    # Build the LightGBM fill values used during forecasting.
    lightgbm_fill_values = observed_history[feature_sets["LightGBM"]].median(numeric_only=True).fillna(0.0)
    # Refit Ridge on all observed history for forecasting.
    forecast_ridge_model = refit_ridge_for_forecast(observed_history, feature_sets["Ridge"])
    # Refit CatBoost on all observed history for forecasting.
    forecast_cat_model = refit_catboost_for_forecast(
        observed_history,
        feature_sets["CatBoost"],
        catboost_result["best_iteration"],
    )
    # Refit XGBoost on all observed history for forecasting.
    forecast_xgb_model = refit_xgboost_for_forecast(
        observed_history,
        feature_sets["XGBoost"],
        xgb_result["best_iteration"],
    )
    # Refit LightGBM on all observed history for forecasting.
    forecast_lightgbm_model = refit_lightgbm_for_forecast(
        observed_history,
        feature_sets["LightGBM"],
        lightgbm_result["best_iteration"],
    )
    # Define the per-model forecast output files.
    forecast_paths = {
        "Ridge": DEMO_DIR / f"{args.output_prefix}_ridge_forecast.csv",
        "CatBoost": DEMO_DIR / f"{args.output_prefix}_catboost_forecast.csv",
        "XGBoost": DEMO_DIR / f"{args.output_prefix}_xgboost_forecast.csv",
        "LightGBM": DEMO_DIR / f"{args.output_prefix}_lightgbm_forecast.csv",
    }
    # Run the Ridge recursive forecast.
    run_recursive_forecast(
        model=forecast_ridge_model,
        history_frame=observed_history,
        train_frame=observed_history,
        active_features=feature_sets["Ridge"],
        feature_set_name="linear_core",
        output_file=forecast_paths["Ridge"],
        target=TARGET,
        train_end=TRAIN_END,
        raw_dir=RAW_DIR,
        build_feature_sets_fn=lambda _: {"linear_core": feature_sets["Ridge"]},
        add_history_features_fn=add_history_features,
        add_train_only_climatology_fn=add_train_only_climatology,
        add_experimental_features_fn=add_experimental_features,
        add_era5_features_fn=add_era5_features,
        predict_fn=lambda fitted_model, X_future: fitted_model.predict(X_future),
        inverse_transform_fn=inverse_none,
        fill_values=ridge_fill_values,
        forecast_months=args.forecast_months,
        era5_feature_level="core",
    )
    # Run the CatBoost recursive forecast.
    run_recursive_forecast(
        model=forecast_cat_model,
        history_frame=observed_history,
        train_frame=observed_history,
        active_features=feature_sets["CatBoost"],
        feature_set_name="base",
        output_file=forecast_paths["CatBoost"],
        target=TARGET,
        train_end=TRAIN_END,
        raw_dir=RAW_DIR,
        build_feature_sets_fn=lambda _: {"base": feature_sets["CatBoost"]},
        add_history_features_fn=add_history_features,
        add_train_only_climatology_fn=add_train_only_climatology,
        add_experimental_features_fn=add_experimental_features,
        add_era5_features_fn=add_era5_features,
        predict_fn=lambda fitted_model, X_future: fitted_model.predict(X_future),
        inverse_transform_fn=inverse_log1p,
        fill_values=catboost_fill_values,
        forecast_months=args.forecast_months,
        era5_feature_level="core",
        prepare_model_frame_fn=lambda frame, feature_list: frame.loc[:, feature_list],
    )
    # Run the XGBoost recursive forecast.
    run_recursive_forecast(
        model=forecast_xgb_model,
        history_frame=observed_history,
        train_frame=observed_history,
        active_features=feature_sets["XGBoost"],
        feature_set_name="trend_region",
        output_file=forecast_paths["XGBoost"],
        target=TARGET,
        train_end=TRAIN_END,
        raw_dir=RAW_DIR,
        build_feature_sets_fn=lambda _: {"trend_region": feature_sets["XGBoost"]},
        add_history_features_fn=add_history_features,
        add_train_only_climatology_fn=add_train_only_climatology,
        add_experimental_features_fn=add_experimental_features,
        add_era5_features_fn=add_era5_features,
        predict_fn=lambda fitted_model, X_future: fitted_model.predict(X_future),
        inverse_transform_fn=inverse_log1p,
        fill_values=xgb_fill_values,
        forecast_months=args.forecast_months,
        era5_feature_level="core",
    )
    # Run the LightGBM recursive forecast.
    run_recursive_forecast(
        model=forecast_lightgbm_model,
        history_frame=observed_history,
        train_frame=observed_history,
        active_features=feature_sets["LightGBM"],
        feature_set_name="trend_region",
        output_file=forecast_paths["LightGBM"],
        target=TARGET,
        train_end=TRAIN_END,
        raw_dir=RAW_DIR,
        build_feature_sets_fn=lambda _: {"trend_region": feature_sets["LightGBM"]},
        add_history_features_fn=add_history_features,
        add_train_only_climatology_fn=add_train_only_climatology,
        add_experimental_features_fn=add_experimental_features,
        add_era5_features_fn=add_era5_features,
        predict_fn=lambda fitted_model, X_future: fitted_model.predict(X_future),
        inverse_transform_fn=inverse_log1p,
        fill_values=lightgbm_fill_values,
        forecast_months=args.forecast_months,
        era5_feature_level="core",
        prepare_model_frame_fn=prepare_lightgbm_frame,
    )
    # Build the combined-forecast output path.
    combined_forecast_path = DEMO_DIR / f"{args.output_prefix}_combined_forecasts.csv"
    # Save one merged forecast-comparison file.
    merge_forecasts(forecast_paths, combined_forecast_path)
    # Print a completion message.
    print("\nTechnical demo run complete.")
    # Print the main metrics-file path.
    print("Main metrics file:", metrics_path)
    # Print the main combined-forecast path.
    print("Main forecast file:", combined_forecast_path)


# Run the script when this file is executed directly.
if __name__ == "__main__":
    # Call the main classroom-demo workflow.
    main()
