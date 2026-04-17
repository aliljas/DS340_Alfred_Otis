# Import argparse so the script can accept command-line options.
import argparse
# Import os so we can manage environment variables for child processes.
import os
# Import subprocess so this launcher can call the real model scripts.
import subprocess
# Import sys so we can reuse the current Python interpreter path.
import sys
# Import shutil so we can copy a direct CSV into the expected folder if needed.
import shutil
# Import tempfile so we can create a writable Matplotlib cache path.
import tempfile
# Import Path so every filesystem path stays portable across machines.
from pathlib import Path

# Import NumPy for random sampling of spatial cells.
import numpy as np
# Import Pandas for reading, filtering, and saving CSV files.
import pandas as pd


# Build a writable Matplotlib cache directory inside the system temp folder.
MPLCONFIGDIR = Path(tempfile.gettempdir()) / "matplotlib-cache"
# Create that cache directory if it does not already exist.
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
# Tell plotting libraries to use that writable cache location.
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))


# Resolve the project root from the location of this script.
BASE_DIR = Path(__file__).resolve().parents[2]
# Point to the raw-data folder in the project.
RAW_DIR = BASE_DIR / "data/raw"
# Point to the processed-data folder in the project.
PROCESSED_DIR = BASE_DIR / "data/processed"
# Store all classroom-demo artifacts inside a dedicated technical-demo folder.
DEMO_DIR = PROCESSED_DIR / "technical_demo"
# Point to the cleaned monthly modeling table used throughout the project.
DATA_FILE = PROCESSED_DIR / "na_pm25_cells_clean.csv"

# Keep the same training cutoff used by the main model scripts.
TRAIN_END = pd.Timestamp("2021-01-01")
# Keep the same validation cutoff used by the main model scripts.
VAL_END = pd.Timestamp("2022-01-01")
# Use one fixed random seed so the sampled classroom demo is reproducible.
RANDOM_SEED = 42

# List the direct-CSV locations the class computer might receive.
CLEANED_DATA_CSV_CANDIDATES = [
    # First, check whether the CSV was placed in the raw-data folder.
    RAW_DIR / "na_pm25_cells_clean.csv",
    # Next, check for a copy placed at the project root.
    BASE_DIR / "na_pm25_cells_clean.csv",
]

# Define the real project model scripts the launcher should run.
MODEL_RUNS = [
    {
        # Give the first run a human-readable model name.
        "name": "Ridge",
        # Point to the actual Ridge model script in the project.
        "script": BASE_DIR / "src/Codes/LR_model.py",
        # Set environment overrides so the demo stays fast and focused.
        "env": {
            # Disable the Ridge ERA5 comparison for the classroom demo.
            "LR_COMPARE_ERA5": "0",
            # Use the strict no-ERA path so forecasting remains causal.
            "LR_USE_ERA5": "0",
            # Skip the optional main summary plot during the demo run.
            "LR_SAVE_PLOT": "0",
            # Skip the optional ERA5 comparison plot during the demo run.
            "LR_SAVE_COMPARISON_PLOT": "0",
            # Keep forecasting enabled so the real script produces a 2023 forecast.
            "LR_RUN_FORECAST": "1",
        },
        # Expect the Ridge metrics file to use the standard project filename.
        "metrics_file": "lr_eval_metrics.csv",
        # Expect the Ridge forecast file to use the standard project filename.
        "forecast_file": "lr_predictions_2023.csv",
    },
    {
        # Give the second run a human-readable model name.
        "name": "CatBoost",
        # Point to the actual CatBoost model script in the project.
        "script": BASE_DIR / "src/Codes/CatBoost_model.py",
        # Set environment overrides so the demo stays fast and focused.
        "env": {
            # Disable the CatBoost ERA5 comparison for the classroom demo.
            "CATBOOST_COMPARE_ERA5": "0",
            # Use the strict no-ERA path so forecasting remains causal.
            "CATBOOST_USE_ERA5": "0",
            # Skip the optional main summary plot during the demo run.
            "CATBOOST_SAVE_PLOT": "0",
            # Skip the optional ERA5 comparison plot during the demo run.
            "CATBOOST_SAVE_COMPARISON_PLOT": "0",
            # Keep forecasting enabled so the real script produces a 2023 forecast.
            "CATBOOST_RUN_FORECAST": "1",
        },
        # Expect the CatBoost metrics file to use the standard project filename.
        "metrics_file": "catboost_eval_metrics.csv",
        # Expect the CatBoost forecast file to use the standard project filename.
        "forecast_file": "catboost_predictions_2023.csv",
    },
    {
        # Give the third run a human-readable model name.
        "name": "XGBoost",
        # Point to the actual XGBoost model script in the project.
        "script": BASE_DIR / "src/Codes/XGB_model.py",
        # Set environment overrides so the demo stays fast and focused.
        "env": {
            # Disable the XGBoost ERA5 comparison for the classroom demo.
            "XGB_COMPARE_ERA5": "0",
            # Use the strict no-ERA path so forecasting remains causal.
            "XGB_USE_ERA5": "0",
            # Skip the optional main summary plot during the demo run.
            "XGB_SAVE_PLOT": "0",
            # Skip the optional ERA5 comparison plot during the demo run.
            "XGB_SAVE_COMPARISON_PLOT": "0",
            # Keep forecasting enabled so the real script produces a 2023 forecast.
            "XGB_RUN_FORECAST": "1",
            # Skip any slower Optuna-style tuning path in sampled demo mode.
            "XGB_SKIP_OPTUNA": "1",
        },
        # Expect the XGBoost metrics file to use the standard project filename.
        "metrics_file": "xgb_eval_metrics.csv",
        # Expect the XGBoost forecast file to use the standard project filename.
        "forecast_file": "xgb_predictions_2023.csv",
    },
    {
        # Give the fourth run a human-readable model name.
        "name": "LightGBM",
        # Point to the actual LightGBM model script in the project.
        "script": BASE_DIR / "src/Codes/LightGBM_model.py",
        # Set environment overrides so the demo stays fast and focused.
        "env": {
            # Disable the LightGBM ERA5 comparison for the classroom demo.
            "LIGHTGBM_COMPARE_ERA5": "0",
            # Use the strict no-ERA path so forecasting remains causal.
            "LIGHTGBM_USE_ERA5": "0",
            # Skip the optional main summary plot during the demo run.
            "LIGHTGBM_SAVE_PLOT": "0",
            # Skip the optional ERA5 comparison plot during the demo run.
            "LIGHTGBM_SAVE_COMPARISON_PLOT": "0",
            # Keep forecasting enabled so the real script produces a 2023 forecast.
            "LIGHTGBM_RUN_FORECAST": "1",
        },
        # Expect the LightGBM metrics file to use the standard project filename.
        "metrics_file": "lightgbm_eval_metrics.csv",
        # Expect the LightGBM forecast file to use the standard project filename.
        "forecast_file": "lightgbm_predictions_2023.csv",
    },
]


# Parse the command-line arguments for the demo launcher.
def parse_args():
    # Create the argument parser with a short description.
    parser = argparse.ArgumentParser(
        description="Run the DS 340W classroom demo by calling the real model scripts on sampled data."
    )
    # Add an option that controls how many spatial cells are sampled.
    parser.add_argument(
        "--sample-cells",
        # Store the option as an integer.
        type=int,
        # Default to 1000 sampled cells for a manageable classroom run.
        default=1000,
        # Explain the meaning of the option in the help text.
        help="Number of spatial cells to sample from the cleaned monthly dataset.",
    )
    # Add an option that controls the forecast horizon for each model.
    parser.add_argument(
        "--forecast-months",
        # Store the option as an integer.
        type=int,
        # Default to a 12-month forecast horizon.
        default=12,
        # Explain the meaning of the option in the help text.
        help="How many future months each model should forecast.",
    )
    # Add an option that controls the filename prefix for demo outputs.
    parser.add_argument(
        "--output-prefix",
        # Use a descriptive default prefix for all generated demo files.
        default="joint_technical_demo",
        # Explain the meaning of the option in the help text.
        help="Prefix for saved combined demo outputs.",
    )
    # Return the parsed arguments to the caller.
    return parser.parse_args()


# Try to copy a direct cleaned CSV into data/processed.
def stage_cleaned_dataset_csv():
    # Loop through each CSV location we are willing to check.
    for csv_path in CLEANED_DATA_CSV_CANDIDATES:
        # Skip this CSV candidate if it does not exist.
        if not csv_path.exists():
            continue
        # Tell the user which direct CSV file is being copied into place.
        print(f"Copying cleaned dataset from: {csv_path}")
        # Copy the CSV into the standard processed-data location.
        shutil.copy2(csv_path, DATA_FILE)
        # Stop immediately after the first successful copy.
        return


# Make sure the cleaned monthly dataset is available before the demo starts.
def ensure_cleaned_dataset_ready():
    # If the cleaned CSV already exists, nothing else is needed.
    if DATA_FILE.exists():
        # Tell the user we found the cleaned dataset already in place.
        print("Found existing cleaned dataset:", DATA_FILE)
        # Return early because setup is complete.
        return

    # Otherwise, try copying a direct CSV into the processed-data folder.
    stage_cleaned_dataset_csv()

    # If the cleaned CSV is still missing, stop with a clear setup message.
    if not DATA_FILE.exists():
        # Raise a controlled exit so the user knows exactly what file is missing.
        raise SystemExit(
            "The class demo expects data/processed/na_pm25_cells_clean.csv. "
            "Place the direct CSV in data/processed, data/raw, or the project root and rerun."
        )

    # Tell the user where the cleaned CSV ended up after staging.
    print("Staged cleaned dataset to:", DATA_FILE)


# Sample a subset of unique latitude/longitude cells from the large cleaned CSV.
def sample_cells_from_csv(data_file, sample_cell_count, random_seed):
    # Create a reproducible random number generator.
    rng = np.random.default_rng(random_seed)
    # Start with an empty set so each spatial cell is stored only once.
    unique_cells = set()

    # Read only the latitude and longitude columns in chunks to stay memory-safe.
    for chunk in pd.read_csv(data_file, usecols=["lat", "lon"], chunksize=250_000):
        # Add the rounded latitude/longitude pairs from this chunk into the set.
        unique_cells.update(zip(chunk["lat"].round(5), chunk["lon"].round(5)))

    # Convert the final set of unique cells into a NumPy array.
    all_cells = np.array(sorted(unique_cells), dtype=np.float32)
    # Cap the requested sample size at the total number of cells available.
    sample_n = min(sample_cell_count, len(all_cells))
    # Randomly select the cell indices we will keep for the demo.
    sample_idx = rng.choice(len(all_cells), size=sample_n, replace=False)

    # Print how many unique cells were available in the full cleaned CSV.
    print(f"Unique cells available: {len(all_cells):,}")
    # Print how many of those cells will be used in the demo sample.
    print(f"Sampled cells for this demo: {sample_n:,}")
    # Return the sampled spatial cells as a Python set of coordinate tuples.
    return {tuple(cell) for cell in all_cells[sample_idx]}


# Write a sampled version of the cleaned monthly dataset to a demo CSV file.
def write_sampled_dataset(data_file, sampled_cells, output_file):
    # Start an empty list to collect the sampled chunks.
    filtered_chunks = []

    # Read the full cleaned CSV in chunks so the sampling step stays lightweight.
    for chunk in pd.read_csv(data_file, chunksize=250_000):
        # Build rounded cell keys for every row in this chunk.
        cell_keys = list(zip(chunk["lat"].round(5), chunk["lon"].round(5)))
        # Mark which rows belong to one of the sampled spatial cells.
        keep_mask = [key in sampled_cells for key in cell_keys]
        # Keep only the sampled rows from this chunk.
        kept = chunk.loc[keep_mask]
        # Save the chunk if at least one sampled row was present.
        if not kept.empty:
            filtered_chunks.append(kept)

    # If no sampled rows were found at all, stop with a clear error.
    if not filtered_chunks:
        raise SystemExit("No sampled rows were loaded from the cleaned monthly dataset.")

    # Combine all sampled chunks into one DataFrame.
    frame = pd.concat(filtered_chunks, ignore_index=True)
    # Convert the date column into datetime values.
    frame["date"] = pd.to_datetime(frame["date"])
    # Sort the sampled rows by location and date so downstream scripts stay consistent.
    frame = frame.sort_values(["lat", "lon", "date"]).reset_index(drop=True)
    # Save the sampled monthly table to disk for the model scripts to read.
    frame.to_csv(output_file, index=False)

    # Tell the user where the sampled CSV was saved.
    print(f"Saved sampled dataset to: {output_file}")
    # Tell the user how many sampled rows were written to that CSV.
    print(f"Sampled rows written: {len(frame):,}")
    # Return the sampled frame so the launcher can also inspect its split sizes.
    return frame


# Run one of the real model scripts against the sampled CSV.
def run_model_script(model_run, sampled_data_file, output_dir, forecast_months):
    # Start from a copy of the current environment so normal paths still work.
    env = os.environ.copy()
    # Preserve the writable Matplotlib cache for the child process.
    env["MPLCONFIGDIR"] = str(MPLCONFIGDIR)

    # Map each human-readable model name to its environment-variable prefix.
    prefix = {
        # Ridge uses the LR_* environment-variable prefix.
        "Ridge": "LR",
        # CatBoost uses the CATBOOST_* environment-variable prefix.
        "CatBoost": "CATBOOST",
        # XGBoost uses the XGB_* environment-variable prefix.
        "XGBoost": "XGB",
        # LightGBM uses the LIGHTGBM_* environment-variable prefix.
        "LightGBM": "LIGHTGBM",
    }[model_run["name"]]

    # Tell the child script to read the sampled demo CSV instead of the full CSV.
    env[f"{prefix}_DATA_FILE"] = str(sampled_data_file)
    # Tell the child script to save all artifacts into the demo output folder.
    env[f"{prefix}_OUTPUT_DIR"] = str(output_dir)
    # Tell the child script how many future months to forecast.
    env[f"{prefix}_FORECAST_MONTHS"] = str(forecast_months)
    # Apply the model-specific demo overrides from the configuration table.
    env.update(model_run["env"])

    # Build the command using the current Python interpreter and the target script.
    cmd = [sys.executable, str(model_run["script"])]
    # Print a banner so the terminal output clearly shows which model is running.
    print(f"\n=== RUNNING {model_run['name'].upper()} SCRIPT ===")
    # Print the exact command for transparency during the demo.
    print("Command:", " ".join(cmd))

    # Run the real model script and stop the launcher if it fails.
    subprocess.run(
        cmd,
        # Execute the child process from the project root directory.
        cwd=str(BASE_DIR),
        # Pass the prepared environment into the child process.
        env=env,
        # Raise an exception automatically if the script exits with an error.
        check=True,
    )


# Combine the per-model metrics CSV files into one demo summary table.
def combine_metrics(model_runs, output_path):
    # Start an empty list that will hold each model's metrics DataFrame.
    frames = []
    # Loop through each configured model run.
    for model_run in model_runs:
        # Build the expected path to this model's metrics CSV inside the demo folder.
        path = DEMO_DIR / model_run["metrics_file"]
        # Skip this model if its metrics CSV does not exist.
        if not path.exists():
            continue
        # Read the metrics CSV and copy it before modifying it.
        df = pd.read_csv(path).copy()
        # Add a demo-model label so the combined table stays easy to read.
        df["DemoModel"] = model_run["name"]
        # Save this model's DataFrame for the final concatenation step.
        frames.append(df)

    # If no metrics files were found, stop with a clear error.
    if not frames:
        raise SystemExit("No model metric files were produced by the demo run.")

    # Stack all model metric tables into one long combined DataFrame.
    combined = pd.concat(frames, ignore_index=True)
    # Save the combined metrics CSV to disk.
    combined.to_csv(output_path, index=False)
    # Tell the user where the combined metrics file was written.
    print("\nSaved combined evaluation metrics to:", output_path)

    # Keep only the test rows for a compact leaderboard in the terminal.
    leaderboard = combined[combined["Dataset"] == "Test"].copy()
    # If test rows exist, sort them by RMSE and print a short summary table.
    if not leaderboard.empty:
        # Lower RMSE is better, so sort ascending.
        leaderboard = leaderboard.sort_values("RMSE")
        # Print the model name and three key test metrics.
        print(leaderboard.loc[:, ["DemoModel", "RMSE", "MAE", "R2"]].to_string(index=False))


# Merge all model forecast CSV files into one side-by-side comparison file.
def merge_forecasts(model_runs, output_path):
    # Start with no merged forecast table yet.
    merged = None
    # Loop through each configured model run.
    for model_run in model_runs:
        # Build the expected path to this model's forecast CSV.
        path = DEMO_DIR / model_run["forecast_file"]
        # Skip this model if its forecast CSV does not exist.
        if not path.exists():
            continue
        # Read the forecast CSV and rename its PM2.5 column to include the model name.
        df = pd.read_csv(path).rename(columns={"pm25": f"{model_run['name'].lower()}_forecast_pm25"})
        # If this is the first forecast file, use it as the base merged table.
        if merged is None:
            merged = df
        # Otherwise merge this forecast onto the existing table by cell and date.
        else:
            merged = merged.merge(df, on=["lat", "lon", "date"], how="inner")

    # If no forecast files were found, stop with a clear error.
    if merged is None:
        raise SystemExit("No forecast CSV files were produced by the demo run.")

    # Save the combined forecast table to disk.
    merged.to_csv(output_path, index=False)
    # Tell the user where the combined forecast table was written.
    print("Saved combined forecast table to:", output_path)


# Run the full classroom-demo workflow from start to finish.
def main():
    # Parse the command-line options.
    args = parse_args()

    # Create the raw-data folder if it does not already exist.
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    # Create the processed-data folder if it does not already exist.
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    # Create the dedicated technical-demo folder if it does not already exist.
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    # Print a clear header so the start of the demo run is easy to spot.
    print("\n=== DS 340W JOINT TECHNICAL DEMO ===")
    # Print the resolved project root.
    print("Project root:", BASE_DIR)
    # Print the cleaned-data CSV path the launcher expects.
    print("Expected cleaned dataset:", DATA_FILE)
    # Print the folder where all demo outputs will be saved.
    print("Processed output folder:", DEMO_DIR)

    # Confirm that the cleaned monthly dataset is available before continuing.
    ensure_cleaned_dataset_ready()

    # Sample a subset of spatial cells from the full cleaned monthly dataset.
    sampled_cells = sample_cells_from_csv(DATA_FILE, args.sample_cells, RANDOM_SEED)
    # Build the output path for the sampled demo CSV.
    sampled_data_file = DEMO_DIR / f"{args.output_prefix}_sampled_clean.csv"
    # Write the sampled rows into that demo CSV and keep the resulting DataFrame.
    sampled_frame = write_sampled_dataset(DATA_FILE, sampled_cells, sampled_data_file)

    # Count how many sampled rows belong to the training period.
    train_rows = int((sampled_frame["date"] < TRAIN_END).sum())
    # Count how many sampled rows belong to the validation period.
    val_rows = int(((sampled_frame["date"] >= TRAIN_END) & (sampled_frame["date"] < VAL_END)).sum())
    # Count how many sampled rows belong to the test period.
    test_rows = int((sampled_frame["date"] >= VAL_END).sum())
    # Print a short split-size summary for the sampled CSV.
    print("\n--- RAW SPLIT COUNTS IN SAMPLED CSV ---")
    # Print the sampled training-row count.
    print(f"Train rows: {train_rows:,}")
    # Print the sampled validation-row count.
    print(f"Validation rows: {val_rows:,}")
    # Print the sampled test-row count.
    print(f"Test rows: {test_rows:,}")

    # Run each real model script on the sampled CSV one after another.
    for model_run in MODEL_RUNS:
        # Call the real model script with demo-specific input/output overrides.
        run_model_script(
            model_run=model_run,
            sampled_data_file=sampled_data_file,
            output_dir=DEMO_DIR,
            forecast_months=args.forecast_months,
        )

    # Build the output path for the combined demo metrics file.
    metrics_path = DEMO_DIR / f"{args.output_prefix}_combined_eval_metrics.csv"
    # Combine the per-model metric CSVs into that one summary file.
    combine_metrics(MODEL_RUNS, metrics_path)

    # Build the output path for the combined demo forecast file.
    combined_forecast_path = DEMO_DIR / f"{args.output_prefix}_combined_forecasts.csv"
    # Merge the per-model forecast CSVs into one comparison file.
    merge_forecasts(MODEL_RUNS, combined_forecast_path)

    # Print a completion message once the entire demo workflow is finished.
    print("\nTechnical demo run complete.")
    # Print the sampled CSV path so the user can inspect the exact demo input.
    print("Sampled CSV:", sampled_data_file)
    # Print the main combined metrics file path.
    print("Main metrics file:", metrics_path)
    # Print the main combined forecast file path.
    print("Main forecast file:", combined_forecast_path)


# Run the main workflow only when this file is executed directly.
if __name__ == "__main__":
    # Start the full classroom-demo launcher.
    main()
