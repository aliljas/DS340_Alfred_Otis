#Load libraries
import argparse
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


MPLCONFIGDIR = Path(tempfile.gettempdir()) / "matplotlib-cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))


BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data/raw"
PROCESSED_DIR = BASE_DIR / "data/processed"
DEMO_DIR = PROCESSED_DIR / "technical_demo"
DATA_FILE = PROCESSED_DIR / "na_pm25_cells_clean.csv"

TRAIN_END = pd.Timestamp("2021-01-01")
VAL_END = pd.Timestamp("2022-01-01")
RANDOM_SEED = 42

CLEANED_DATA_ZIP_CANDIDATES = [
    PROCESSED_DIR / "na_pm25_cells_clean.csv.zip",
    RAW_DIR / "na_pm25_cells_clean.csv.zip",
    RAW_DIR / "na_pm25_cells_clean.zip",
]

MODEL_RUNS = [
    {
        "name": "Ridge",
        "script": BASE_DIR / "src/Codes/LR_model.py",
        "env": {
            "LR_COMPARE_ERA5": "0",
            "LR_USE_ERA5": "0",
            "LR_SAVE_PLOT": "0",
            "LR_SAVE_COMPARISON_PLOT": "0",
            "LR_RUN_FORECAST": "1",
        },
        "metrics_file": "lr_eval_metrics.csv",
        "forecast_file": "lr_predictions_2023.csv",
    },
    {
        "name": "CatBoost",
        "script": BASE_DIR / "src/Codes/CatBoost_model.py",
        "env": {
            "CATBOOST_COMPARE_ERA5": "0",
            "CATBOOST_USE_ERA5": "0",
            "CATBOOST_SAVE_PLOT": "0",
            "CATBOOST_SAVE_COMPARISON_PLOT": "0",
            "CATBOOST_RUN_FORECAST": "1",
        },
        "metrics_file": "catboost_eval_metrics.csv",
        "forecast_file": "catboost_predictions_2023.csv",
    },
    {
        "name": "XGBoost",
        "script": BASE_DIR / "src/Codes/XGB_model.py",
        "env": {
            "XGB_COMPARE_ERA5": "0",
            "XGB_USE_ERA5": "0",
            "XGB_SAVE_PLOT": "0",
            "XGB_SAVE_COMPARISON_PLOT": "0",
            "XGB_RUN_FORECAST": "1",
            "XGB_SKIP_OPTUNA": "1",
        },
        "metrics_file": "xgb_eval_metrics.csv",
        "forecast_file": "xgb_predictions_2023.csv",
    },
    {
        "name": "LightGBM",
        "script": BASE_DIR / "src/Codes/LightGBM_model.py",
        "env": {
            "LIGHTGBM_COMPARE_ERA5": "0",
            "LIGHTGBM_USE_ERA5": "0",
            "LIGHTGBM_SAVE_PLOT": "0",
            "LIGHTGBM_SAVE_COMPARISON_PLOT": "0",
            "LIGHTGBM_RUN_FORECAST": "1",
        },
        "metrics_file": "lightgbm_eval_metrics.csv",
        "forecast_file": "lightgbm_predictions_2023.csv",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the DS 340W classroom demo by calling the real model scripts on sampled data."
    )
    parser.add_argument(
        "--sample-cells",
        type=int,
        default=1000,
        help="Number of spatial cells to sample from the cleaned monthly dataset.",
    )
    parser.add_argument(
        "--forecast-months",
        type=int,
        default=12,
        help="How many future months each model should forecast.",
    )
    parser.add_argument(
        "--output-prefix",
        default="joint_technical_demo",
        help="Prefix for saved combined demo outputs.",
    )
    return parser.parse_args()


def extract_cleaned_dataset_zip():
    for zip_path in CLEANED_DATA_ZIP_CANDIDATES:
        if not zip_path.exists():
            continue
        print(f"Extracting cleaned dataset from: {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(PROCESSED_DIR)
        return


def ensure_cleaned_dataset_ready():
    if DATA_FILE.exists():
        print("Found existing cleaned dataset:", DATA_FILE)
        return

    extract_cleaned_dataset_zip()

    if not DATA_FILE.exists():
        raise SystemExit(
            "The class demo expects data/processed/na_pm25_cells_clean.csv "
            "or a ZIP copy of that file. Place the cleaned CSV or "
            "na_pm25_cells_clean.csv.zip in data/processed or data/raw and rerun."
        )

    print("Extracted cleaned dataset to:", DATA_FILE)


def sample_cells_from_csv(data_file, sample_cell_count, random_seed):
    rng = np.random.default_rng(random_seed)
    unique_cells = set()

    for chunk in pd.read_csv(data_file, usecols=["lat", "lon"], chunksize=250_000):
        unique_cells.update(zip(chunk["lat"].round(5), chunk["lon"].round(5)))

    all_cells = np.array(sorted(unique_cells), dtype=np.float32)
    sample_n = min(sample_cell_count, len(all_cells))
    sample_idx = rng.choice(len(all_cells), size=sample_n, replace=False)

    print(f"Unique cells available: {len(all_cells):,}")
    print(f"Sampled cells for this demo: {sample_n:,}")
    return {tuple(cell) for cell in all_cells[sample_idx]}


def write_sampled_dataset(data_file, sampled_cells, output_file):
    filtered_chunks = []

    for chunk in pd.read_csv(data_file, chunksize=250_000):
        cell_keys = list(zip(chunk["lat"].round(5), chunk["lon"].round(5)))
        keep_mask = [key in sampled_cells for key in cell_keys]
        kept = chunk.loc[keep_mask]
        if not kept.empty:
            filtered_chunks.append(kept)

    if not filtered_chunks:
        raise SystemExit("No sampled rows were loaded from the cleaned monthly dataset.")

    frame = pd.concat(filtered_chunks, ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["lat", "lon", "date"]).reset_index(drop=True)
    frame.to_csv(output_file, index=False)

    print(f"Saved sampled dataset to: {output_file}")
    print(f"Sampled rows written: {len(frame):,}")
    return frame


def run_model_script(model_run, sampled_data_file, output_dir, forecast_months):
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(MPLCONFIGDIR)

    prefix = {
        "Ridge": "LR",
        "CatBoost": "CATBOOST",
        "XGBoost": "XGB",
        "LightGBM": "LIGHTGBM",
    }[model_run["name"]]

    env[f"{prefix}_DATA_FILE"] = str(sampled_data_file)
    env[f"{prefix}_OUTPUT_DIR"] = str(output_dir)
    env[f"{prefix}_FORECAST_MONTHS"] = str(forecast_months)
    env.update(model_run["env"])

    cmd = [sys.executable, str(model_run["script"])]
    print(f"\n=== RUNNING {model_run['name'].upper()} SCRIPT ===")
    print("Command:", " ".join(cmd))

    subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        env=env,
        check=True,
    )


def combine_metrics(model_runs, output_path):
    frames = []
    for model_run in model_runs:
        path = DEMO_DIR / model_run["metrics_file"]
        if not path.exists():
            continue
        df = pd.read_csv(path).copy()
        df["DemoModel"] = model_run["name"]
        frames.append(df)

    if not frames:
        raise SystemExit("No model metric files were produced by the demo run.")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_path, index=False)
    print("\nSaved combined evaluation metrics to:", output_path)

    leaderboard = combined[combined["Dataset"] == "Test"].copy()
    if not leaderboard.empty:
        leaderboard = leaderboard.sort_values("RMSE")
        print(leaderboard.loc[:, ["DemoModel", "RMSE", "MAE", "R2"]].to_string(index=False))


def merge_forecasts(model_runs, output_path):
    merged = None
    for model_run in model_runs:
        path = DEMO_DIR / model_run["forecast_file"]
        if not path.exists():
            continue
        df = pd.read_csv(path).rename(columns={"pm25": f"{model_run['name'].lower()}_forecast_pm25"})
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=["lat", "lon", "date"], how="inner")

    if merged is None:
        raise SystemExit("No forecast CSV files were produced by the demo run.")

    merged.to_csv(output_path, index=False)
    print("Saved combined forecast table to:", output_path)


def main():
    args = parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== DS 340W JOINT TECHNICAL DEMO ===")
    print("Project root:", BASE_DIR)
    print("Expected cleaned dataset:", DATA_FILE)
    print("Processed output folder:", DEMO_DIR)

    ensure_cleaned_dataset_ready()

    sampled_cells = sample_cells_from_csv(DATA_FILE, args.sample_cells, RANDOM_SEED)
    sampled_data_file = DEMO_DIR / f"{args.output_prefix}_sampled_clean.csv"
    sampled_frame = write_sampled_dataset(DATA_FILE, sampled_cells, sampled_data_file)

    train_rows = int((sampled_frame["date"] < TRAIN_END).sum())
    val_rows = int(((sampled_frame["date"] >= TRAIN_END) & (sampled_frame["date"] < VAL_END)).sum())
    test_rows = int((sampled_frame["date"] >= VAL_END).sum())
    print("\n--- RAW SPLIT COUNTS IN SAMPLED CSV ---")
    print(f"Train rows: {train_rows:,}")
    print(f"Validation rows: {val_rows:,}")
    print(f"Test rows: {test_rows:,}")

    for model_run in MODEL_RUNS:
        run_model_script(
            model_run=model_run,
            sampled_data_file=sampled_data_file,
            output_dir=DEMO_DIR,
            forecast_months=args.forecast_months,
        )

    metrics_path = DEMO_DIR / f"{args.output_prefix}_combined_eval_metrics.csv"
    combine_metrics(MODEL_RUNS, metrics_path)

    combined_forecast_path = DEMO_DIR / f"{args.output_prefix}_combined_forecasts.csv"
    merge_forecasts(MODEL_RUNS, combined_forecast_path)

    print("\nTechnical demo run complete.")
    print("Sampled CSV:", sampled_data_file)
    print("Main metrics file:", metrics_path)
    print("Main forecast file:", combined_forecast_path)


if __name__ == "__main__":
    main()
