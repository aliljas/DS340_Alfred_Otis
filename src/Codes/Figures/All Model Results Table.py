import os
import tempfile
from pathlib import Path


# Use a writable Matplotlib cache so the table figure works on school machines.
MPLCONFIGDIR = Path(tempfile.gettempdir()) / "matplotlib-cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from figure_path_utils import (
    ALL_MODEL_RESULTS_PLOT_FILE,
    ALL_MODEL_RESULTS_TABLE_FILE,
    ALL_MODEL_RESULTS_WITH_ERA_PLOT_FILE,
    ALL_MODEL_RESULTS_WITH_ERA_TABLE_FILE,
    PROCESSED_DIR,
)


# Main metrics files already produced by the project.
LR_METRICS_FILE = PROCESSED_DIR / "lr_eval_metrics.csv"
CATBOOST_METRICS_FILE = PROCESSED_DIR / "catboost_eval_metrics.csv"
CATBOOST_ERA_ONLY_METRICS_FILE = PROCESSED_DIR / "catboost_era_only" / "catboost_eval_metrics.csv"
LIGHTGBM_METRICS_FILE = PROCESSED_DIR / "lightgbm_eval_metrics.csv"
XGB_METRICS_FILE = PROCESSED_DIR / "xgb_eval_metrics.csv"

# Comparison files preserve both the no-ERA baseline and the with-ERA reruns.
LR_COMPARISON_FILE = PROCESSED_DIR / "lr_era5_comparison_metrics.csv"
CATBOOST_COMPARISON_FILE = PROCESSED_DIR / "catboost_era5_comparison_metrics.csv"
LIGHTGBM_COMPARISON_FILE = PROCESSED_DIR / "lightgbm_era5_comparison_metrics.csv"
XGB_COMPARISON_FILE = PROCESSED_DIR / "xgb_era5_comparison_metrics.csv"


def load_test_row(path, model_name, display_scenario):
    # Read the standard per-model metrics CSV and keep only the held-out test row.
    frame = pd.read_csv(path)
    test_row = frame[frame["Dataset"] == "Test"].copy()
    if test_row.empty:
        raise SystemExit(f"No Test row was found in {path}.")
    row = test_row.iloc[0]
    return {
        "Model": model_name,
        "Scenario": display_scenario,
        "RMSE": float(row["RMSE"]),
        "MAE": float(row["MAE"]),
        "R2": float(row["R2"]),
        "Source": path.name,
    }


def load_comparison_test_row(path, model_name, scenario_key, display_scenario):
    # Read the comparison CSV and keep the requested test row.
    frame = pd.read_csv(path)
    test_row = frame[(frame["Scenario"] == scenario_key) & (frame["Dataset"] == "Test")].copy()
    if test_row.empty:
        raise SystemExit(f"No Test {scenario_key} row was found in {path}.")
    row = test_row.iloc[0]
    return {
        "Model": model_name,
        "Scenario": display_scenario,
        "RMSE": float(row["RMSE"]),
        "MAE": float(row["MAE"]),
        "R2": float(row["R2"]),
        "Source": path.name,
    }


def load_naive_row():
    # Pull the shared naive benchmark from the first metrics CSV that still has it.
    candidate_paths = [
        LR_METRICS_FILE,
        CATBOOST_METRICS_FILE,
        CATBOOST_ERA_ONLY_METRICS_FILE,
        LIGHTGBM_METRICS_FILE,
        XGB_METRICS_FILE,
    ]
    for path in candidate_paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        naive_row = frame[frame["Dataset"] == "Naive_Test"].copy()
        if naive_row.empty:
            continue
        row = naive_row.iloc[0]
        return {
            "Model": "Naive",
            "Scenario": "lag1 baseline",
            "RMSE": float(row["RMSE"]),
            "MAE": float(row["MAE"]),
            "R2": float(row["R2"]),
            "Source": path.name,
        }

    raise SystemExit("No Naive_Test row was found in the available model metrics files.")


def load_model_row(model_name, comparison_file, metrics_file, scenario_key, display_scenario):
    # Prefer the explicit comparison CSV because those files preserve both scenarios.
    if comparison_file.exists():
        return load_comparison_test_row(comparison_file, model_name, scenario_key, display_scenario)
    # Fall back to the main per-model metrics file when the comparison CSV is missing.
    if metrics_file.exists():
        return load_test_row(metrics_file, model_name, display_scenario)
    raise SystemExit(f"Neither {comparison_file} nor {metrics_file} exists for {model_name}.")


def build_results_table(use_era):
    # Choose the internal scenario key and the paper-friendly label together.
    scenario_key = "With_ERA5" if use_era else "Without_ERA5"
    display_scenario = "With ERA5" if use_era else "Without ERA5"

    # Collect one held-out test row for each final model plus the shared naive baseline.
    results = [
        load_naive_row(),
        load_model_row("Ridge Regression", LR_COMPARISON_FILE, LR_METRICS_FILE, scenario_key, display_scenario),
        load_model_row("XGBoost", XGB_COMPARISON_FILE, XGB_METRICS_FILE, scenario_key, display_scenario),
        load_model_row("CatBoost", CATBOOST_COMPARISON_FILE, CATBOOST_ERA_ONLY_METRICS_FILE, scenario_key, display_scenario),
        load_model_row("LightGBM", LIGHTGBM_COMPARISON_FILE, LIGHTGBM_METRICS_FILE, scenario_key, display_scenario),
    ]

    results_df = pd.DataFrame(results)

    # Keep the models in the presentation order used throughout the project.
    model_order = ["Naive", "Ridge Regression", "XGBoost", "CatBoost", "LightGBM"]
    results_df["Model"] = pd.Categorical(results_df["Model"], categories=model_order, ordered=True)
    results_df = results_df.sort_values("Model").reset_index(drop=True)
    return results_df


def format_results_table(results_df):
    # Round the display columns for a cleaner paper-ready table.
    display_df = results_df.copy()
    display_df["RMSE"] = display_df["RMSE"].map(lambda value: f"{value:.3f}")
    display_df["MAE"] = display_df["MAE"].map(lambda value: f"{value:.3f}")
    display_df["R2"] = display_df["R2"].map(lambda value: f"{value:.3f}")
    return display_df


def save_table_figure(display_df, title, output_path):
    # Build a clean table-only figure for the paper.
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    # Style the table so it reads more like a paper figure than a spreadsheet.
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.1, 1.7)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#7f8c8d")
        if row == 0:
            cell.set_facecolor("#1f3a5f")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif row == 1:
            cell.set_facecolor("#eef4fb")
        elif display_df.iloc[row - 1, 0] == "Naive":
            cell.set_facecolor("#f5f5f5")
        else:
            cell.set_facecolor("white")

    ax.set_title(title, fontsize=15, pad=18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_results_bundle(results_df, csv_path, plot_path, title):
    # Save the underlying summary table as CSV too.
    results_df.to_csv(csv_path, index=False)

    # Keep only the presentation-friendly columns in the figure.
    display_df = results_df.loc[:, ["Model", "Scenario", "RMSE", "MAE", "R2"]]
    display_df = format_results_table(display_df)
    save_table_figure(display_df, title, plot_path)


def main():
    # Build the final no-ERA summary from the explicit baseline comparison rows.
    baseline_df = build_results_table(use_era=False)
    save_results_bundle(
        baseline_df,
        ALL_MODEL_RESULTS_TABLE_FILE,
        ALL_MODEL_RESULTS_PLOT_FILE,
        "Test-Set Forecasting Performance Across All Models",
    )

    # Build a second summary from the with-ERA comparison rows.
    with_era_df = build_results_table(use_era=True)
    save_results_bundle(
        with_era_df,
        ALL_MODEL_RESULTS_WITH_ERA_TABLE_FILE,
        ALL_MODEL_RESULTS_WITH_ERA_PLOT_FILE,
        "Test-Set Forecasting Performance Across All Models With ERA5",
    )

    print("Saved all-model baseline results summary table to:", ALL_MODEL_RESULTS_TABLE_FILE)
    print("Saved all-model baseline results summary figure to:", ALL_MODEL_RESULTS_PLOT_FILE)
    print("Saved all-model with-ERA results summary table to:", ALL_MODEL_RESULTS_WITH_ERA_TABLE_FILE)
    print("Saved all-model with-ERA results summary figure to:", ALL_MODEL_RESULTS_WITH_ERA_PLOT_FILE)


if __name__ == "__main__":
    main()
