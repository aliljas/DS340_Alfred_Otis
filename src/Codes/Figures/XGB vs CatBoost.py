import tempfile
from pathlib import Path


# Use a writable plotting cache so the figure also works on school machines.
MPLCONFIGDIR = Path(tempfile.gettempdir()) / "matplotlib-cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_path_utils import (
    PROCESSED_DIR,
    XGB_CATBOOST_COMPARISON_PLOT_FILE,
    XGB_CATBOOST_COMPARISON_TABLE_FILE,
)


# Prefer real project output files when they exist.
XGB_EVAL_FILE = PROCESSED_DIR / "xgb_eval_metrics.csv"
XGB_ERA5_COMPARISON_FILE = PROCESSED_DIR / "xgb_era5_comparison_metrics.csv"
CATBOOST_EVAL_FILE = PROCESSED_DIR / "catboost_eval_metrics.csv"
CATBOOST_ERA5_COMPARISON_FILE = PROCESSED_DIR / "catboost_era5_comparison_metrics.csv"


# Fallback full-run XGBoost results taken from the saved project console output.
XGB_FALLBACK_RESULTS = [
    {
        "Model": "XGBoost",
        "Dataset": "Validation",
        "RMSE": 3.0682,
        "MAE": 1.5815,
        "R2": 0.5725,
        "Source": "manual_full_run_fallback",
    },
    {
        "Model": "XGBoost",
        "Dataset": "Test",
        "RMSE": 1.8635,
        "MAE": 1.2095,
        "R2": 0.6078,
        "Source": "manual_full_run_fallback",
    },
]


# Fallback CatBoost no-ERA results taken from the project's comparison CSV.
CATBOOST_FALLBACK_RESULTS = [
    {
        "Model": "CatBoost",
        "Dataset": "Validation",
        "RMSE": 3.227361,
        "MAE": 1.488512,
        "R2": 0.520329,
        "Source": "manual_no_era_fallback",
    },
    {
        "Model": "CatBoost",
        "Dataset": "Test",
        "RMSE": 1.792935,
        "MAE": 1.158893,
        "R2": 0.632510,
        "Source": "manual_no_era_fallback",
    },
]


def load_xgboost_results():
    # Prefer the no-ERA XGBoost comparison so the comparison stays fair.
    if XGB_ERA5_COMPARISON_FILE.exists():
        frame = pd.read_csv(XGB_ERA5_COMPARISON_FILE)
        baseline = frame[frame["Scenario"] == "Without_ERA5"].copy()
        if not baseline.empty:
            baseline["Model"] = "XGBoost"
            baseline["Source"] = "xgb_era5_comparison_metrics.csv"
            return baseline.loc[:, ["Model", "Dataset", "RMSE", "MAE", "R2", "Source"]]

    # Use the real XGBoost evaluation CSV when it exists.
    if XGB_EVAL_FILE.exists():
        frame = pd.read_csv(XGB_EVAL_FILE)
        frame = frame[frame["Dataset"].isin(["Validation", "Test"])].copy()
        frame["Model"] = "XGBoost"
        frame["Source"] = "xgb_eval_metrics.csv"
        return frame.loc[:, ["Model", "Dataset", "RMSE", "MAE", "R2", "Source"]]

    # Otherwise fall back to the stored project run results.
    return pd.DataFrame(XGB_FALLBACK_RESULTS)


def load_catboost_results():
    # Prefer the no-ERA CatBoost comparison so the comparison stays fair.
    if CATBOOST_ERA5_COMPARISON_FILE.exists():
        frame = pd.read_csv(CATBOOST_ERA5_COMPARISON_FILE)
        baseline = frame[frame["Scenario"] == "Without_ERA5"].copy()
        if not baseline.empty:
            baseline["Model"] = "CatBoost"
            baseline["Source"] = "catboost_era5_comparison_metrics.csv"
            return baseline.loc[:, ["Model", "Dataset", "RMSE", "MAE", "R2", "Source"]]

    # If the no-ERA comparison file is missing, use the main CatBoost metrics file.
    if CATBOOST_EVAL_FILE.exists():
        frame = pd.read_csv(CATBOOST_EVAL_FILE)
        frame = frame[frame["Dataset"].isin(["Validation", "Test"])].copy()
        frame["Model"] = "CatBoost"
        frame["Source"] = "catboost_eval_metrics.csv"
        return frame.loc[:, ["Model", "Dataset", "RMSE", "MAE", "R2", "Source"]]

    # Final fallback keeps the figure script usable even if project outputs are missing.
    return pd.DataFrame(CATBOOST_FALLBACK_RESULTS)


def add_value_labels(ax, bars, decimals=3):
    # Label each bar so the visual works even without surrounding text.
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def main():
    # Combine the two model tables into one plot-ready frame.
    comparison_df = pd.concat(
        [load_xgboost_results(), load_catboost_results()],
        ignore_index=True,
    )

    # Keep the rows in a predictable order for plotting and export.
    dataset_order = ["Validation", "Test"]
    model_order = ["XGBoost", "CatBoost"]
    comparison_df["Dataset"] = pd.Categorical(comparison_df["Dataset"], categories=dataset_order, ordered=True)
    comparison_df["Model"] = pd.Categorical(comparison_df["Model"], categories=model_order, ordered=True)
    comparison_df = comparison_df.sort_values(["Dataset", "Model"]).reset_index(drop=True)

    # Save the combined table so the paper has a matching source file.
    comparison_df.to_csv(XGB_CATBOOST_COMPARISON_TABLE_FILE, index=False)

    # Build side-by-side metric panels for a quick visual comparison.
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    metrics = [("RMSE", "RMSE"), ("MAE", "MAE"), ("R2", "R²")]
    x = np.arange(len(dataset_order))
    width = 0.34
    colors = {"XGBoost": "#1f77b4", "CatBoost": "#ff7f0e"}

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        xgb_values = (
            comparison_df[comparison_df["Model"] == "XGBoost"]
            .set_index("Dataset")
            .loc[dataset_order, metric_key]
            .values
        )
        cat_values = (
            comparison_df[comparison_df["Model"] == "CatBoost"]
            .set_index("Dataset")
            .loc[dataset_order, metric_key]
            .values
        )

        xgb_bars = ax.bar(x - width / 2, xgb_values, width=width, color=colors["XGBoost"], label="XGBoost")
        cat_bars = ax.bar(x + width / 2, cat_values, width=width, color=colors["CatBoost"], label="CatBoost")

        add_value_labels(ax, xgb_bars)
        add_value_labels(ax, cat_bars)

        ax.set_xticks(x)
        ax.set_xticklabels(dataset_order)
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label} Comparison")
        ax.grid(axis="y", alpha=0.25, linestyle="--")

    # Add one shared title and one shared legend so the figure stays compact.
    fig.suptitle("XGBoost vs CatBoost on the Shared PM2.5 Forecasting Pipeline", fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(XGB_CATBOOST_COMPARISON_PLOT_FILE, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved XGBoost vs CatBoost comparison table to:", XGB_CATBOOST_COMPARISON_TABLE_FILE)
    print("Saved XGBoost vs CatBoost comparison plot to:", XGB_CATBOOST_COMPARISON_PLOT_FILE)


if __name__ == "__main__":
    main()
