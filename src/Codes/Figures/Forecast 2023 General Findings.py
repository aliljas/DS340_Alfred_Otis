import os
import tempfile
from pathlib import Path


# Use a writable Matplotlib cache so the script works across machines.
MPLCONFIGDIR = Path(tempfile.gettempdir()) / "matplotlib-cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from figure_path_utils import (
    CATBOOST_PREDICTIONS_FILE,
    FORECAST_2023_GENERAL_FINDINGS_FILE,
    FORECAST_2023_GENERAL_TRENDS_PLOT,
    FORECAST_2023_MONTHLY_SUMMARY_FILE,
    FORECAST_2023_TIMESERIES_PLOT,
    LIGHTGBM_PREDICTIONS_FILE,
    LR_PREDICTIONS_FILE,
    XGBOOST_PREDICTIONS_FILE,
    require_file,
)


MODEL_SOURCES = [
    ("Ridge Regression", LR_PREDICTIONS_FILE, "#1b9e77"),
    ("XGBoost", XGBOOST_PREDICTIONS_FILE, "#7570b3"),
    ("CatBoost", CATBOOST_PREDICTIONS_FILE, "#d95f02"),
    ("LightGBM", LIGHTGBM_PREDICTIONS_FILE, "#e7298a"),
]


def build_model_summaries():
    monthly_frames = []
    findings_rows = []

    for model_name, pred_file, _color in MODEL_SOURCES:
        require_file(pred_file, f"{model_name} 2023 forecast file")
        frame = pd.read_csv(pred_file)
        frame["date"] = pd.to_datetime(frame["date"])

        monthly_df = (
            frame.groupby("date", as_index=False)["pm25"]
            .agg(mean_pm25="mean", min_pm25="min", max_pm25="max", observation_count="count")
            .sort_values("date")
        )
        monthly_df["range_pm25"] = monthly_df["max_pm25"] - monthly_df["min_pm25"]
        monthly_df["model"] = model_name
        monthly_frames.append(monthly_df)

        overall_min_idx = frame["pm25"].idxmin()
        overall_max_idx = frame["pm25"].idxmax()
        overall_min_row = frame.loc[overall_min_idx, ["lat", "lon", "pm25", "date"]]
        overall_max_row = frame.loc[overall_max_idx, ["lat", "lon", "pm25", "date"]]
        highest_avg_month = monthly_df.loc[monthly_df["mean_pm25"].idxmax()]
        lowest_avg_month = monthly_df.loc[monthly_df["mean_pm25"].idxmin()]

        findings_rows.append(
            {
                "model": model_name,
                "annual_mean_pm25": float(frame["pm25"].mean()),
                "annual_min_pm25": float(overall_min_row["pm25"]),
                "annual_min_date": pd.Timestamp(overall_min_row["date"]).strftime("%Y-%m-%d"),
                "annual_min_lat": float(overall_min_row["lat"]),
                "annual_min_lon": float(overall_min_row["lon"]),
                "annual_max_pm25": float(overall_max_row["pm25"]),
                "annual_max_date": pd.Timestamp(overall_max_row["date"]).strftime("%Y-%m-%d"),
                "annual_max_lat": float(overall_max_row["lat"]),
                "annual_max_lon": float(overall_max_row["lon"]),
                "highest_average_month": pd.Timestamp(highest_avg_month["date"]).strftime("%Y-%m-%d"),
                "highest_average_month_mean_pm25": float(highest_avg_month["mean_pm25"]),
                "lowest_average_month": pd.Timestamp(lowest_avg_month["date"]).strftime("%Y-%m-%d"),
                "lowest_average_month_mean_pm25": float(lowest_avg_month["mean_pm25"]),
            }
        )

    findings_df = pd.DataFrame(findings_rows)
    monthly_summary_df = pd.concat(monthly_frames, ignore_index=True)
    return findings_df, monthly_summary_df


def save_timeseries_plot(monthly_summary_df):
    fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=300)
    for model_name, _pred_file, color in MODEL_SOURCES:
        model_monthly = monthly_summary_df[monthly_summary_df["model"] == model_name].copy()
        ax.plot(
            model_monthly["date"],
            model_monthly["mean_pm25"],
            marker="o",
            linewidth=2,
            label=model_name,
            color=color,
        )

    ax.set_title("Forecasted Monthly Mean PM2.5 in 2023")
    ax.set_ylabel("Mean PM2.5 (µg/m³)")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FORECAST_2023_TIMESERIES_PLOT, bbox_inches="tight")
    plt.close(fig)


def save_summary_plot(findings_df):
    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=300)
    bar_positions = range(len(findings_df))
    ax.bar(
        [x - 0.2 for x in bar_positions],
        findings_df["annual_min_pm25"],
        width=0.2,
        label="Annual min",
        color="#91bfdb",
    )
    ax.bar(
        bar_positions,
        findings_df["annual_mean_pm25"],
        width=0.2,
        label="Annual mean",
        color="#4575b4",
    )
    ax.bar(
        [x + 0.2 for x in bar_positions],
        findings_df["annual_max_pm25"],
        width=0.2,
        label="Annual max",
        color="#d73027",
    )
    ax.set_xticks(list(bar_positions))
    ax.set_xticklabels(findings_df["model"], rotation=20)
    ax.set_title("Forecasted 2023 Min, Mean, and Max PM2.5 by Model")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FORECAST_2023_GENERAL_TRENDS_PLOT, bbox_inches="tight")
    plt.close(fig)


def main():
    findings_df, monthly_summary_df = build_model_summaries()
    findings_df.to_csv(FORECAST_2023_GENERAL_FINDINGS_FILE, index=False)
    monthly_summary_df.to_csv(FORECAST_2023_MONTHLY_SUMMARY_FILE, index=False)
    save_timeseries_plot(monthly_summary_df)
    save_summary_plot(findings_df)

    print("Saved forecast 2023 general findings to:", FORECAST_2023_GENERAL_FINDINGS_FILE)
    print("Saved forecast 2023 monthly summary to:", FORECAST_2023_MONTHLY_SUMMARY_FILE)
    print("Saved forecast 2023 time-series plot to:", FORECAST_2023_TIMESERIES_PLOT)
    print("Saved forecast 2023 trends plot to:", FORECAST_2023_GENERAL_TRENDS_PLOT)

    for row in findings_df.itertuples(index=False):
        print(
            f" - {row.model}: max {row.annual_max_pm25:.3f} on {row.annual_max_date}, "
            f"min {row.annual_min_pm25:.3f} on {row.annual_min_date}, "
            f"highest average month {row.highest_average_month}, "
            f"lowest average month {row.lowest_average_month}"
        )


if __name__ == "__main__":
    main()
