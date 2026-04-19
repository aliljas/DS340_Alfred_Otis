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
    FORECAST_2023_MONTHLY_SUMMARY_FILE,
    FORECAST_VS_ACTUAL_MONTHLY_FILE,
    FORECAST_VS_ACTUAL_MONTHLY_PLOT,
    PM25_CLIMATOLOGY_SUMMARY_FILE,
    require_file,
)


MODEL_COLORS = {
    "Ridge Regression": "#1b9e77",
    "XGBoost": "#7570b3",
    "CatBoost": "#d95f02",
    "LightGBM": "#e7298a",
}


def month_name_from_date(date_value):
    return pd.Timestamp(date_value).strftime("%B")


def build_comparison_frame():
    forecast_file = require_file(FORECAST_2023_MONTHLY_SUMMARY_FILE, "Forecast 2023 monthly summary")
    actual_file = require_file(PM25_CLIMATOLOGY_SUMMARY_FILE, "Observed PM2.5 calendar-month summary")

    forecast_df = pd.read_csv(forecast_file)
    forecast_df["date"] = pd.to_datetime(forecast_df["date"])
    forecast_df["calendar_month"] = forecast_df["date"].dt.month
    forecast_df["month_name"] = forecast_df["date"].map(month_name_from_date)

    actual_df = pd.read_csv(actual_file)
    actual_df = actual_df.rename(columns={"mean_pm25": "actual_mean_pm25"})

    comparison_df = forecast_df.merge(
        actual_df.loc[:, ["calendar_month", "month_name", "actual_mean_pm25"]],
        on=["calendar_month", "month_name"],
        how="left",
    )
    comparison_df["forecast_minus_actual"] = comparison_df["mean_pm25"] - comparison_df["actual_mean_pm25"]
    return comparison_df


def save_comparison_plot(comparison_df):
    month_order = list(range(1, 13))
    month_labels = [
        pd.Timestamp(year=2000, month=month_number, day=1).strftime("%b")
        for month_number in month_order
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    actual_series = (
        comparison_df.loc[:, ["calendar_month", "actual_mean_pm25"]]
        .drop_duplicates()
        .sort_values("calendar_month")
    )
    axes[0].plot(
        actual_series["calendar_month"],
        actual_series["actual_mean_pm25"],
        color="#333333",
        linewidth=2.5,
        marker="o",
        label="Observed monthly average (2017–2022)",
    )

    for model_name, color in MODEL_COLORS.items():
        model_df = comparison_df[comparison_df["model"] == model_name].sort_values("calendar_month")
        axes[0].plot(
            model_df["calendar_month"],
            model_df["mean_pm25"],
            linewidth=2,
            marker="o",
            label=model_name,
            color=color,
        )

    axes[0].set_title("Forecasted 2023 Monthly Means vs Observed Monthly Averages")
    axes[0].set_ylabel("Mean PM2.5 (µg/m³)")
    axes[0].set_xticks(month_order)
    axes[0].set_xticklabels(month_labels)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    for model_name, color in MODEL_COLORS.items():
        model_df = comparison_df[comparison_df["model"] == model_name].sort_values("calendar_month")
        axes[1].plot(
            model_df["calendar_month"],
            model_df["forecast_minus_actual"],
            linewidth=2,
            marker="o",
            label=model_name,
            color=color,
        )

    axes[1].axhline(0, color="#333333", linewidth=1, linestyle="--")
    axes[1].set_title("Forecasted 2023 Monthly Means Minus Observed Monthly Averages")
    axes[1].set_ylabel("Forecast - Observed (µg/m³)")
    axes[1].set_xticks(month_order)
    axes[1].set_xticklabels(month_labels)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FORECAST_VS_ACTUAL_MONTHLY_PLOT, bbox_inches="tight")
    plt.close(fig)


def main():
    comparison_df = build_comparison_frame()
    comparison_df.to_csv(FORECAST_VS_ACTUAL_MONTHLY_FILE, index=False)
    save_comparison_plot(comparison_df)

    print("Saved forecast-vs-actual monthly comparison table to:", FORECAST_VS_ACTUAL_MONTHLY_FILE)
    print("Saved forecast-vs-actual monthly comparison plot to:", FORECAST_VS_ACTUAL_MONTHLY_PLOT)


if __name__ == "__main__":
    main()
