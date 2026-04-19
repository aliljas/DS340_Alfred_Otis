import os
import tempfile
from collections import defaultdict
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
    PM25_CLIMATOLOGY_SUMMARY_FILE,
    PM25_GENERAL_FINDINGS_FILE,
    PM25_GENERAL_TRENDS_PLOT,
    PM25_MONTHLY_SUMMARY_FILE,
    PM25_YEARLY_SUMMARY_FILE,
    PROCESSED_DIR,
    require_file,
)


DATA_FILE = PROCESSED_DIR / "na_pm25_cells_clean.csv"
CHUNK_SIZE = 500_000


def update_bucket(bucket, key, values):
    # Accumulate sum, count, min, and max so we can build descriptive summaries
    # without loading the full 23M-row table into memory at once.
    stats = bucket[key]
    stats["sum"] += float(values["sum"])
    stats["count"] += int(values["count"])
    stats["min"] = min(stats["min"], float(values["min"]))
    stats["max"] = max(stats["max"], float(values["max"]))


def finalize_bucket(bucket, key_name, sort_key):
    rows = []
    for key, stats in bucket.items():
        rows.append(
            {
                key_name: key,
                "mean_pm25": stats["sum"] / stats["count"],
                "min_pm25": stats["min"],
                "max_pm25": stats["max"],
                "range_pm25": stats["max"] - stats["min"],
                "observation_count": stats["count"],
            }
        )
    frame = pd.DataFrame(rows).sort_values(sort_key).reset_index(drop=True)
    return frame


def month_name_from_number(month_number):
    return pd.Timestamp(year=2000, month=int(month_number), day=1).strftime("%B")


def build_summaries(data_file):
    # Track global descriptive statistics while streaming through the dataset.
    overall_sum = 0.0
    overall_count = 0
    overall_min = None
    overall_max = None
    overall_min_row = None
    overall_max_row = None

    monthly_stats = defaultdict(lambda: {"sum": 0.0, "count": 0, "min": float("inf"), "max": float("-inf")})
    yearly_stats = defaultdict(lambda: {"sum": 0.0, "count": 0, "min": float("inf"), "max": float("-inf")})
    climatology_stats = defaultdict(lambda: {"sum": 0.0, "count": 0, "min": float("inf"), "max": float("-inf")})

    for chunk in pd.read_csv(data_file, chunksize=CHUNK_SIZE):
        chunk["date"] = pd.to_datetime(chunk["date"])
        chunk["year"] = chunk["date"].dt.year
        chunk["calendar_month"] = chunk["date"].dt.month
        chunk["date_month"] = chunk["date"].dt.to_period("M").dt.to_timestamp()

        overall_sum += float(chunk["pm25"].sum())
        overall_count += int(len(chunk))

        chunk_min_idx = chunk["pm25"].idxmin()
        chunk_max_idx = chunk["pm25"].idxmax()
        chunk_min_row = chunk.loc[chunk_min_idx, ["lat", "lon", "pm25", "date"]]
        chunk_max_row = chunk.loc[chunk_max_idx, ["lat", "lon", "pm25", "date"]]

        if overall_min is None or float(chunk_min_row["pm25"]) < overall_min:
            overall_min = float(chunk_min_row["pm25"])
            overall_min_row = chunk_min_row.to_dict()

        if overall_max is None or float(chunk_max_row["pm25"]) > overall_max:
            overall_max = float(chunk_max_row["pm25"])
            overall_max_row = chunk_max_row.to_dict()

        monthly_grouped = chunk.groupby("date_month")["pm25"].agg(["sum", "count", "min", "max"])
        for key, values in monthly_grouped.iterrows():
            update_bucket(monthly_stats, pd.Timestamp(key), values)

        yearly_grouped = chunk.groupby("year")["pm25"].agg(["sum", "count", "min", "max"])
        for key, values in yearly_grouped.iterrows():
            update_bucket(yearly_stats, int(key), values)

        climatology_grouped = chunk.groupby("calendar_month")["pm25"].agg(["sum", "count", "min", "max"])
        for key, values in climatology_grouped.iterrows():
            update_bucket(climatology_stats, int(key), values)

    monthly_df = finalize_bucket(monthly_stats, "date", "date")
    yearly_df = finalize_bucket(yearly_stats, "year", "year")
    climatology_df = finalize_bucket(climatology_stats, "calendar_month", "calendar_month")
    climatology_df["month_name"] = climatology_df["calendar_month"].map(month_name_from_number)

    highest_avg_month = monthly_df.loc[monthly_df["mean_pm25"].idxmax()]
    lowest_avg_month = monthly_df.loc[monthly_df["mean_pm25"].idxmin()]
    highest_avg_year = yearly_df.loc[yearly_df["mean_pm25"].idxmax()]
    lowest_avg_year = yearly_df.loc[yearly_df["mean_pm25"].idxmin()]
    highest_climatology_month = climatology_df.loc[climatology_df["mean_pm25"].idxmax()]
    lowest_climatology_month = climatology_df.loc[climatology_df["mean_pm25"].idxmin()]

    findings_rows = [
        {"finding": "total_observations", "value": int(overall_count)},
        {"finding": "overall_mean_pm25", "value": overall_sum / overall_count},
        {"finding": "overall_min_pm25", "value": overall_min},
        {"finding": "overall_min_date", "value": pd.Timestamp(overall_min_row["date"]).strftime("%Y-%m-%d")},
        {"finding": "overall_min_lat", "value": float(overall_min_row["lat"])},
        {"finding": "overall_min_lon", "value": float(overall_min_row["lon"])},
        {"finding": "overall_max_pm25", "value": overall_max},
        {"finding": "overall_max_date", "value": pd.Timestamp(overall_max_row["date"]).strftime("%Y-%m-%d")},
        {"finding": "overall_max_lat", "value": float(overall_max_row["lat"])},
        {"finding": "overall_max_lon", "value": float(overall_max_row["lon"])},
        {"finding": "highest_average_month", "value": pd.Timestamp(highest_avg_month["date"]).strftime("%Y-%m-%d")},
        {"finding": "highest_average_month_mean_pm25", "value": float(highest_avg_month["mean_pm25"])},
        {"finding": "lowest_average_month", "value": pd.Timestamp(lowest_avg_month["date"]).strftime("%Y-%m-%d")},
        {"finding": "lowest_average_month_mean_pm25", "value": float(lowest_avg_month["mean_pm25"])},
        {"finding": "highest_average_year", "value": int(highest_avg_year["year"])},
        {"finding": "highest_average_year_mean_pm25", "value": float(highest_avg_year["mean_pm25"])},
        {"finding": "lowest_average_year", "value": int(lowest_avg_year["year"])},
        {"finding": "lowest_average_year_mean_pm25", "value": float(lowest_avg_year["mean_pm25"])},
        {"finding": "highest_climatology_month", "value": highest_climatology_month["month_name"]},
        {"finding": "highest_climatology_month_mean_pm25", "value": float(highest_climatology_month["mean_pm25"])},
        {"finding": "lowest_climatology_month", "value": lowest_climatology_month["month_name"]},
        {"finding": "lowest_climatology_month_mean_pm25", "value": float(lowest_climatology_month["mean_pm25"])},
    ]
    findings_df = pd.DataFrame(findings_rows)

    return findings_df, monthly_df, yearly_df, climatology_df


def save_trend_plot(monthly_df, yearly_df, climatology_df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=300)
    axes = axes.flatten()

    # Overall monthly mean trend across the full study period.
    axes[0].plot(monthly_df["date"], monthly_df["mean_pm25"], color="#1f77b4", linewidth=2)
    axes[0].set_title("Monthly Mean PM2.5 Across North America")
    axes[0].set_ylabel("Mean PM2.5 (µg/m³)")
    axes[0].grid(alpha=0.25)

    # Show the spread between monthly min and max values so spike periods stand out.
    axes[1].fill_between(
        monthly_df["date"],
        monthly_df["min_pm25"],
        monthly_df["max_pm25"],
        color="#ffcc80",
        alpha=0.55,
        label="Monthly min-max range",
    )
    axes[1].plot(monthly_df["date"], monthly_df["mean_pm25"], color="#d95f02", linewidth=1.8, label="Monthly mean")
    axes[1].set_title("Monthly PM2.5 Range and Mean")
    axes[1].set_ylabel("PM2.5 (µg/m³)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    # Summarize yearly mean PM2.5 so broad temporal changes are easy to discuss.
    axes[2].bar(yearly_df["year"].astype(str), yearly_df["mean_pm25"], color="#7570b3")
    axes[2].set_title("Yearly Mean PM2.5")
    axes[2].set_ylabel("Mean PM2.5 (µg/m³)")
    axes[2].grid(axis="y", alpha=0.25)

    # Show the average seasonal cycle by calendar month.
    axes[3].bar(climatology_df["month_name"], climatology_df["mean_pm25"], color="#1b9e77")
    axes[3].set_title("Average PM2.5 by Calendar Month")
    axes[3].set_ylabel("Mean PM2.5 (µg/m³)")
    axes[3].tick_params(axis="x", rotation=45)
    axes[3].grid(axis="y", alpha=0.25)

    fig.suptitle("General PM2.5 Findings and Time Trends", fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig(PM25_GENERAL_TRENDS_PLOT, bbox_inches="tight")
    plt.close(fig)


def main():
    data_file = require_file(DATA_FILE, "Cleaned PM2.5 modeling table")
    findings_df, monthly_df, yearly_df, climatology_df = build_summaries(data_file)

    findings_df.to_csv(PM25_GENERAL_FINDINGS_FILE, index=False)
    monthly_df.to_csv(PM25_MONTHLY_SUMMARY_FILE, index=False)
    yearly_df.to_csv(PM25_YEARLY_SUMMARY_FILE, index=False)
    climatology_df.to_csv(PM25_CLIMATOLOGY_SUMMARY_FILE, index=False)
    save_trend_plot(monthly_df, yearly_df, climatology_df)

    print("Saved general PM2.5 findings to:", PM25_GENERAL_FINDINGS_FILE)
    print("Saved monthly PM2.5 summary to:", PM25_MONTHLY_SUMMARY_FILE)
    print("Saved yearly PM2.5 summary to:", PM25_YEARLY_SUMMARY_FILE)
    print("Saved calendar-month PM2.5 summary to:", PM25_CLIMATOLOGY_SUMMARY_FILE)
    print("Saved PM2.5 general trends plot to:", PM25_GENERAL_TRENDS_PLOT)

    overall_max = findings_df.loc[findings_df["finding"] == "overall_max_pm25", "value"].iloc[0]
    overall_max_date = findings_df.loc[findings_df["finding"] == "overall_max_date", "value"].iloc[0]
    overall_min = findings_df.loc[findings_df["finding"] == "overall_min_pm25", "value"].iloc[0]
    overall_min_date = findings_df.loc[findings_df["finding"] == "overall_min_date", "value"].iloc[0]
    high_month = findings_df.loc[findings_df["finding"] == "highest_average_month", "value"].iloc[0]
    low_month = findings_df.loc[findings_df["finding"] == "lowest_average_month", "value"].iloc[0]

    print("\nKey findings:")
    print(f" - Overall max PM2.5: {float(overall_max):.3f} on {overall_max_date}")
    print(f" - Overall min PM2.5: {float(overall_min):.3f} on {overall_min_date}")
    print(f" - Highest continental-average month: {high_month}")
    print(f" - Lowest continental-average month: {low_month}")


if __name__ == "__main__":
    main()
