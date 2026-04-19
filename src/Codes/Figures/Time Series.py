from figure_path_utils import (
    CATBOOST_PREDICTIONS_FILE,
    CATBOOST_TIMESERIES_FILE,
    CATBOOST_TIMESERIES_PLOT,
    FORECAST_OVERLAY_FILE,
    FORECAST_OVERLAY_PLOT,
    LIGHTGBM_PREDICTIONS_FILE,
    LR_PREDICTIONS_FILE,
    XGBOOST_PREDICTIONS_FILE,
    require_file,
)
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#Paths
PRED_FILE = CATBOOST_PREDICTIONS_FILE
OUTPUT_FIG = CATBOOST_TIMESERIES_PLOT
OUTPUT_CSV = CATBOOST_TIMESERIES_FILE
OVERLAY_OUTPUT_FIG = FORECAST_OVERLAY_PLOT
OVERLAY_OUTPUT_CSV = FORECAST_OVERLAY_FILE

require_file(PRED_FILE, "CatBoost 2023 forecast file")
df = pd.read_csv(PRED_FILE)
df["date"] = pd.to_datetime(df["date"])

monthly_ts = (
    df.groupby("date", as_index=False)["pm25"]
    .mean()
    .sort_values("date")
    .rename(columns={"pm25": "avg_pm25"})
)

monthly_ts.to_csv(OUTPUT_CSV, index=False)

plt.figure(figsize=(9, 5), dpi=300)
plt.plot(monthly_ts["date"], monthly_ts["avg_pm25"], linewidth=2, marker="o")
plt.xlabel("Date")
plt.ylabel("Average Forecasted PM2.5 (µg/m³)")
plt.title("Forecasted Monthly PM2.5 Across North America in 2023")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_FIG, bbox_inches="tight")
plt.close()


def build_monthly_series(pred_file, value_name):
    model_df = pd.read_csv(pred_file)
    model_df["date"] = pd.to_datetime(model_df["date"])
    return (
        model_df.groupby("date", as_index=False)["pm25"]
        .mean()
        .sort_values("date")
        .rename(columns={"pm25": value_name})
    )


overlay_sources = [
    ("CatBoost", CATBOOST_PREDICTIONS_FILE, "catboost_avg_pm25", "#d95f02"),
    ("Ridge Regression", LR_PREDICTIONS_FILE, "ridge_avg_pm25", "#1b9e77"),
    ("XGBoost", XGBOOST_PREDICTIONS_FILE, "xgboost_avg_pm25", "#7570b3"),
    ("LightGBM", LIGHTGBM_PREDICTIONS_FILE, "lightgbm_avg_pm25", "#e7298a"),
]

available_models = []
overlay_df = None

for model_name, pred_file, value_name, color in overlay_sources:
    if not pred_file.exists():
        continue

    monthly_model = build_monthly_series(pred_file, value_name)
    if overlay_df is None:
        overlay_df = monthly_model
    else:
        overlay_df = overlay_df.merge(monthly_model, on="date", how="outer")

    available_models.append((model_name, value_name, color))

if overlay_df is None:
    raise SystemExit("No forecast prediction files were found for the 2023 overlay figure.")

overlay_df = overlay_df.sort_values("date")
overlay_df.to_csv(OVERLAY_OUTPUT_CSV, index=False)

plt.figure(figsize=(10, 5.5), dpi=300)
for model_name, value_name, color in available_models:
    plt.plot(
        overlay_df["date"],
        overlay_df[value_name],
        linewidth=2.2,
        marker="o",
        label=model_name,
        color=color,
    )

plt.xlabel("Date")
plt.ylabel("Average Forecasted PM2.5 (µg/m³)")
plt.title("Forecasted Monthly PM2.5 Across North America in 2023 by Model")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OVERLAY_OUTPUT_FIG, bbox_inches="tight")
plt.close()
