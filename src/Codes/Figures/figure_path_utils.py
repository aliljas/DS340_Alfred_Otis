import os
import tempfile
from pathlib import Path


#Cross-platform Matplotlib cache
MPLCONFIGDIR = Path(tempfile.gettempdir()) / "matplotlib-cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

#Project paths
BASE_DIR = Path(__file__).resolve().parents[3]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIGURE_DIR = PROCESSED_DIR / "figure data"
FIGURE_OUTPUT_DIR = FIGURE_DIR / "output"
FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#Shared inputs
FIGURE1_FILE = FIGURE_DIR / "Figure 1.xlsx"
FIGURE6_FILE = FIGURE_DIR / "Figure 6.xlsx"
CATBOOST_PREDICTIONS_FILE = PROCESSED_DIR / "catboost_predictions_2023.csv"
LR_PREDICTIONS_FILE = PROCESSED_DIR / "lr_predictions_2023.csv"
XGBOOST_PREDICTIONS_FILE = PROCESSED_DIR / "xgb_predictions_2023.csv"
LIGHTGBM_PREDICTIONS_FILE = PROCESSED_DIR / "lightgbm_predictions_2023.csv"
CATBOOST_TIMESERIES_FILE = PROCESSED_DIR / "catboost_2023_monthly_timeseries.csv"
CATBOOST_TIMESERIES_PLOT = PROCESSED_DIR / "catboost_2023_time_series.png"
FORECAST_OVERLAY_FILE = PROCESSED_DIR / "forecast_model_overlay_2023.csv"
FORECAST_OVERLAY_PLOT = PROCESSED_DIR / "forecast_model_overlay_2023.png"
TOP_FEATURES_FIGURE_FILE = PROCESSED_DIR / "top5_features_all_models.png"
TOP_FEATURES_TABLE_FILE = PROCESSED_DIR / "top5_features_all_models.csv"
XGB_CATBOOST_COMPARISON_TABLE_FILE = PROCESSED_DIR / "xgb_catboost_comparison.csv"
XGB_CATBOOST_COMPARISON_PLOT_FILE = PROCESSED_DIR / "xgb_catboost_comparison.png"
ALL_MODEL_RESULTS_TABLE_FILE = PROCESSED_DIR / "all_model_results_summary.csv"
ALL_MODEL_RESULTS_PLOT_FILE = PROCESSED_DIR / "all_model_results_summary.png"
ALL_MODEL_RESULTS_WITH_ERA_TABLE_FILE = PROCESSED_DIR / "all_model_results_summary_with_era.csv"
ALL_MODEL_RESULTS_WITH_ERA_PLOT_FILE = PROCESSED_DIR / "all_model_results_summary_with_era.png"
PM25_GENERAL_FINDINGS_FILE = PROCESSED_DIR / "pm25_general_findings.csv"
PM25_MONTHLY_SUMMARY_FILE = PROCESSED_DIR / "pm25_monthly_summary.csv"
PM25_YEARLY_SUMMARY_FILE = PROCESSED_DIR / "pm25_yearly_summary.csv"
PM25_CLIMATOLOGY_SUMMARY_FILE = PROCESSED_DIR / "pm25_calendar_month_summary.csv"
PM25_GENERAL_TRENDS_PLOT = PROCESSED_DIR / "pm25_general_trends.png"
FORECAST_2023_GENERAL_FINDINGS_FILE = PROCESSED_DIR / "forecast_2023_general_findings.csv"
FORECAST_2023_MONTHLY_SUMMARY_FILE = PROCESSED_DIR / "forecast_2023_monthly_summary.csv"
FORECAST_2023_GENERAL_TRENDS_PLOT = PROCESSED_DIR / "forecast_2023_general_trends.png"
FORECAST_2023_TIMESERIES_PLOT = PROCESSED_DIR / "forecast_2023_timeseries.png"
FORECAST_VS_ACTUAL_MONTHLY_FILE = PROCESSED_DIR / "forecast_vs_actual_monthly_summary.csv"
FORECAST_VS_ACTUAL_MONTHLY_PLOT = PROCESSED_DIR / "forecast_vs_actual_monthly_comparison.png"
WORLD_COUNTRIES_ZIP = FIGURE_DIR / "ne_110m_admin_0_countries.zip"
RAW_NC_DIR = FIGURE_DIR / "raw_nc"
GEOTIFF_OUTPUT_DIR = FIGURE_OUTPUT_DIR / "geotiff"
GEOTIFF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


#Missing-file guard
def require_file(path, description):
    path = Path(path)
    if not path.exists():
        raise SystemExit(
            f"{description} was not found at {path}. "
            "Make sure the project data folders were downloaded and placed in the expected locations."
        )
    return path
