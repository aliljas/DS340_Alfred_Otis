import pandas as pd
import numpy as np
import optuna
import cfgrib
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import matplotlib.pyplot as plt
from xgboost import plot_importance
import warnings
warnings.filterwarnings("ignore")

# Silence detailed Optuna logs to keep the console clean during the 40 trials
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- CONFIG ---
BASE_DIR  = Path(r"C:\Users\user\Downloads")
DATA_FILE = BASE_DIR / "na_pm25_cells_clean.csv"
GRIB_FILE = BASE_DIR / "18244b03099c9fa9b4eaa99fbc97822d.grib"
TARGET    = "pm25"

# Meteorological features extracted/derived from the ERA5 GRIB file
ERA5_COLS = [
    "era5_temperature_2m",
    "era5_surface_pressure",
    "era5_wind_speed",
    "era5_wind_dir_sin",
    "era5_wind_dir_cos",
    "era5_rh",
    "era5_precip_mm",
    "era5_dewpoint_spread",
]

# Combined feature list including spatial, temporal, and meteorological data
FEATURES = [
    "lat", "lon",
    "pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_lag7",
    "pm25_roll3_mean", "pm25_roll7_mean", "pm25_roll30_mean",
    "pm25_roll7_std", "pm25_ewm7",
    "pm25_deviation_7", "pm25_deviation_30",
    "pm25_ratio_7",
    "lat_lon_mean", "lat_lon_p75", "lat_lon_p90",
    "pm25_above_p75", "pm25_above_p90",
    "pm25_zscore",
    "neighbor_mean_lag1", "neighbor_max_lag1",
    "month_sin", "month_cos",
    "doy_sin", "doy_cos",
] + ERA5_COLS

# ================================================================
# STEP 1 — LOAD ERA5 GRIB
# ================================================================
# GRIB files are hierarchical. cfgrib often opens them as a list of datasets 
# because variables like temperature and pressure may be on different 'steps' or 'levels'.
print("Loading ERA5 grib file...")
datasets = cfgrib.open_datasets(GRIB_FILE)
print(f"  Found {len(datasets)} dataset(s) in grib")

all_parts = []
for i, ds in enumerate(datasets):
    # Convert xarray dataset to a pandas dataframe for easier merging
    df_part = ds.to_dataframe().reset_index()
    print(f"  Dataset {i}: vars={list(ds.data_vars)}, cols={df_part.columns.tolist()}, shape={df_part.shape}")
    all_parts.append(df_part)

# Standardization: cfgrib uses different names for the time dimension depending on the GRIB type.
normalized = []
for df_part in all_parts:
    time_col = None
    for c in ["valid_time", "time"]:
        if c in df_part.columns:
            time_col = c
            break
    if time_col is None:
        print("  WARNING: no time column found in a dataset, skipping")
        continue
    
    df_part = df_part.rename(columns={time_col: "era5_time"})
    df_part["era5_time"] = pd.to_datetime(df_part["era5_time"])
    
    # Filter out metadata columns (like 'step' or 'surface') to keep the dataframe lean
    keep = ["latitude", "longitude", "era5_time"] + [
        c for c in df_part.columns
        if c not in ["latitude", "longitude", "era5_time",
                     "step", "number", "surface", "heightAboveGround",
                     "heightAboveSea", "level", "valid_time", "time"]
        and not df_part[c].dtype == object
    ]
    normalized.append(df_part[[c for c in keep if c in df_part.columns]])

# Combine separate GRIB variable groups into a single wide weather dataframe
print("  Merging ERA5 datasets...")
df_era5 = normalized[0]
for df_part in normalized[1:]:
    df_era5 = df_era5.merge(
        df_part,
        on=["latitude", "longitude", "era5_time"],
        how="outer"
    )

# Longitude Alignment: ERA5 uses 0 to 360°. Most CSVs use -180 to 180°. 
# This logic converts ERA5 to the standard -180/180 format.
if df_era5["longitude"].max() > 180:
    df_era5["longitude"] = (df_era5["longitude"] + 180) % 360 - 180

# Clean up duplicates and rename coordinates to avoid confusion with PM2.5 coordinates
df_era5 = df_era5.groupby(["latitude", "longitude", "era5_time"]).first().reset_index()
df_era5 = df_era5.rename(columns={"latitude": "lat_era5", "longitude": "lon_era5"})

# Rename cryptic GRIB shortnames (e.g., '2t' or 't2m') to human-readable feature names
var_map = {}
for col in df_era5.columns:
    cl = col.lower()
    if cl in ["t2m", "2t"]:          var_map[col] = "temperature_2m"
    elif cl in ["d2m", "2d"]:        var_map[col] = "dewpoint_2m"
    elif cl in ["u10", "10u"]:       var_map[col] = "u_wind"
    elif cl in ["v10", "10v"]:       var_map[col] = "v_wind"
    elif cl in ["sp"]:               var_map[col] = "surface_pressure"
    elif cl in ["tp"]:               var_map[col] = "precip_raw"
df_era5 = df_era5.rename(columns=var_map)

# Unit Conversions:
# ERA5 default is Kelvin (K) and Pascals (Pa). These lines convert to Celsius (C) and Hectopascals (hPa).
if "temperature_2m" in df_era5.columns and df_era5["temperature_2m"].mean() > 100:
    df_era5["temperature_2m"] = df_era5["temperature_2m"] - 273.15

if "dewpoint_2m" in df_era5.columns and df_era5["dewpoint_2m"].mean() > 100:
    df_era5["dewpoint_2m"] = df_era5["dewpoint_2m"] - 273.15

if "surface_pressure" in df_era5.columns and df_era5["surface_pressure"].mean() > 10000:
    df_era5["surface_pressure"] = df_era5["surface_pressure"] / 100

# Monthly ERA5 precipitation is cumulative; convert from meters to millimeters.
if "precip_raw" in df_era5.columns:
    df_era5["precip_mm"] = df_era5["precip_raw"] * 1000
else:
    df_era5["precip_mm"] = 0.0

# Derived Physical Features:
# Calculating Wind Speed from U (East-West) and V (North-South) components.
if "u_wind" in df_era5.columns and "v_wind" in df_era5.columns:
    df_era5["wind_speed"] = np.sqrt(df_era5["u_wind"]**2 + df_era5["v_wind"]**2)
    wind_rad = np.arctan2(df_era5["u_wind"], df_era5["v_wind"])
    df_era5["wind_dir_sin"] = np.sin(wind_rad)
    df_era5["wind_dir_cos"] = np.cos(wind_rad)

# Calculating Relative Humidity (RH) and Dewpoint Spread (a proxy for atmospheric stability/inversions).
if "temperature_2m" in df_era5.columns and "dewpoint_2m" in df_era5.columns:
    T  = df_era5["temperature_2m"]
    Td = df_era5["dewpoint_2m"]
    df_era5["rh"] = (
        100 * np.exp((17.625 * Td) / (243.04 + Td)) /
              np.exp((17.625 * T)  / (243.04 + T))
    )
    df_era5["dewpoint_spread"] = T - Td

# Create a 'year_month' key to join monthly weather averages to daily pollution data.
df_era5["year_month"] = df_era5["era5_time"].dt.to_period("M")

# Grid Snapping: Round coords to match the ERA5 0.25-degree resolution.
df_era5["lat_era5"] = df_era5["lat_era5"].round(2)
df_era5["lon_era5"] = df_era5["lon_era5"].round(2)

# Final cleanup of weather dataframe
rename_final = {
    "temperature_2m":  "era5_temperature_2m",
    "surface_pressure":"era5_surface_pressure",
    "wind_speed":      "era5_wind_speed",
    "wind_dir_sin":    "era5_wind_dir_sin",
    "wind_dir_cos":    "era5_wind_dir_cos",
    "rh":              "era5_rh",
    "precip_mm":       "era5_precip_mm",
    "dewpoint_spread": "era5_dewpoint_spread",
}
df_era5 = df_era5.rename(columns=rename_final)
keep_cols = ["lat_era5", "lon_era5", "year_month"] + list(rename_final.values())
df_era5 = df_era5[keep_cols].drop_duplicates(subset=["lat_era5", "lon_era5", "year_month"])

# ================================================================
# STEP 2 — LOAD PM2.5 AND ENGINEER FEATURES
# ================================================================
print("\nLoading PM2.5 data...")
df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])
df[["lat", "lon", "pm25"]] = df[["lat", "lon", "pm25"]].astype(np.float32)
df = df.sort_values(["lat", "lon", "date"]).reset_index(drop=True)

# Lags and Rolling statistics help the model understand the recent history of pollution.
print("Engineering PM2.5 features...")
grp    = df.groupby(["lat", "lon"])["pm25"]
lagged = grp.shift(1)

for lag in [1, 2, 3, 7]:
    df[f"pm25_lag{lag}"] = grp.shift(lag)

df["pm25_roll3_mean"]  = lagged.transform(lambda x: x.rolling(3,  min_periods=1).mean())
df["pm25_roll7_mean"]  = lagged.transform(lambda x: x.rolling(7,  min_periods=1).mean())
df["pm25_roll30_mean"] = lagged.transform(lambda x: x.rolling(30, min_periods=1).mean())
df["pm25_roll7_std"]   = lagged.transform(lambda x: x.rolling(7,  min_periods=1).std().fillna(0))
df["pm25_ewm7"]        = lagged.transform(lambda x: x.ewm(span=7, min_periods=1).mean())

# Percentile-based features define "local extreme pollution" for every grid cell.
cell_stats = df.groupby(["lat", "lon"])["pm25"].agg(
    lat_lon_mean="mean",
    lat_lon_std="std",
    lat_lon_p75=lambda x: x.quantile(0.75),
    lat_lon_p90=lambda x: x.quantile(0.90),
).reset_index()
df = df.merge(cell_stats, on=["lat", "lon"], how="left")

df["pm25_above_p75"] = (df["pm25_lag1"] > df["lat_lon_p75"]).astype(np.float32)
df["pm25_above_p90"] = (df["pm25_lag1"] > df["lat_lon_p90"]).astype(np.float32)
df["pm25_zscore"]    = ((df["pm25_lag1"] - df["lat_lon_mean"]) / (df["lat_lon_std"] + 1e-3)).astype(np.float32)

# Spatial Features: Calculates the mean/max pollution of the 8 neighbors of every cell.
print("Computing spatial neighbor features...")
df["lat_r"] = df["lat"].round(1)
df["lon_r"] = df["lon"].round(1)
pivot      = df.pivot_table(index="date", columns=["lat_r", "lon_r"], values="pm25_lag1")
pivot_dict = {(round(k[0], 1), round(k[1], 1)): v for k, v in pivot.items()}

neighbor_means = np.full(len(df), np.nan, dtype=np.float32)
neighbor_maxes = np.full(len(df), np.nan, dtype=np.float32)
for i in range(len(df)):
    lat, lon, date = df["lat_r"].values[i], df["lon_r"].values[i], df["date"].values[i]
    vals = []
    for dlat in [-0.1, 0.0, 0.1]:
        for dlon in [-0.1, 0.0, 0.1]:
            if dlat == 0.0 and dlon == 0.0: continue
            s = pivot_dict.get((round(lat + dlat, 1), round(lon + dlon, 1)))
            if s is not None and date in s.index:
                v = s[date]
                if not np.isnan(v): vals.append(v)
    if vals:
        neighbor_means[i] = np.mean(vals)
        neighbor_maxes[i] = np.max(vals)

df["neighbor_mean_lag1"] = neighbor_means
df["neighbor_max_lag1"]  = neighbor_maxes
df["neighbor_mean_lag1"].fillna(df["pm25_lag1"], inplace=True)
df["neighbor_max_lag1"].fillna(df["pm25_lag1"],  inplace=True)

# Seasonal cyclical encoding (Sin/Cos)
df["month"]     = df["date"].dt.month
df["doy"]       = df["date"].dt.dayofyear
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).astype(np.float32)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).astype(np.float32)
df["doy_sin"]   = np.sin(2 * np.pi * df["doy"]   / 365).astype(np.float32)
df["doy_cos"]   = np.cos(2 * np.pi * df["doy"]   / 365).astype(np.float32)

# ================================================================
# STEP 3 — MERGE MONTHLY ERA5
# ================================================================
# Link the daily PM2.5 rows to the monthly ERA5 values using spatial snapping.
# PM2.5 is 0.1° resolution; ERA5 is 0.25°. Multiplying by 4 and rounding to the nearest integer
# effectively snaps the high-res PM2.5 point to the nearest low-res ERA5 grid box.
print("\nMerging monthly ERA5...")
df["lat_era5"]   = (np.round(df["lat"].values * 4) / 4).round(2)
df["lon_era5"]   = (np.round(df["lon"].values * 4) / 4).round(2)
df["year_month"] = df["date"].dt.to_period("M")

df = df.merge(df_era5, on=["lat_era5", "lon_era5", "year_month"], how="left")

# Data Imputation: If a location falls outside the ERA5 box (e.g., coastal cells),
# we fill the missing weather with the median value for that month.
for col in ERA5_COLS:
    if col in df.columns and df[col].isna().any():
        monthly_med = df.groupby("year_month")[col].transform("median")
        df[col]     = df[col].fillna(monthly_med)
        global_med  = df[col].median()
        df[col]     = df[col].fillna(global_med).astype(np.float32)

pm25_features = [f for f in FEATURES if not f.startswith("era5_")]
df = df.dropna(subset=pm25_features)

# ================================================================
# STEP 4 — SPLIT, OPTUNA, TRAIN
# ================================================================
# Temporal Splitting: Train (pre-2021), Validate (2021), Test (2022+)
train_df = df[df["date"] <  "2021-01-01"]
val_df   = df[(df["date"] >= "2021-01-01") & (df["date"] < "2022-01-01")]
test_df  = df[df["date"] >= "2022-01-01"]

# Objective function for Optuna to find the most accurate hyperparameters
def objective(trial):
    param = {
        "n_estimators":          600,
        "learning_rate":         trial.suggest_float("learning_rate",    0.01, 0.15, log=True),
        "max_depth":             trial.suggest_int(  "max_depth",        3,    7),
        "subsample":             trial.suggest_float("subsample",        0.6,  1.0),
        "colsample_bytree":      trial.suggest_float("colsample_bytree", 0.6,  1.0),
        "min_child_weight":      trial.suggest_int(  "min_child_weight", 5,    30),
        "reg_alpha":             trial.suggest_float("reg_alpha",        0.1,  10.0, log=True),
        "reg_lambda":            trial.suggest_float("reg_lambda",       0.1,  10.0, log=True),
        "gamma":                 trial.suggest_float("gamma",            0.0,  3.0),
        "objective":             "reg:squarederror",
        "tree_method":           "hist",
        "early_stopping_rounds": 50,
        "random_state":          42,
        "n_jobs":                -1,
    }
    tune_df = train_df.sample(n=min(1_000_000, len(train_df)), random_state=42)
    model    = XGBRegressor(**param)
    model.fit(tune_df[FEATURES], tune_df[TARGET], eval_set=[(val_df[FEATURES], val_df[TARGET])], verbose=False)
    preds = model.predict(val_df[FEATURES])
    return mean_squared_error(val_df[TARGET], preds) ** 0.5

# Run the hyperparameter search
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=40, show_progress_bar=True)

# Train the final high-resolution model with the best parameters
print("\nTraining final model...")
final_model = XGBRegressor(
    **study.best_params,
    n_estimators=2000,
    objective="reg:squarederror",
    tree_method="hist",
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1,
)
final_model.fit(train_df[FEATURES], train_df[TARGET], eval_set=[(val_df[FEATURES], val_df[TARGET])], verbose=100)

# ================================================================
# STEP 5 — EVALUATION & PLOTTING
# ================================================================
def evaluate(model, X, y, split_name):
    preds = model.predict(X)
    rmse  = mean_squared_error(y, preds) ** 0.5
    mae   = mean_absolute_error(y, preds)
    r2    = r2_score(y, preds)
    print(f"\n[{split_name}]  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")
    return preds

test_preds = evaluate(final_model, test_df[FEATURES], test_df[TARGET], "Test")

# Generate visualization of feature importance, error scatter, and residual distribution
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
plot_importance(final_model, ax=axes[0], max_num_features=15, importance_type="gain")
axes[1].scatter(test_df[TARGET], test_preds, alpha=0.3, s=5, color="steelblue")
lims = [min(test_df[TARGET].min(), test_preds.min()), max(test_df[TARGET].max(), test_preds.max())]
axes[1].plot(lims, lims, "r--")
residuals = test_df[TARGET].values - test_preds
axes[2].hist(residuals, bins=60, color="coral", edgecolor="white")
plt.tight_layout()
plt.show()
