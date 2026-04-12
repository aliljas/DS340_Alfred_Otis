import pandas as pd
import numpy as np
import optuna
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import matplotlib.pyplot as plt
from xgboost import plot_importance

# Set Optuna to quiet mode so it only prints warnings, not every trial result
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- CONFIG ---
# Define file paths and the target column for the machine learning model
BASE_DIR = Path(r"C:\Users\user\Downloads")
DATA_FILE = BASE_DIR / "na_pm25_cells_clean.csv"
TARGET = "pm25"

# List of all input features (independent variables) used to predict PM2.5
FEATURES = [
    "lat", "lon",
    "pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_lag7",
    "pm25_roll3_mean", "pm25_roll7_mean", "pm25_roll30_mean",
    "pm25_roll7_std",
    "pm25_ewm7",
    "pm25_deviation_7", "pm25_deviation_30",
    "pm25_ratio_7",
    "lat_lon_mean",
    "lat_lon_p75",         # 75th percentile baseline per cell — helps model learn high-pollution threshold
    "lat_lon_p90",         # 90th percentile — flags chronically polluted cells
    "pm25_above_p75",      # is today's lag1 above the cell's 75th pct? (binary episode flag)
    "pm25_above_p90",      # is today's lag1 above the cell's 90th pct?
    "pm25_zscore",         # standardized anomaly within cell — scale-invariant spike detector
    "neighbor_mean_lag1",  # spatial: mean PM2.5 of nearby grid cells yesterday
    "neighbor_max_lag1",   # spatial: max PM2.5 of nearby grid cells — catches upwind plumes
    "month_sin", "month_cos",
    "doy_sin", "doy_cos",
]

# --- LOAD ---
print("Loading data...")
df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])
# Using float32 reduces memory usage without significant loss in precision
df[["lat", "lon", "pm25"]] = df[["lat", "lon", "pm25"]].astype(np.float32)

# --- FEATURE ENGINEERING ---
print("Engineering features...")
# Sorting is critical for time-series features like lags and rolling means
df = df.sort_values(["lat", "lon", "date"]).reset_index(drop=True)

# Grouping by location ensures that 'yesterday' refers to the same place, not a different city
grp    = df.groupby(["lat", "lon"])["pm25"]
lagged = grp.shift(1)

# Autoregressive features: Using values from 1, 2, 3, and 7 days ago
for lag in [1, 2, 3, 7]:
    df[f"pm25_lag{lag}"] = grp.shift(lag)

# Rolling statistics: Captures short-term (3d), mid-term (7d), and long-term (30d) air quality trends
df["pm25_roll3_mean"]  = lagged.transform(lambda x: x.rolling(3,  min_periods=1).mean())
df["pm25_roll7_mean"]  = lagged.transform(lambda x: x.rolling(7,  min_periods=1).mean())
df["pm25_roll30_mean"] = lagged.transform(lambda x: x.rolling(30, min_periods=1).mean())
# Volatility: Shows if air quality has been stable or fluctuating recently
df["pm25_roll7_std"]   = lagged.transform(lambda x: x.rolling(7,  min_periods=1).std().fillna(0))
# Exponentially Weighted Mean: Gives more importance to the most recent days
df["pm25_ewm7"]        = lagged.transform(lambda x: x.ewm(span=7, min_periods=1).mean())

# Relative features: Is today much higher or lower than the recent average?
df["pm25_deviation_7"]  = df["pm25_lag1"] - df["pm25_roll7_mean"]
df["pm25_deviation_30"] = df["pm25_lag1"] - df["pm25_roll30_mean"]
df["pm25_ratio_7"]      = df["pm25_lag1"] / (df["pm25_roll7_mean"] + 1e-3)

# --- CELL-LEVEL PERCENTILE FEATURES ---
# Normalization: This defines a "local normal" for every cell. 
# It helps the model distinguish a spike from a baseline high-pollution area.
cell_stats = df.groupby(["lat", "lon"])["pm25"].agg(
    lat_lon_mean="mean",
    lat_lon_std="std",
    lat_lon_p75=lambda x: x.quantile(0.75),
    lat_lon_p90=lambda x: x.quantile(0.90),
).reset_index()

df = df.merge(cell_stats, on=["lat", "lon"], how="left")
for col in ["lat_lon_mean", "lat_lon_std", "lat_lon_p75", "lat_lon_p90"]:
    df[col] = df[col].astype(np.float32)

# Categorizing episodes: Flags if the air was in the "extreme" range for that specific location yesterday
df["pm25_above_p75"] = (df["pm25_lag1"] > df["lat_lon_p75"]).astype(np.float32)
df["pm25_above_p90"] = (df["pm25_lag1"] > df["lat_lon_p90"]).astype(np.float32)

# Scaling: Z-score identifies how many standard deviations today's value is from the local mean
df["pm25_zscore"] = (
    (df["pm25_lag1"] - df["lat_lon_mean"]) / (df["lat_lon_std"] + 1e-3)
).astype(np.float32)

# --- SPATIAL NEIGHBOR FEATURES ---
# Spatiotemporal link: Pollution in one cell often travels to the neighbor the next day.
print("Computing spatial neighbor features (this may take a minute)...")

df["lat_r"] = df["lat"].round(1)
df["lon_r"] = df["lon"].round(1)

# Pivoting makes spatial lookup efficient by creating a 2D grid indexed by date
pivot = df.pivot_table(index="date", columns=["lat_r", "lon_r"], values="pm25_lag1")

# Pre-convert pivot to a dictionary for faster iterative access
pivot_dict = {}
for (lat_r, lon_r), series in pivot.items():
    pivot_dict[(round(lat_r, 1), round(lon_r, 1))] = series

# Prepare empty arrays to store spatial results
neighbor_means = np.full(len(df), np.nan, dtype=np.float32)
neighbor_maxes = np.full(len(df), np.nan, dtype=np.float32)

# Iterate through every row to check the 8 surrounding cells (3x3 grid minus center)
for i in range(len(df)):
    lat, lon, date = df["lat_r"].values[i], df["lon_r"].values[i], df["date"].values[i]
    vals = []
    for dlat in [-0.1, 0.0, 0.1]:
        for dlon in [-0.1, 0.0, 0.1]:
            if dlat == 0.0 and dlon == 0.0:
                continue # Skip the current cell itself
            key = (round(lat + dlat, 1), round(lon + dlon, 1))
            s = pivot_dict.get(key)
            if s is not None and date in s.index:
                v = s[date]
                if not np.isnan(v):
                    vals.append(v)
    if vals:
        neighbor_means[i] = np.mean(vals)
        neighbor_maxes[i] = np.max(vals)

df["neighbor_mean_lag1"] = neighbor_means
df["neighbor_max_lag1"]  = neighbor_maxes

# Domain edges: If a cell is on the coast, fill missing neighbor values with the cell's own value
df["neighbor_mean_lag1"].fillna(df["pm25_lag1"], inplace=True)
df["neighbor_max_lag1"].fillna(df["pm25_lag1"],  inplace=True)

# Cyclical Encoding: Sin/Cos transforms convert months/days into a circle.
# This ensures December (12) and January (1) are seen as close together by the model.
df["month"] = df["date"].dt.month
df["doy"]   = df["date"].dt.dayofyear
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).astype(np.float32)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).astype(np.float32)
df["doy_sin"]   = np.sin(2 * np.pi * df["doy"]   / 365).astype(np.float32)
df["doy_cos"]   = np.cos(2 * np.pi * df["doy"]   / 365).astype(np.float32)

# Drop any rows that couldn't compute features (e.g., the first 7 days of the dataset)
df = df.dropna(subset=FEATURES)

# --- SPLIT ---
# Time-based splitting: We train on history, validate on the mid-term, and test on the "future"
train_df = df[df["date"] <  "2021-01-01"]
val_df   = df[(df["date"] >= "2021-01-01") & (df["date"] < "2022-01-01")]
test_df  = df[df["date"] >= "2022-01-01"]

# --- OPTUNA HYPERPARAMETER TUNING ---
# Optuna searches for the best combination of settings (learning rate, depth, etc.) to minimize error.
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
        "n_jobs":                -1, # Use all available CPU cores
    }
    # Use a sample of the training data to speed up the tuning process
    tune_df = train_df.sample(n=min(1_000_000, len(train_df)), random_state=42)
    model    = XGBRegressor(**param)
    model.fit(
        tune_df[FEATURES], tune_df[TARGET],
        eval_set=[(val_df[FEATURES], val_df[TARGET])],
        verbose=False,
    )
    preds = model.predict(val_df[FEATURES])
    return mean_squared_error(val_df[TARGET], preds) ** 0.5

print("\nStarting Hyperparameter Optimization...")
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=40, show_progress_bar=True)

# --- FINAL TRAINING ---
# Train the model one last time using the best parameters found by Optuna
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
final_model.fit(
    train_df[FEATURES], train_df[TARGET],
    eval_set=[(val_df[FEATURES], val_df[TARGET])],
    verbose=100,
)

# --- EVALUATION ---
# Calculate standard metrics: RMSE (error magnitude), MAE (average error), and R2 (fit quality)
def evaluate(model, X, y, split_name):
    preds = model.predict(X)
    rmse  = mean_squared_error(y, preds) ** 0.5
    mae   = mean_absolute_error(y, preds)
    r2    = r2_score(y, preds)
    mape  = np.mean(np.abs((y - preds) / np.where(y == 0, 1, y))) * 100
    print(f"\n[{split_name}]  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}  MAPE={mape:.2f}%")
    return preds

val_preds  = evaluate(final_model, val_df[FEATURES],  val_df[TARGET],  "Validation")
test_preds = evaluate(final_model, test_df[FEATURES], test_df[TARGET], "Test")

# --- VISUALIZATION ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Which features were most useful for the model's decisions?
plot_importance(final_model, ax=axes[0], max_num_features=15, importance_type="gain")
axes[0].set_title("Feature Importance (Gain)")

# Plot 2: Scatter plot of predictions vs actual values (Perfect model = straight diagonal line)
axes[1].scatter(test_df[TARGET], test_preds, alpha=0.3, s=5, color="steelblue")
lims = [min(test_df[TARGET].min(), test_preds.min()), max(test_df[TARGET].max(), test_preds.max())]
axes[1].plot(lims, lims, "r--", linewidth=1)
axes[1].set_xlabel("Actual PM2.5")
axes[1].set_ylabel("Predicted PM2.5")
axes[1].set_title("Predicted vs Actual (Test Set)")

# Plot 3: Histogram of errors. A narrow peak at 0 means the model is accurate.
residuals = test_df[TARGET].values - test_preds
axes[2].hist(residuals, bins=60, color="coral", edgecolor="white")
axes[2].axvline(0, color="black", linewidth=1)
axes[2].set_xlabel("Residual (Actual − Predicted)")
axes[2].set_title("Residual Distribution (Test Set)")

plt.tight_layout()
plt.savefig(BASE_DIR / "pm25_model_results.png", dpi=150)
plt.show()
print("\nDone.")
