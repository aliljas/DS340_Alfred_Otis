import pandas as pd
import numpy as np
import optuna
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import matplotlib.pyplot as plt
from xgboost import plot_importance

optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- CONFIG ---
BASE_DIR = Path(r"C:\Users\user\Downloads")
DATA_FILE = BASE_DIR / "na_pm25_cells_clean.csv"
TARGET = "pm25"

FEATURES = [
    "lat", "lon",
    "pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_lag7",
    "pm25_roll3_mean", "pm25_roll7_mean", "pm25_roll30_mean",
    "pm25_roll7_std",
    "pm25_ewm7",
    "pm25_deviation_7",    # lag1 minus 7-day mean  -> captures short-term spikes
    "pm25_deviation_30",   # lag1 minus 30-day mean -> captures seasonal anomalies
    "pm25_ratio_7",        # lag1 / 7-day mean      -> relative spike magnitude
    "lat_lon_mean",        # long-run cell baseline
    "month_sin", "month_cos",
    "doy_sin", "doy_cos",
]

# --- LOAD ---
print("Loading data...")
df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])
df[["lat", "lon", "pm25"]] = df[["lat", "lon", "pm25"]].astype(np.float32)

# --- FEATURE ENGINEERING ---
print("Engineering features...")
df = df.sort_values(["lat", "lon", "date"]).reset_index(drop=True)

grp    = df.groupby(["lat", "lon"])["pm25"]
lagged = grp.shift(1)  # all rolling stats use shift(1) to prevent leakage

# Lags
for lag in [1, 2, 3, 7]:
    df[f"pm25_lag{lag}"] = grp.shift(lag)

# Rolling stats
df["pm25_roll3_mean"]  = lagged.transform(lambda x: x.rolling(3,  min_periods=1).mean())
df["pm25_roll7_mean"]  = lagged.transform(lambda x: x.rolling(7,  min_periods=1).mean())
df["pm25_roll30_mean"] = lagged.transform(lambda x: x.rolling(30, min_periods=1).mean())
df["pm25_roll7_std"]   = lagged.transform(lambda x: x.rolling(7,  min_periods=1).std().fillna(0))
df["pm25_ewm7"]        = lagged.transform(lambda x: x.ewm(span=7, min_periods=1).mean())

# KEY NEW FEATURES: deviation/ratio — tells model if current PM2.5 is anomalous
# relative to recent history. These generalize across seasons because they are
# scale-invariant (ratio) or mean-centered (deviation).
df["pm25_deviation_7"]  = df["pm25_lag1"] - df["pm25_roll7_mean"]
df["pm25_deviation_30"] = df["pm25_lag1"] - df["pm25_roll30_mean"]
df["pm25_ratio_7"]      = df["pm25_lag1"] / (df["pm25_roll7_mean"] + 1e-3)

# Long-run spatial baseline
df["lat_lon_mean"] = grp.transform("mean").astype(np.float32)

# Cyclical time
df["month"] = df["date"].dt.month
df["doy"]   = df["date"].dt.dayofyear
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).astype(np.float32)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).astype(np.float32)
df["doy_sin"]   = np.sin(2 * np.pi * df["doy"]   / 365).astype(np.float32)
df["doy_cos"]   = np.cos(2 * np.pi * df["doy"]   / 365).astype(np.float32)

df = df.dropna(subset=FEATURES)
print(f"Dataset shape: {df.shape}")

# --- SPLIT ---
# IMPORTANT: val now includes a full year so Optuna sees both winter AND summer
# patterns — this is the main reason for the val/test gap before.
train_df = df[df["date"] <  "2021-01-01"]
val_df   = df[(df["date"] >= "2021-01-01") & (df["date"] < "2022-01-01")]
test_df  = df[df["date"] >= "2022-01-01"]

print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

# --- OPTUNA ---
def objective(trial):
    param = {
        "n_estimators":          500,
        "learning_rate":         trial.suggest_float("learning_rate",    0.01,  0.2,  log=True),
        "max_depth":             trial.suggest_int(  "max_depth",        3,     7),       # capped at 7 — deeper = overfit
        "subsample":             trial.suggest_float("subsample",        0.6,   1.0),
        "colsample_bytree":      trial.suggest_float("colsample_bytree", 0.6,   1.0),
        "min_child_weight":      trial.suggest_int(  "min_child_weight", 5,     30),      # raised floor — less overfitting
        "reg_alpha":             trial.suggest_float("reg_alpha",        0.1,   10.0, log=True),
        "reg_lambda":            trial.suggest_float("reg_lambda",       0.1,   10.0, log=True),
        "gamma":                 trial.suggest_float("gamma",            0.0,   3.0),
        "objective":             "reg:squarederror",   # back to squared error — simpler and works better here
        "tree_method":           "hist",
        "early_stopping_rounds": 50,
        "random_state":          42,
        "n_jobs":                -1,
    }

    tune_df = train_df.sample(n=min(1_000_000, len(train_df)), random_state=42)
    model   = XGBRegressor(**param)
    model.fit(
        tune_df[FEATURES], tune_df[TARGET],
        eval_set=[(val_df[FEATURES], val_df[TARGET])],
        verbose=False,
    )
    preds = model.predict(val_df[FEATURES])
    return mean_squared_error(val_df[TARGET], preds) ** 0.5

print("\nStarting Hyperparameter Optimization...")
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
)
study.optimize(objective, n_trials=40, show_progress_bar=True)
print(f"\nBest Val RMSE : {study.best_value:.4f}")
print(f"Best Params   : {study.best_params}")

# --- FINAL TRAINING ---
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

final_rmse = mean_squared_error(test_df[TARGET], test_preds) ** 0.5
final_r2   = r2_score(test_df[TARGET], test_preds)
print(f"\n{'='*40}")
print(f"Final Test RMSE : {final_rmse:.4f}")
print(f"Final Test R²   : {final_r2:.4f}")
print(f"{'='*40}")

# --- PLOTS ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

plot_importance(final_model, ax=axes[0], max_num_features=15, importance_type="gain")
axes[0].set_title("Feature Importance (Gain)")

axes[1].scatter(test_df[TARGET], test_preds, alpha=0.3, s=5, color="steelblue")
lims = [min(test_df[TARGET].min(), test_preds.min()),
        max(test_df[TARGET].max(), test_preds.max())]
axes[1].plot(lims, lims, "r--", linewidth=1)
axes[1].set_xlabel("Actual PM2.5")
axes[1].set_ylabel("Predicted PM2.5")
axes[1].set_title("Predicted vs Actual (Test Set)")

residuals = test_df[TARGET].values - test_preds
axes[2].hist(residuals, bins=60, color="coral", edgecolor="white")
axes[2].axvline(0, color="black", linewidth=1)
axes[2].set_xlabel("Residual (Actual − Predicted)")
axes[2].set_title("Residual Distribution (Test Set)")

plt.tight_layout()
plt.savefig(BASE_DIR / "pm25_model_results.png", dpi=150)
plt.show()
print("\nDone.")
