# =========================
# Imports
# =========================
from pathlib import Path
import re
import numpy as np
import pandas as pd
import xarray as xr

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR.parent
CACHE_DIR = BASE_DIR / "Data"
CACHE_DIR.mkdir(exist_ok=True)

NA_CACHE = CACHE_DIR / "na_means_na_cache.csv"

# =========================
# North America bounds
# =========================
LAT_MIN, LAT_MAX = 7.0, 84.0
LON_MIN, LON_MAX = -168.0, -52.0

# =========================
# Helpers
# =========================
def parse_yyyymm_from_name(fname: str):
    m = re.search(r"_(\d{6})_", fname)
    if not m:
        return None
    yyyymm = m.group(1)
    return int(yyyymm[:4]), int(yyyymm[4:])

def load_split_csv(csv_path, split_name):
    df = pd.read_csv(csv_path)
    df["split"] = split_name
    return df

def compute_na_mean(nc_path):
    ds = xr.open_dataset(nc_path)
    da = ds["PM2.5"]

    da_na = da.sel(
        lat=slice(LAT_MIN, LAT_MAX),
        lon=slice(LON_MIN, LON_MAX)
    )

    val = float(da_na.mean().values)
    ds.close()
    return val

# =========================
# Main
# =========================
def main():
    # -------------------------
    # Load split CSVs
    # -------------------------
    train_df = load_split_csv(BASE_DIR / "train_files.csv", "train")
    val_df   = load_split_csv(BASE_DIR / "val_files.csv", "val")
    test_df  = load_split_csv(BASE_DIR / "test_files.csv", "test")

    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    print("\nFiles per split (from CSVs):")
    print(all_df["split"].value_counts())

    # -------------------------
    # Load or compute NA means
    # -------------------------
    if NA_CACHE.exists():
        cache = pd.read_csv(NA_CACHE)
        print(f"\nLoaded cache: {NA_CACHE}")
    else:
        rows = []
        for _, row in all_df.iterrows():
            fname = row["file"]
            nc_path = DATA_DIR / fname

            print(f"Computing NA mean for {fname} ...")
            mean_val = compute_na_mean(nc_path)

            year, month = parse_yyyymm_from_name(fname)
            rows.append({
                "file": fname,
                "year": year,
                "month": month,
                "pm25": mean_val,
                "split": row["split"]
            })

        cache = pd.DataFrame(rows)
        cache.to_csv(NA_CACHE, index=False)
        print(f"\nCache saved: {NA_CACHE}")

    # -------------------------
    # Sort + lag feature
    # -------------------------
    cache = cache.sort_values(["year", "month"]).reset_index(drop=True)
    cache["pm25_prev"] = cache["pm25"].shift(1)
    cache = cache.dropna().reset_index(drop=True)

    # -------------------------
    # Seasonal features
    # -------------------------
    cache["sin_m"] = np.sin(2 * np.pi * cache["month"] / 12)
    cache["cos_m"] = np.cos(2 * np.pi * cache["month"] / 12)

    FEATURES = ["pm25_prev", "sin_m", "cos_m"]
    TARGET = "pm25"

    # -------------------------
    # Split
    # -------------------------
    train = cache[cache["split"] == "train"]
    val   = cache[cache["split"] == "val"]
    test  = cache[cache["split"] == "test"]

    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val     = val[FEATURES], val[TARGET]
    X_test, y_test   = test[FEATURES], test[TARGET]

    print("\nSamples by split (after lag):")
    print(f"train: {len(train)}")
    print(f"val:   {len(val)}")
    print(f"test:  {len(test)}")

    # -------------------------
    # XGBoost Model
    # -------------------------
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(X_train, y_train)

    # -------------------------
    # Evaluation
    # -------------------------
    def report(name, y_true, y_pred):
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2 = r2_score(y_true, y_pred)
        print(f"{name} RMSE: {rmse:.3f}")
        print(f"{name} R²:   {r2:.3f}")

    print("\n--- XGBoost Performance ---")
    report("Train", y_train, model.predict(X_train))
    report("Val",   y_val,   model.predict(X_val))
    report("Test",  y_test,  model.predict(X_test))


if __name__ == "__main__":
    main()
