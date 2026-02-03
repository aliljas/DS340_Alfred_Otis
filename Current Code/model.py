from pathlib import Path
import re
import numpy as np
import pandas as pd
import xarray as xr

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# =========================
# CONFIG
# =========================
DATA_DIR = Path("Data/raw_nc")

# Cache NA means so we don't recompute every run
CACHE_PATH = Path("Data/na_means_na_cache.csv")

# North America bounding box
LAT_MIN, LAT_MAX = 7.0, 84.0
LON_MIN, LON_MAX = -168.0, -52.0

# Speed knobs (smaller stride = more accurate, slower)
LAT_STRIDE = 20
LON_STRIDE = 20

RIDGE_ALPHA = 1.0

# =========================
# HELPERS
# =========================
def parse_yyyymm_from_name(fname: str) -> pd.Timestamp:
    m = re.search(r"_(\d{6})_", fname)
    if not m:
        raise ValueError(f"Could not parse YYYYMM from: {fname}")
    return pd.to_datetime(m.group(1), format="%Y%m")


def load_split_csv(csv_path: str, split_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "File Path" not in df.columns:
        raise ValueError(f"{csv_path} missing 'File Path' column")
    out = df.copy()
    out["split"] = split_name
    out["filename"] = out["File Path"].astype(str).apply(lambda s: Path(s).name)
    out["date"] = out["filename"].apply(parse_yyyymm_from_name)
    out["local_path"] = out["filename"].apply(lambda f: str(DATA_DIR / f))
    return out[["split", "date", "filename", "local_path"]]


def na_monthly_mean(path: Path) -> float:
    """
    Compute North America mean PM2.5 for one .nc file.
    Uses stride downsampling + safe masking.
    Returns NaN if subset is empty.
    """
    ds = xr.open_dataset(path, engine="netcdf4")

    if "PM2.5" not in ds.data_vars:
        raise RuntimeError(f"'PM2.5' variable not found in {path}. Found: {list(ds.data_vars)}")

    da = ds["PM2.5"]

    # Downsample first (major speedup)
    da = da.isel(lat=slice(None, None, LAT_STRIDE),
                 lon=slice(None, None, LON_STRIDE))

    # Mask to North America
    da = da.where((da["lat"] >= LAT_MIN) & (da["lat"] <= LAT_MAX), drop=True)
    da = da.where((da["lon"] >= LON_MIN) & (da["lon"] <= LON_MAX), drop=True)

    # If empty, return NaN
    if da.sizes.get("lat", 0) == 0 or da.sizes.get("lon", 0) == 0:
        return np.nan

    # Mean over lat/lon; skip NaNs
    return float(da.mean(dim=("lat", "lon"), skipna=True).values)


def add_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    month = df["date"].dt.month.astype(int)
    df["sin_month"] = np.sin(2 * np.pi * month / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * month / 12.0)
    return df


def eval_split(name: str, y_true: np.ndarray, y_pred: np.ndarray):
    """
    Robust evaluation:
    - RMSE always computed
    - R² only computed if n>=2 (otherwise it can throw / be undefined)
    """
    n = len(y_true)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    print(f"{name} N:    {n}")
    print(f"{name} RMSE: {rmse:.3f}")

    if n >= 2:
        r2 = r2_score(y_true, y_pred)
        print(f"{name} R²:   {r2:.3f}")
    else:
        print(f"{name} R²:   (undefined for n<2)")


# =========================
# MAIN
# =========================
def main():
    Path("Data").mkdir(exist_ok=True)

    # Load your split lists
    train_df = load_split_csv("train_files.csv", "train")
    val_df   = load_split_csv("val_files.csv",   "val")
    test_df  = load_split_csv("test_files.csv",  "test")

    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    all_df = all_df.sort_values("date").reset_index(drop=True)

    print("Files per split (from CSVs):")
    print(all_df["split"].value_counts().to_string())

    # -------------------------
    # Load or build NA mean cache
    # -------------------------
    cache = None
    if CACHE_PATH.exists():
        cache = pd.read_csv(CACHE_PATH, parse_dates=["date"])
        # Only use cache if it matches current stride & bbox config
        # (simple check: required columns exist)
        required_cols = {"filename", "date", "na_mean_pm25"}
        if not required_cols.issubset(set(cache.columns)):
            cache = None

    # Compute missing means (or all if no cache)
    cached_map = {}
    if cache is not None:
        cached_map = dict(zip(cache["filename"], cache["na_mean_pm25"]))

    means = []
    missing_files = 0
    computed_now = 0

    print(f"\nFound {len(all_df)} files in your split lists.")
    for _, row in all_df.iterrows():
        fname = row["filename"]
        p = Path(row["local_path"])

        if not p.exists():
            print(f"⚠️ Missing local file: {p}")
            means.append(np.nan)
            missing_files += 1
            continue

        if fname in cached_map and pd.notna(cached_map[fname]):
            means.append(float(cached_map[fname]))
            continue

        print(f"Computing NA mean for {fname} ...")
        try:
            m = na_monthly_mean(p)
            means.append(m)
            computed_now += 1
        except Exception as e:
            print(f"⚠️ Failed on {fname}: {e}")
            means.append(np.nan)

    all_df["na_mean_pm25"] = means
    all_df = all_df.dropna(subset=["na_mean_pm25"]).reset_index(drop=True)

    print(f"\nValid NA means: {len(all_df)} (missing files: {missing_files}, newly computed: {computed_now})")

    # Update cache
    cache_out = all_df[["filename", "date", "na_mean_pm25"]].drop_duplicates("filename")
    cache_out.to_csv(CACHE_PATH, index=False)
    print(f"Cache saved: {CACHE_PATH}")

    if len(all_df) < 6:
        raise RuntimeError("Too few valid months after NA mean computation.")

    # -------------------------
    # Build forecasting dataset: predict current month from previous month + seasonality
    # -------------------------
    all_df = add_seasonality(all_df)

    # IMPORTANT: lag is chronological by date across all data
    all_df = all_df.sort_values("date").reset_index(drop=True)
    all_df["na_mean_prev"] = all_df["na_mean_pm25"].shift(1)

    feat_df = all_df.dropna(subset=["na_mean_prev"]).reset_index(drop=True)

    X = feat_df[["na_mean_prev", "sin_month", "cos_month"]].values
    y = feat_df["na_mean_pm25"].values
    split = feat_df["split"].values

    train_mask = split == "train"
    val_mask   = split == "val"
    test_mask  = split == "test"

    print("\nSamples by split (after lag):")
    print("train:", int(train_mask.sum()))
    print("val:  ", int(val_mask.sum()))
    print("test: ", int(test_mask.sum()))

    if train_mask.sum() == 0:
        raise RuntimeError(
            "No TRAIN samples after lag creation.\n"
            "If your split CSVs were randomly shuffled, lagging can reduce train rows.\n"
            "Fix options: build lag within each split, or use chronological year split."
        )

    # -------------------------
    # Train + Evaluate
    # -------------------------
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(X[train_mask], y[train_mask])

    print("\n--- Performance ---")
    eval_split("Train", y[train_mask], model.predict(X[train_mask]))

    if val_mask.sum() > 0:
        eval_split("Val", y[val_mask], model.predict(X[val_mask]))
    else:
        print("Val: no samples (mask empty)")

    if test_mask.sum() > 0:
        eval_split("Test", y[test_mask], model.predict(X[test_mask]))
    else:
        print("Test: no samples (mask empty)")

    print("\nCoefficients [prev, sin, cos]:", model.coef_)
    print("Intercept:", model.intercept_)

    # -------------------------
    # Save outputs
    # -------------------------
    out = feat_df[["date", "split", "filename", "na_mean_prev", "sin_month", "cos_month", "na_mean_pm25"]].copy()
    out["pred_pm25"] = model.predict(X)
    out_path = Path("Data/na_mean_forecast_results.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
