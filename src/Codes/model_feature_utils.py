from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


#Shared constants
TARGET = "pm25"
ERA5_FILE_PATTERNS = ("meteorlogy*.nc", "meteorology*.nc", "era5*.nc")
ERA5_ANOMALY_FEATURES = [
    "era5_temp_c_anom",
    "era5_pressure_kpa_anom",
    "era5_precip_log1p_anom",
    "era5_cloud_cover_anom",
    "era5_wind_speed_anom",
    "era5_rel_humidity_anom",
    "era5_temp_dew_gap_c_anom",
    "era5_stagnation_anom",
]
ERA5_INTERACTION_FEATURES = [
    "wx_stagnation_x_lag1",
    "wx_pressure_x_lag1",
    "wx_wind_x_lag1",
    "wx_precip_x_lag1",
    "wx_humidity_x_lag1",
    "wx_tempgap_x_lag1",
    "wx_stagnation_x_dev30",
    "wx_wind_x_dev30",
    "wx_precip_x_dev30",
    "wx_temp_x_yoy",
]
ERA5_EXTENDED_FEATURES = [
    "era5_temp_c_anom_lag1",
    "era5_precip_log1p_anom_lag1",
    "era5_wind_speed_anom_lag1",
    "era5_stagnation_anom_lag1",
    "era5_smoke_trap_index",
    "era5_fire_weather_index",
    "era5_smoke_trap_lag1",
    "era5_fire_weather_lag1",
    "era5_smoke_trap_roll3",
    "era5_fire_weather_roll3",
    "wx_smoke_trap_x_lag1",
    "wx_fire_weather_x_lag1",
    "wx_smoke_trap_x_dev30",
]


#Cell sampling
def maybe_sample_cells(frame, sample_cell_count=None, random_seed=42):
    if sample_cell_count is None:
        print("Sampling: full dataset")
        return frame

    unique_cells = frame[["lat", "lon"]].drop_duplicates()
    sample_n = min(sample_cell_count, len(unique_cells))
    sampled_cells = unique_cells.sample(n=sample_n, random_state=random_seed)

    sampled_frame = frame.merge(sampled_cells.assign(_keep=1), on=["lat", "lon"], how="inner")
    sampled_frame = sampled_frame.drop(columns="_keep")

    print(f"Sampling: {sample_n:,} cells -> {len(sampled_frame):,} rows")
    return sampled_frame


#PM history features
def add_history_features(frame, target=TARGET):
    grp = frame.groupby(["lat", "lon"], sort=False)[target]
    lagged = grp.shift(1)

    frame["month"] = frame["date"].dt.month.astype(np.int8)
    frame["year"] = frame["date"].dt.year.astype(np.int16)
    frame["dayofyear"] = frame["date"].dt.dayofyear.astype(np.int16)
    frame["time_index"] = (
        (frame["year"] - frame["year"].min()) * 12 + frame["month"]
    ).astype(np.int16)

    frame["month_sin"] = np.sin(2 * np.pi * frame["month"] / 12).astype(np.float32)
    frame["month_cos"] = np.cos(2 * np.pi * frame["month"] / 12).astype(np.float32)
    frame["doy_sin"] = np.sin(2 * np.pi * frame["dayofyear"] / 365).astype(np.float32)
    frame["doy_cos"] = np.cos(2 * np.pi * frame["dayofyear"] / 365).astype(np.float32)

    for lag in [1, 2, 3, 6, 7, 12, 24]:
        frame[f"pm25_lag{lag}"] = grp.shift(lag).astype(np.float32)

    for win in [3, 6, 7, 12, 30]:
        frame[f"pm25_roll{win}_mean"] = lagged.transform(
            lambda x, window=win: x.rolling(window, min_periods=window).mean()
        ).astype(np.float32)

    for win in [3, 6, 7, 12]:
        frame[f"pm25_roll{win}_std"] = lagged.transform(
            lambda x, window=win: x.rolling(window, min_periods=window).std()
        ).astype(np.float32)

    frame["pm25_roll12_max"] = lagged.transform(
        lambda x: x.rolling(12, min_periods=12).max()
    ).astype(np.float32)
    frame["pm25_roll12_min"] = lagged.transform(
        lambda x: x.rolling(12, min_periods=12).min()
    ).astype(np.float32)
    frame["pm25_ewm7"] = lagged.transform(
        lambda x: x.ewm(span=7, min_periods=7, adjust=False).mean()
    ).astype(np.float32)

    frame["pm25_diff1"] = (frame["pm25_lag1"] - frame["pm25_lag2"]).astype(np.float32)
    frame["pm25_diff2"] = (frame["pm25_lag2"] - frame["pm25_lag3"]).astype(np.float32)
    frame["pm25_vs_roll3"] = (frame["pm25_lag1"] - frame["pm25_roll3_mean"]).astype(np.float32)
    frame["pm25_vs_roll6"] = (frame["pm25_lag1"] - frame["pm25_roll6_mean"]).astype(np.float32)
    frame["pm25_deviation_7"] = (frame["pm25_lag1"] - frame["pm25_roll7_mean"]).astype(np.float32)
    frame["pm25_deviation_30"] = (frame["pm25_lag1"] - frame["pm25_roll30_mean"]).astype(np.float32)
    frame["pm25_ratio_7"] = (
        frame["pm25_lag1"] / (frame["pm25_roll7_mean"] + 1e-3)
    ).astype(np.float32)
    frame["pm25_yoy_change"] = (frame["pm25_lag1"] - frame["pm25_lag12"]).astype(np.float32)
    frame["pm25_roll12_range"] = (
        frame["pm25_roll12_max"] - frame["pm25_roll12_min"]
    ).astype(np.float32)
    frame["pm25_short_long_gap"] = (
        frame["pm25_roll3_mean"] - frame["pm25_roll12_mean"]
    ).astype(np.float32)
    frame["pm25_mid_long_gap"] = (
        frame["pm25_roll6_mean"] - frame["pm25_roll12_mean"]
    ).astype(np.float32)
    frame["pm25_zscore_12"] = (
        (frame["pm25_lag1"] - frame["pm25_roll12_mean"]) / (frame["pm25_roll12_std"] + 1e-3)
    ).astype(np.float32)

    return frame


#Train-only climatology
def add_train_only_climatology(frame, train_end, target=TARGET):
    frame["region_lat_bin"] = np.round(frame["lat"] / 5).astype(np.int16)
    frame["region_lon_bin"] = np.round(frame["lon"] / 5).astype(np.int16)

    train_hist = frame.loc[frame["date"] < train_end, ["lat", "lon", "month", target]].copy()
    region_hist = frame.loc[
        frame["date"] < train_end,
        ["region_lat_bin", "region_lon_bin", "month", target],
    ].copy()

    cell_stats = (
        train_hist.groupby(["lat", "lon"], as_index=False)[target]
        .agg(
            cell_pm25_mean_train="mean",
            cell_pm25_std_train="std",
            cell_pm25_median_train="median",
        )
    )

    cell_month_stats = (
        train_hist.groupby(["lat", "lon", "month"], as_index=False)[target]
        .agg(
            cell_month_mean_train="mean",
            cell_month_median_train="median",
            cell_month_std_train="std",
        )
    )

    region_month_stats = (
        region_hist.groupby(["region_lat_bin", "region_lon_bin", "month"], as_index=False)[target]
        .agg(
            region_month_mean_train="mean",
            region_month_std_train="std",
        )
    )

    frame = frame.merge(cell_stats, on=["lat", "lon"], how="left")
    frame = frame.merge(cell_month_stats, on=["lat", "lon", "month"], how="left")
    frame = frame.merge(
        region_month_stats,
        on=["region_lat_bin", "region_lon_bin", "month"],
        how="left",
    )

    climatology_cols = [
        "cell_pm25_mean_train",
        "cell_pm25_std_train",
        "cell_pm25_median_train",
        "cell_month_mean_train",
        "cell_month_median_train",
        "cell_month_std_train",
        "region_month_mean_train",
        "region_month_std_train",
    ]
    frame[climatology_cols] = frame[climatology_cols].astype(np.float32)

    return frame


#Regional anomaly features
def add_experimental_features(frame):
    frame["cell_vs_region_month"] = (
        frame["cell_month_mean_train"] - frame["region_month_mean_train"]
    ).astype(np.float32)
    frame["pm25_region_zscore"] = (
        (frame["pm25_lag1"] - frame["region_month_mean_train"]) / (frame["region_month_std_train"] + 1e-3)
    ).astype(np.float32)
    return frame


#ERA5 file discovery
def find_era5_files(raw_dir, file_patterns=ERA5_FILE_PATTERNS):
    raw_dir = Path(raw_dir)
    paths = {}
    for pattern in file_patterns:
        for path in raw_dir.glob(pattern):
            if path.is_file() and not path.name.startswith("GHAP_"):
                paths[path.resolve()] = path
    return sorted(paths.values())


#Humidity helper
def compute_relative_humidity(temp_c, dewpoint_c):
    temp_term = (17.625 * temp_c) / (243.04 + temp_c)
    dew_term = (17.625 * dewpoint_c) / (243.04 + dewpoint_c)
    rh = 100.0 * np.exp(dew_term - temp_term)
    return np.clip(rh, 0.0, 100.0).astype(np.float32)


#ERA5 anomaly and interaction features
def add_era5_features(frame, raw_dir, train_end, use_era5=True):
    if not use_era5:
        print("ERA5 meteorology disabled via model-specific *_USE_ERA5=0")
        return frame, []

    #Find matching ERA5 files
    era5_files = find_era5_files(raw_dir)
    if not era5_files:
        raise SystemExit(
            "ERA5 meteorology files were not found in data/raw. "
            "Place the ERA5 NetCDF files in data/raw "
            "(for example meteorlogy-data.nc and meteorlogy-data-other.nc), "
            "or set the model-specific *_USE_ERA5=0 flag to run without ERA5."
        )

    print("\n--- ADDING ERA5 METEOROLOGY ---")
    print("ERA5 files:")
    for path in era5_files:
        print(" -", path.name)

    #Read the shared grid and time axis once
    with xr.open_dataset(era5_files[0]) as ref_ds:
        era5_time = pd.Index(pd.to_datetime(ref_ds["valid_time"].values).to_period("M").to_timestamp())
        lat_values = ref_ds["latitude"].values
        lon_values = ref_ds["longitude"].values

    month_dates = frame["date"].dt.to_period("M").dt.to_timestamp()
    time_idx = era5_time.get_indexer(month_dates)
    if (time_idx < 0).any():
        missing_dates = month_dates[time_idx < 0].drop_duplicates().tolist()
        raise ValueError(f"Missing ERA5 data for dates: {missing_dates[:5]}")

    #Map each PM2.5 row to its nearest ERA5 grid cell
    lat_step = float(abs(lat_values[1] - lat_values[0]))
    lon_step = float(abs(lon_values[1] - lon_values[0]))
    lat_idx = np.rint((lat_values[0] - frame["lat"].to_numpy()) / lat_step).astype(np.int32)
    lon_idx = np.rint((frame["lon"].to_numpy() - lon_values[0]) / lon_step).astype(np.int32)
    lat_idx = np.clip(lat_idx, 0, len(lat_values) - 1)
    lon_idx = np.clip(lon_idx, 0, len(lon_values) - 1)
    frame["era5_lat_idx"] = lat_idx
    frame["era5_lon_idx"] = lon_idx

    raw_name_map = {
        "u10": "era5_u10",
        "v10": "era5_v10",
        "d2m": "era5_dewpoint_k",
        "t2m": "era5_temp_k",
        "sp": "era5_surface_pressure_pa",
        "tcc": "era5_cloud_cover",
        "tp": "era5_total_precip_m",
    }

    #Pull monthly ERA5 values from each file
    for path in era5_files:
        with xr.open_dataset(path) as ds:
            file_time = pd.Index(pd.to_datetime(ds["valid_time"].values).to_period("M").to_timestamp())
            if not file_time.equals(era5_time):
                raise ValueError(f"ERA5 time axis mismatch in {path.name}")

            for raw_name, feature_name in raw_name_map.items():
                if raw_name not in ds.data_vars:
                    continue
                values = ds[raw_name].values
                frame[feature_name] = values[time_idx, lat_idx, lon_idx].astype(np.float32)
                del values

    #Convert raw ERA5 variables into model-friendly weather fields
    frame["era5_temp_c"] = (frame["era5_temp_k"] - 273.15).astype(np.float32)
    frame["era5_dewpoint_c"] = (frame["era5_dewpoint_k"] - 273.15).astype(np.float32)
    frame["era5_temp_dew_gap_c"] = (
        frame["era5_temp_c"] - frame["era5_dewpoint_c"]
    ).astype(np.float32)
    frame["era5_pressure_kpa"] = (frame["era5_surface_pressure_pa"] / 1000.0).astype(np.float32)
    frame["era5_precip_log1p"] = np.log1p(frame["era5_total_precip_m"] * 1000.0).astype(
        np.float32
    )
    frame["era5_wind_speed"] = np.sqrt(
        frame["era5_u10"] ** 2 + frame["era5_v10"] ** 2
    ).astype(np.float32)
    frame["era5_rel_humidity"] = compute_relative_humidity(
        frame["era5_temp_c"].to_numpy(),
        frame["era5_dewpoint_c"].to_numpy(),
    )
    frame["era5_cloud_cover"] = frame["era5_cloud_cover"].astype(np.float32)
    frame["era5_stagnation"] = (
        frame["era5_pressure_kpa"] / (frame["era5_wind_speed"] + 0.5)
    ).astype(np.float32)

    weather_base_features = [
        "era5_temp_c",
        "era5_pressure_kpa",
        "era5_precip_log1p",
        "era5_cloud_cover",
        "era5_wind_speed",
        "era5_rel_humidity",
        "era5_temp_dew_gap_c",
        "era5_stagnation",
    ]

    #Build train-only monthly weather normals on the ERA5 grid
    weather_stats = (
        frame.loc[frame["date"] < train_end, ["era5_lat_idx", "era5_lon_idx", "month"] + weather_base_features]
        .groupby(["era5_lat_idx", "era5_lon_idx", "month"], as_index=False)
        .agg(
            era5_temp_c_month_mean_train=("era5_temp_c", "mean"),
            era5_pressure_kpa_month_mean_train=("era5_pressure_kpa", "mean"),
            era5_precip_log1p_month_mean_train=("era5_precip_log1p", "mean"),
            era5_cloud_cover_month_mean_train=("era5_cloud_cover", "mean"),
            era5_wind_speed_month_mean_train=("era5_wind_speed", "mean"),
            era5_rel_humidity_month_mean_train=("era5_rel_humidity", "mean"),
            era5_temp_dew_gap_c_month_mean_train=("era5_temp_dew_gap_c", "mean"),
            era5_stagnation_month_mean_train=("era5_stagnation", "mean"),
        )
    )
    frame = frame.merge(
        weather_stats,
        on=["era5_lat_idx", "era5_lon_idx", "month"],
        how="left",
    )

    #Turn same-month weather into anomaly features
    for base_name in weather_base_features:
        mean_col = f"{base_name}_month_mean_train"
        frame[f"{base_name}_anom"] = (frame[base_name] - frame[mean_col]).astype(np.float32)

    #Build weather interactions and persistence-style features
    weather_group = frame.groupby(["lat", "lon"], sort=False)
    era5_extra_cols = {
        "wx_stagnation_x_lag1": frame["era5_stagnation_anom"] * frame["pm25_lag1"],
        "wx_pressure_x_lag1": frame["era5_pressure_kpa_anom"] * frame["pm25_lag1"],
        "wx_wind_x_lag1": frame["era5_wind_speed_anom"] * frame["pm25_lag1"],
        "wx_precip_x_lag1": frame["era5_precip_log1p_anom"] * frame["pm25_lag1"],
        "wx_humidity_x_lag1": frame["era5_rel_humidity_anom"] * frame["pm25_lag1"],
        "wx_tempgap_x_lag1": frame["era5_temp_dew_gap_c_anom"] * frame["pm25_lag1"],
        "wx_stagnation_x_dev30": frame["era5_stagnation_anom"] * frame["pm25_deviation_30"],
        "wx_wind_x_dev30": frame["era5_wind_speed_anom"] * frame["pm25_deviation_30"],
        "wx_precip_x_dev30": frame["era5_precip_log1p_anom"] * frame["pm25_deviation_30"],
        "wx_temp_x_yoy": frame["era5_temp_c_anom"] * frame["pm25_yoy_change"],
        "era5_temp_c_anom_lag1": weather_group["era5_temp_c_anom"].shift(1),
        "era5_precip_log1p_anom_lag1": weather_group["era5_precip_log1p_anom"].shift(1),
        "era5_wind_speed_anom_lag1": weather_group["era5_wind_speed_anom"].shift(1),
        "era5_stagnation_anom_lag1": weather_group["era5_stagnation_anom"].shift(1),
        "era5_smoke_trap_index": (
            frame["era5_stagnation_anom"]
            - frame["era5_wind_speed_anom"]
            - frame["era5_precip_log1p_anom"]
        ),
        "era5_fire_weather_index": (
            frame["era5_temp_c_anom"]
            + frame["era5_stagnation_anom"]
            - frame["era5_precip_log1p_anom"]
        ),
    }
    era5_extra_df = pd.DataFrame(era5_extra_cols, index=frame.index).astype(np.float32)
    smoke_trap_group = era5_extra_df.groupby([frame["lat"], frame["lon"]], sort=False)["era5_smoke_trap_index"]
    fire_weather_group = era5_extra_df.groupby([frame["lat"], frame["lon"]], sort=False)["era5_fire_weather_index"]
    era5_extra_df["era5_smoke_trap_lag1"] = smoke_trap_group.shift(1).astype(np.float32)
    era5_extra_df["era5_fire_weather_lag1"] = fire_weather_group.shift(1).astype(np.float32)
    era5_extra_df["era5_smoke_trap_roll3"] = smoke_trap_group.transform(
        lambda x: x.shift(1).rolling(3, min_periods=3).mean()
    ).astype(np.float32)
    era5_extra_df["era5_fire_weather_roll3"] = fire_weather_group.transform(
        lambda x: x.shift(1).rolling(3, min_periods=3).mean()
    ).astype(np.float32)
    era5_extra_df["wx_smoke_trap_x_lag1"] = (
        era5_extra_df["era5_smoke_trap_index"] * frame["pm25_lag1"]
    ).astype(np.float32)
    era5_extra_df["wx_fire_weather_x_lag1"] = (
        era5_extra_df["era5_fire_weather_index"] * frame["pm25_lag1"]
    ).astype(np.float32)
    era5_extra_df["wx_smoke_trap_x_dev30"] = (
        era5_extra_df["era5_smoke_trap_index"] * frame["pm25_deviation_30"]
    ).astype(np.float32)
    frame = pd.concat([frame, era5_extra_df], axis=1)

    #Drop temporary raw columns once engineered features are ready
    drop_cols = [
        "era5_u10",
        "era5_v10",
        "era5_dewpoint_c",
        "era5_temp_k",
        "era5_dewpoint_k",
        "era5_surface_pressure_pa",
        "era5_total_precip_m",
        "era5_lat_idx",
        "era5_lon_idx",
        "era5_temp_c",
        "era5_pressure_kpa",
        "era5_precip_log1p",
        "era5_cloud_cover",
        "era5_wind_speed",
        "era5_rel_humidity",
        "era5_temp_dew_gap_c",
        "era5_stagnation",
        "era5_temp_c_month_mean_train",
        "era5_pressure_kpa_month_mean_train",
        "era5_precip_log1p_month_mean_train",
        "era5_cloud_cover_month_mean_train",
        "era5_wind_speed_month_mean_train",
        "era5_rel_humidity_month_mean_train",
        "era5_temp_dew_gap_c_month_mean_train",
        "era5_stagnation_month_mean_train",
    ]
    frame = frame.drop(columns=drop_cols)

    added_features = ERA5_ANOMALY_FEATURES + ERA5_INTERACTION_FEATURES + ERA5_EXTENDED_FEATURES
    print(f"Added ERA5 anomaly/interaction/persistence features: {', '.join(added_features)}")
    return frame, added_features


#Feature set definitions
def build_feature_sets(era5_feature_names=None):
    era5_feature_names = list(era5_feature_names or [])
    core_era5_features = [
        feature
        for feature in era5_feature_names
        if feature not in ERA5_EXTENDED_FEATURES
    ]
    base_features = [
        "lat",
        "lon",
        "month",
        "region_lat_bin",
        "region_lon_bin",
        "time_index",
        "month_sin",
        "month_cos",
        "doy_sin",
        "doy_cos",
        "pm25_lag1",
        "pm25_lag2",
        "pm25_lag3",
        "pm25_lag6",
        "pm25_lag7",
        "pm25_lag12",
        "pm25_lag24",
        "pm25_roll3_mean",
        "pm25_roll6_mean",
        "pm25_roll7_mean",
        "pm25_roll12_mean",
        "pm25_roll30_mean",
        "pm25_roll3_std",
        "pm25_roll6_std",
        "pm25_roll7_std",
        "pm25_roll12_std",
        "pm25_ewm7",
        "pm25_diff1",
        "pm25_diff2",
        "pm25_vs_roll3",
        "pm25_vs_roll6",
        "pm25_deviation_7",
        "pm25_deviation_30",
        "pm25_ratio_7",
        "pm25_yoy_change",
        "cell_pm25_mean_train",
        "cell_pm25_std_train",
        "cell_pm25_median_train",
        "cell_month_mean_train",
        "cell_month_std_train",
    ]

    if core_era5_features:
        base_features = base_features + core_era5_features

    #Keep separate sets so we can compare core ERA5 features against the
    #larger persistence-heavy batch.
    feature_sets = {
        "base": base_features,
        "trend": base_features
        + [
            "cell_month_median_train",
            "pm25_roll12_max",
            "pm25_roll12_min",
            "pm25_roll12_range",
            "pm25_short_long_gap",
            "pm25_mid_long_gap",
            "pm25_zscore_12",
        ],
        "region": base_features
        + [
            "cell_month_median_train",
            "region_month_mean_train",
            "region_month_std_train",
            "cell_vs_region_month",
            "pm25_region_zscore",
        ],
        "trend_region": base_features
        + [
            "cell_month_median_train",
            "pm25_roll12_max",
            "pm25_roll12_min",
            "pm25_roll12_range",
            "pm25_short_long_gap",
            "pm25_mid_long_gap",
            "pm25_zscore_12",
            "region_month_mean_train",
            "region_month_std_train",
            "cell_vs_region_month",
            "pm25_region_zscore",
        ],
        "trend_region_era5plus": base_features
        + [
            "cell_month_median_train",
            "pm25_roll12_max",
            "pm25_roll12_min",
            "pm25_roll12_range",
            "pm25_short_long_gap",
            "pm25_mid_long_gap",
            "pm25_zscore_12",
            "region_month_mean_train",
            "region_month_std_train",
            "cell_vs_region_month",
            "pm25_region_zscore",
        ]
        + [feature for feature in ERA5_EXTENDED_FEATURES if feature in era5_feature_names],
        "compact_era5": [
            "lat",
            "lon",
            "month",
            "time_index",
            "month_sin",
            "month_cos",
            "pm25_lag1",
            "pm25_lag12",
            "pm25_lag24",
            "pm25_deviation_30",
            "pm25_yoy_change",
            "cell_month_mean_train",
            "cell_month_std_train",
            "cell_month_median_train",
            "region_month_std_train",
            "pm25_region_zscore",
        ] + list(era5_feature_names or []),
    }
    return feature_sets


#End-to-end modeling table
def prepare_modeling_frame(
    data_file,
    raw_dir,
    train_end,
    sample_cell_count=None,
    random_seed=42,
    use_era5=True,
    target=TARGET,
):
    print("\n--- LOADING DATA ---")
    data_file = Path(data_file)
    raw_dir = Path(raw_dir)

    #Fail early with a clear setup message when the expected project files
    #have not been placed in the standard folders yet.
    if not data_file.exists():
        raise SystemExit(
            "The modeling table data/processed/na_pm25_cells_clean.csv was not found. "
            "Place the raw GHAP NetCDF files in data/raw and run "
            "src/Codes/Extract_Monthly_nc_values.py first."
        )

    #Load and normalize the base PM2.5 panel
    frame = pd.read_csv(data_file)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["lat", "lon", "date"]).reset_index(drop=True)
    frame[["lat", "lon", target]] = frame[["lat", "lon", target]].astype(np.float32)
    frame = maybe_sample_cells(frame, sample_cell_count=sample_cell_count, random_seed=random_seed)

    print("Rows:", len(frame))
    print("\n--- FEATURE ENGINEERING ---")
    #Build the shared feature pipeline used by all model scripts
    frame = add_history_features(frame, target=target)
    frame = add_train_only_climatology(frame, train_end=train_end, target=target)
    frame = add_experimental_features(frame)
    frame, era5_feature_names = add_era5_features(
        frame,
        raw_dir=raw_dir,
        train_end=train_end,
        use_era5=use_era5,
    )
    frame = frame.dropna().copy()

    print("After features:", len(frame))
    return frame, era5_feature_names
