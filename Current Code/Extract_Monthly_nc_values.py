from pathlib import Path
import re

try:
    import pandas as pd
    import xarray as xr
except ImportError as exc:
    raise SystemExit(
        "Required packages for Extract_Monthly_nc_values.py are not installed. "
        "Install the project requirements first, then rerun this script."
    ) from exc


#Paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data/raw"
PROCESSED_DIR = BASE_DIR / "data/processed"
OUT_FILE = PROCESSED_DIR / "na_pm25_cells_clean.csv"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

#North America bounds
LAT_MIN, LAT_MAX = 7.0, 84.0
LON_MIN, LON_MAX = -168.0, -52.0

#Downsample grid
LAT_STRIDE = 10
LON_STRIDE = 10


def parse_date(filename):
    match = re.search(r'_(\d{6})_', filename)
    if match is None:
        raise ValueError(f"Cannot extract YYYYMM from filename: {filename}")
    return pd.to_datetime(match.group(1), format="%Y%m")


def extract_values(nc_file):
    #Open one monthly GHAP file
    ds = xr.open_dataset(nc_file)

    #Detect the main PM2.5 field
    var = list(ds.data_vars)[0]
    da = ds[var]

    #Downsample the grid before converting to a DataFrame
    da = da.isel(
        lat=slice(None, None, LAT_STRIDE),
        lon=slice(None, None, LON_STRIDE)
    )

    #Keep only the North America study domain
    da = da.where(
        (da.lat >= LAT_MIN) &
        (da.lat <= LAT_MAX) &
        (da.lon >= LON_MIN) &
        (da.lon <= LON_MAX),
        drop=True
    )

    df = da.to_dataframe().reset_index()

    df.rename(columns={var: "pm25"}, inplace=True)

    df["date"] = parse_date(nc_file.name)

    #Drop empty PM2.5 cells before stacking all months
    df = df.dropna(subset=["pm25"])

    return df


def main():
    #Use only the monthly GHAP files that feed the modeling panel
    files = sorted(DATA_DIR.glob("GHAP_PM2.5_M1K_*.nc"))
    if not files:
        raise SystemExit(
            "No monthly GHAP NetCDF files were found in data/raw. "
            "Place the GHAP_PM2.5_M1K_*.nc files in data/raw and rerun this script."
        )

    print("Monthly files found:", len(files))

    all_dfs = []

    for f in files:
        #Extract one month at a time to keep the logic easy to follow
        print("Processing:", f.name)
        df = extract_values(f)
        all_dfs.append(df)

    #Stack all months into the main modeling table
    full_df = pd.concat(all_dfs, ignore_index=True)

    full_df.to_csv(OUT_FILE, index=False)

    print("\nSaved cleaned dataset:", OUT_FILE)
    print("Total rows:", len(full_df))
    print(full_df.head())


if __name__ == "__main__":
    main()
