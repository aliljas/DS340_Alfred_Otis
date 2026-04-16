# PM2.5 air pollution data processing project

This project builds a North America monthly PM2.5 panel from GHAP NetCDF files and trains multiple forecasting-style benchmark models. The current external weather input is limited to the two ERA5 monthly NetCDF files placed in `data/raw`.

## Project structure

`data/raw/`  
Raw GHAP monthly NetCDF files and raw ERA5 NetCDF files.

`data/processed/`  
Derived modeling tables, metrics, feature importance files, and plots.

`src/Codes/`  
Preprocessing, modeling, and figure scripts.

`requirements.txt`  
Pinned Python dependencies for a clean environment.

## Dataset access

This repository does not store the full raw NetCDF datasets because they are too large for normal GitHub use. The raw files should be downloaded separately and then placed in `data/raw/`.

Official dataset sources:

- GHAP monthly PM2.5 data: [https://zenodo.org/records/10800980](https://zenodo.org/records/10800980)
- ERA5 monthly averaged data on single levels: [https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means?tab=download](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means?tab=download)

Convenience folder used for this project:

- Google Drive ZIPs and paper PDFs: [https://drive.google.com/drive/folders/19ZsUzhA5A0bFTib73QfEwwCQGnO4vFwU?usp=sharing](https://drive.google.com/drive/folders/19ZsUzhA5A0bFTib73QfEwwCQGnO4vFwU?usp=sharing)

## GHAP raw data

Download the monthly GHAP NetCDF files and place them directly in:

`data/raw/`

The preprocessing script expects filenames that match:

`GHAP_PM2.5_M1K_*.nc`

Those files are the monthly PM2.5 source used to build the cleaned North America panel.

## ERA5 raw data

The model scripts expect two monthly ERA5 NetCDF files in:

`data/raw/meteorlogy-data.nc`

`data/raw/meteorlogy-data-other.nc`

The code is slightly forgiving on the exact names because it looks for `meteorlogy*.nc`, `meteorology*.nc`, or `era5*.nc`, but using the filenames above is the safest option.

### ERA5 download steps

1. Create or sign in to a Copernicus Climate Data Store account.
2. Open the ERA5 dataset page: [https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means?tab=download](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means?tab=download)
3. Choose the product for monthly averaged single-level reanalysis data.
4. Select the time range from `2017-01` through `2023-12`.
5. Use the North America area subset in CDS order:
   - North: `84`
   - West: `-168`
   - South: `7`
   - East: `-52`
6. Choose `NetCDF` as the download format.
7. Download the variables in two requests so the files stay manageable.

### ERA5 request 1

Recommended output filename:

`meteorlogy-data-other.nc`

Variables:

- `2m_temperature`
- `2m_dewpoint_temperature`
- `surface_pressure`
- `total_cloud_cover`
- `10m_u_component_of_wind`
- `10m_v_component_of_wind`

### ERA5 request 2

Recommended output filename:

`meteorlogy-data.nc`

Variables:

- `total_precipitation`

### ERA5 notes

- The downloaded files used in this project cover `2017-01` through `2023-12`.
- The first file contains `d2m`, `sp`, `t2m`, `tcc`, `u10`, and `v10`.
- The second file contains `tp`.
- These two files are enough for the shared ERA5 feature pipeline used by the model scripts.

## Portable path setup

The active scripts do not use hard-coded user-specific file paths. Each script resolves the project root from its own location with `Path(__file__).resolve().parents[2]`, so the code should work on any machine as long as the project folder structure is preserved.

## Required folder layout

Place the raw files here before running anything:

`data/raw/GHAP_PM2.5_M1K_*.nc`  
Monthly GHAP PM2.5 files.

`data/raw/meteorlogy-data.nc`  
ERA5 monthly weather file.

`data/raw/meteorlogy-data-other.nc`  
Second ERA5 monthly weather file.

The model scripts expect the derived panel here:

`data/processed/na_pm25_cells_clean.csv`

If that file does not exist yet, create it with the preprocessing script shown below.

## Environment setup

From the project root:

```bash
python -m venv venv
```

Mac/Linux activation:

```bash
source venv/bin/activate
```

Windows activation:

```bash
venv\Scripts\activate
```

Then install the project requirements:

```bash
python -m pip install -r requirements.txt
```

## Preprocessing step

If you are starting from the raw GHAP NetCDF files, first build the modeling table:

```bash
python src/Codes/Extract_Monthly_nc_values.py
```

This reads the GHAP monthly files from `data/raw/` and writes:

`data/processed/na_pm25_cells_clean.csv`

## Main model scripts

`src/Codes/LightGBM_model.py`  
Primary LightGBM benchmark script. The default run is now the forecast-ready no-ERA baseline, and ERA5 comparison mode can be enabled with environment variables.

`src/Codes/CatBoost_model.py`  
CatBoost benchmark on the shared feature pipeline.

`src/Codes/XGB_model.py`  
XGBoost benchmark on the shared feature pipeline.

`src/Codes/LR_model.py`  
Linear regression baseline on the compact feature subset.

`src/Codes/model_feature_utils.py`  
Shared feature engineering used by all model scripts.

## Run examples

Full LightGBM run:

```bash
python src/Codes/LightGBM_model.py
```

Full LightGBM run with ERA5 enabled:

```bash
LIGHTGBM_USE_ERA5=1 python src/Codes/LightGBM_model.py
```

Full LightGBM run with before/after ERA5 comparison outputs:

```bash
LIGHTGBM_USE_ERA5=1 LIGHTGBM_COMPARE_ERA5=1 python src/Codes/LightGBM_model.py
```

Full CatBoost run:

```bash
python src/Codes/CatBoost_model.py
```

Full CatBoost run with ERA5 enabled:

```bash
CATBOOST_USE_ERA5=1 python src/Codes/CatBoost_model.py
```

Full CatBoost run with before/after ERA5 comparison outputs:

```bash
CATBOOST_USE_ERA5=1 CATBOOST_COMPARE_ERA5=1 CATBOOST_EVAL_SAMPLE_ROWS=250000 python src/Codes/CatBoost_model.py
```

Full XGBoost run:

```bash
python src/Codes/XGB_model.py
```

Full ridge baseline run:

```bash
python src/Codes/LR_model.py
```

Full ridge baseline run with ERA5 enabled:

```bash
LR_USE_ERA5=1 python src/Codes/LR_model.py
```

Full ridge baseline run with before/after ERA5 comparison outputs:

```bash
LR_USE_ERA5=1 LR_COMPARE_ERA5=1 python src/Codes/LR_model.py
```

## Forecast-ready runs

For a strict recursive future forecast, use the baseline no-ERA path for each model. The scripts automatically skip forecasting in ERA5 or comparison mode so the forecast remains causally clean.

Forecast-ready baseline runs:

```bash
python src/Codes/LightGBM_model.py
```

```bash
python src/Codes/CatBoost_model.py
```

```bash
python src/Codes/XGB_model.py
```

```bash
python src/Codes/LR_model.py
```

## ERA5 notes

All four model scripts now support the shared ERA5 feature pipeline. If the ERA5 files are missing and the model-specific `*_USE_ERA5` flag is left on, the script will stop with a clear setup message instead of silently continuing.

ERA5 effects are model-dependent rather than universal. In the current experiments, ERA5 provided a modest improvement for LightGBM and improved the ridge baseline, while the strongest full CatBoost result remained the baseline model without ERA5. For that reason, `CatBoost_model.py` runs without ERA5 unless `CATBOOST_USE_ERA5=1` is set explicitly, and the comparison mode can be used when a before/after ERA5 table or figure is needed for analysis.

In other words, ERA5 is supported across the project as an additive external feature source, but its benefit should be evaluated by model rather than assumed to help every model equally.

For strict future forecasting, same-month ERA5 reanalysis should not be treated as a valid ex-ante predictor. The commented forecast blocks remain in the scripts for reference, but same-month ERA5 should stay disabled there unless forecast meteorology is available.

## Reproducibility

A TA can reproduce the intended workflow on a clean machine by:

1. cloning the repository
2. creating and activating a fresh virtual environment
3. running `python -m pip install -r requirements.txt`
4. placing the GHAP and ERA5 raw NetCDF files in `data/raw/`
5. running `python src/Codes/Extract_Monthly_nc_values.py`
6. running any model script from the project root

All active scripts include import guards with clear setup messages if required Python packages are missing.

## Authors

Alfred Liljas  
The Pennsylvania State University

Otis Murray  
The Pennsylvania State University  
Major in statistical modeling data science, minor in math
