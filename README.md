#  PM2.5 air pollution data processing project

This project builds a North America monthly PM2.5 panel from GHAP NetCDF files and trains multiple forecasting-style benchmark models. The current external weather input is limited to the two ERA5 monthly NetCDF files placed in `data/raw`.

##  Project structure

`data/raw/`  
Raw GHAP monthly NetCDF files and raw ERA5 NetCDF files.

`data/processed/`  
Derived modeling tables, metrics, feature importance files, and plots.

`src/Codes/`  
Preprocessing, modeling, and figure scripts.

`requirements.txt`  
Pinned Python dependencies for a clean environment.

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
Primary model script and before/after ERA5 comparison outputs.

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

Quick sample LightGBM run:

```bash
LIGHTGBM_SAMPLE_CELLS=1000 python src/Codes/LightGBM_model.py
```

Full CatBoost run:

```bash
python src/Codes/CatBoost_model.py
```

Full XGBoost run:

```bash
python src/Codes/XGB_model.py
```

Full linear baseline run:

```bash
python src/Codes/LR_model.py
```

## ERA5 notes

All four model scripts now support the shared ERA5 feature pipeline. If the ERA5 files are missing and the model-specific `*_USE_ERA5` flag is left on, the script will stop with a clear setup message instead of silently continuing.

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
Major in Computational Data Science

Otis Murray  
The Pennsylvania State University  
Major in Statistical Modeling Data Science, Minor in Mathematics
