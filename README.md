# PM2.5 Forecasting Project

This repository builds a cleaned monthly North America PM2.5 panel and trains four forecasting models on that shared dataset:

- Ridge Regression
- XGBoost
- CatBoost
- LightGBM

The project also supports optional ERA5 meteorological features for before-versus-after comparison runs.

## Read This First

If you are a TA or first-time user on a clean computer, start here before doing anything else.

Use this order:

1. Download the project code from GitHub.
2. Download the release-asset data files in the `Data Sources` section below.
3. Follow the `First-Time VS Code Setup For The TA` section in this README exactly.

That beginner section explains, step by step, how to:

- install VS Code for the first time
- install a compatible Python version
- open the project folder in VS Code
- create the virtual environment
- install the requirements
- select the correct Python interpreter
- use the `Run Python File` button in VS Code

If you follow that section exactly, the TA should be able to reproduce the clean-machine workflow shown in the demo.

## Repository Layout

`data/raw/`  
Raw GHAP monthly NetCDF files and ERA5 NetCDF files.

`data/processed/`  
Cleaned modeling table, prediction outputs, evaluation metrics, feature importance files, and saved figures.

`src/Codes/`  
Preprocessing, modeling, shared utilities, and figure-generation scripts.

`requirements.txt`  
Primary dependency list for the project environment.

## Data Sources

Official sources:

- GHAP monthly PM2.5: [https://zenodo.org/records/10800980](https://zenodo.org/records/10800980)
- ERA5 monthly averaged single-level data: [https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means?tab=download](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means?tab=download)

Convenience sources used for this project:

- Google Drive folder: [https://drive.google.com/drive/folders/19ZsUzhA5A0bFTib73QfEwwCQGnO4vFwU?usp=sharing](https://drive.google.com/drive/folders/19ZsUzhA5A0bFTib73QfEwwCQGnO4vFwU?usp=sharing)
- GitHub release assets:
  - Cleaned modeling table CSV: [https://github.com/aliljas/DS340_Alfred_Otis/releases/download/project-data-v1/na_pm25_cells_clean.csv](https://github.com/aliljas/DS340_Alfred_Otis/releases/download/project-data-v1/na_pm25_cells_clean.csv)
  - ERA raw data ZIP: [https://github.com/aliljas/DS340_Alfred_Otis/releases/download/project-data-v1/ERA.Data.zip](https://github.com/aliljas/DS340_Alfred_Otis/releases/download/project-data-v1/ERA.Data.zip)

## Two Ways To Reproduce The Project

You can reproduce the project in either of these ways:

1. Use the prebuilt cleaned PM2.5 CSV and skip GHAP preprocessing.
2. Start from the raw GHAP monthly NetCDF files and rebuild the cleaned panel yourself.

Both approaches still require the ERA5 files if you want ERA5-enhanced runs.

## Option A: Fastest Setup With Final Data Files

Use this option if you want the fastest exact reproduction of the modeling workflow.

Required files:

- `data/processed/na_pm25_cells_clean.csv`
- `data/raw/meteorlogy-data.nc`
- `data/raw/meteorlogy-data-other.nc`

Setup:

1. Download `na_pm25_cells_clean.csv` from the release asset link above.
2. Place it in `data/processed/`.
3. Download `ERA.Data.zip`.
4. Extract it into `data/raw/`.

After setup, you should have:

`data/processed/na_pm25_cells_clean.csv`  
Cleaned monthly North America PM2.5 modeling table.

`data/raw/meteorlogy-data.nc`  
ERA5 monthly precipitation file.

`data/raw/meteorlogy-data-other.nc`  
ERA5 monthly weather file containing temperature, dewpoint, pressure, cloud cover, and wind variables.

## Option B: Full Rebuild From Raw GHAP Files

Use this option if you want to reproduce the cleaned PM2.5 table from the original GHAP NetCDF files.

Required raw files:

- GHAP monthly NetCDF files matching `GHAP_PM2.5_M1K_*.nc`
- `meteorlogy-data.nc`
- `meteorlogy-data-other.nc`

Place them in:

`data/raw/`

Then run the preprocessing script:

Mac/Linux:

```bash
python src/Codes/Extract_Monthly_nc_values.py
```

Windows:

```powershell
python src\Codes\Extract_Monthly_nc_values.py
```

That script writes:

`data/processed/na_pm25_cells_clean.csv`

## ERA5 Download Instructions

If you need to rebuild or replace the ERA5 files, use these settings in the Copernicus Climate Data Store.

Dataset:

- Monthly averaged ERA5 single-level reanalysis

Time range:

- `2017-01` through `2023-12`

Geographic subset in CDS order:

- North: `84`
- West: `-168`
- South: `7`
- East: `-52`

Format:

- `NetCDF`

Split the variables into two requests:

Request 1 output filename:

`meteorlogy-data-other.nc`

Variables:

- `2m_temperature`
- `2m_dewpoint_temperature`
- `surface_pressure`
- `total_cloud_cover`
- `10m_u_component_of_wind`
- `10m_v_component_of_wind`

Request 2 output filename:

`meteorlogy-data.nc`

Variables:

- `total_precipitation`

Notes:

- The code is forgiving on exact ERA filenames and looks for `meteorlogy*.nc`, `meteorology*.nc`, or `era5*.nc`.
- The filenames above are still the safest choice.

## First-Time VS Code Setup For The TA

If the TA has never used VS Code or Python on the computer before, follow these steps exactly.

1. Download Visual Studio Code from the official page: [https://code.visualstudio.com/download](https://code.visualstudio.com/download)
2. Install VS Code with the normal default installer options.
3. Download Python from the official page: [https://www.python.org/downloads/](https://www.python.org/downloads/)
4. For this project, use Python `3.12` or Python `3.13`. Do **not** use Python `3.14` for the first clean-machine run.
5. Install Python with the normal default installer options. If Windows offers an option to add Python to `PATH`, enable it.
6. Download the repository from GitHub and extract it to a normal folder.
7. Download the release assets listed above in this README.
8. Extract `na_pm25_cells_clean.csv.zip` so that this file exists in the project:

`data/processed/na_pm25_cells_clean.csv`

9. Extract `ERA.Data.zip` so that these files exist in the project:

`data/raw/meteorlogy-data.nc`

`data/raw/meteorlogy-data-other.nc`

10. Open VS Code.
11. In VS Code, choose `File` -> `Open Folder...`
12. Open the main project folder that contains this `README.md` file.
13. If VS Code asks whether you trust the authors, choose the option to trust the folder.
14. Open any Python file, for example `src/Codes/LightGBM_model.py`
15. If VS Code asks to install the Python extension, install the official Python extension from Microsoft.
16. Open a terminal inside VS Code by choosing `Terminal` -> `New Terminal`
17. Create the project virtual environment from the project root.

Mac/Linux:

```bash
python3 -m venv venv
```

Windows:

```powershell
py -m venv venv
```

18. Activate the virtual environment.

Mac/Linux:

```bash
source venv/bin/activate
```

Windows Command Prompt:

```powershell
venv\Scripts\activate
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

19. Install the project requirements:

```bash
python -m pip install -r requirements.txt
python -m pip install matplotlib
```

20. Set the Python interpreter in VS Code.
21. Look at the bottom status bar in VS Code for the Python version.
22. Click that Python version.
23. Choose the interpreter inside this project’s `venv` folder.
24. On Mac/Linux, the correct interpreter is usually `venv/bin/python`
25. On Windows, the correct interpreter is usually `venv\Scripts\python.exe`
26. If the interpreter is not listed, press `Cmd+Shift+P` on Mac or `Ctrl+Shift+P` on Windows, type `Python: Select Interpreter`, and choose the interpreter inside the local `venv` folder.
27. After the interpreter is selected, open a `.py` file and look in the top-right corner for the play button labeled `Run Python File`.
28. That button should run the script with the selected interpreter.

Recommended first clean-machine run:

```bash
python src/Codes/LightGBM_model.py
```

If the TA wants to use the VS Code run button instead of typing the command:

1. Open `src/Codes/LightGBM_model.py`
2. Confirm the selected interpreter is the local `venv`
3. Click the top-right `Run Python File` play button

VS Code should open a terminal automatically and run the script with the selected interpreter.

## Environment Setup

Run all commands from the project root.

### Mac/Linux

Create the virtual environment:

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install matplotlib
```

### Windows

Create the virtual environment:

```powershell
py -m venv venv
```

Activate it in Command Prompt:

```powershell
venv\Scripts\activate
```

Or in PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
python -m pip install matplotlib
```

Important note:

- `matplotlib` is required by the model scripts and figure scripts. Install it even if your environment was just created from `requirements.txt`.

## Main Model Scripts

Primary modeling scripts:

- `src/Codes/LR_model.py`
- `src/Codes/XGB_model.py`
- `src/Codes/CatBoost_model.py`
- `src/Codes/LightGBM_model.py`

Shared utilities:

- `src/Codes/model_feature_utils.py`
- `src/Codes/common_model_utils.py`

## Final Baseline Runs

These are the main no-ERA baseline runs. These are also the forecast-ready runs used for strict recursive forecasting.

### Mac/Linux

Ridge Regression:

```bash
python src/Codes/LR_model.py
```

XGBoost:

```bash
python src/Codes/XGB_model.py
```

CatBoost:

```bash
python src/Codes/CatBoost_model.py
```

LightGBM:

```bash
python src/Codes/LightGBM_model.py
```

### Windows

Ridge Regression:

```powershell
python src\Codes\LR_model.py
```

XGBoost:

```powershell
python src\Codes\XGB_model.py
```

CatBoost:

```powershell
python src\Codes\CatBoost_model.py
```

LightGBM:

```powershell
python src\Codes\LightGBM_model.py
```

## ERA5-Enhanced Runs

Use these runs when you want the model trained with ERA5-enabled features.

### Mac/Linux

Ridge Regression:

```bash
LR_USE_ERA5=1 python src/Codes/LR_model.py
```

XGBoost:

```bash
XGB_USE_ERA5=1 python src/Codes/XGB_model.py
```

CatBoost:

```bash
CATBOOST_USE_ERA5=1 python src/Codes/CatBoost_model.py
```

LightGBM:

```bash
LIGHTGBM_USE_ERA5=1 python src/Codes/LightGBM_model.py
```

### Windows

Ridge Regression:

```powershell
set LR_USE_ERA5=1
python src\Codes\LR_model.py
```

XGBoost:

```powershell
set XGB_USE_ERA5=1
python src\Codes\XGB_model.py
```

CatBoost:

```powershell
set CATBOOST_USE_ERA5=1
python src\Codes\CatBoost_model.py
```

LightGBM:

```powershell
set LIGHTGBM_USE_ERA5=1
python src\Codes\LightGBM_model.py
```

## ERA5 Comparison Runs

Use these when you want the before-versus-after ERA5 comparison outputs.

### Mac/Linux

Ridge Regression:

```bash
LR_USE_ERA5=1 LR_COMPARE_ERA5=1 python src/Codes/LR_model.py
```

XGBoost:

```bash
XGB_USE_ERA5=1 XGB_COMPARE_ERA5=1 python src/Codes/XGB_model.py
```

CatBoost:

```bash
CATBOOST_USE_ERA5=1 CATBOOST_COMPARE_ERA5=1 python src/Codes/CatBoost_model.py
```

LightGBM:

```bash
LIGHTGBM_USE_ERA5=1 LIGHTGBM_COMPARE_ERA5=1 python src/Codes/LightGBM_model.py
```

### Windows

Ridge Regression:

```powershell
set LR_USE_ERA5=1
set LR_COMPARE_ERA5=1
python src\Codes\LR_model.py
```

XGBoost:

```powershell
set XGB_USE_ERA5=1
set XGB_COMPARE_ERA5=1
python src\Codes\XGB_model.py
```

CatBoost:

```powershell
set CATBOOST_USE_ERA5=1
set CATBOOST_COMPARE_ERA5=1
python src\Codes\CatBoost_model.py
```

LightGBM:

```powershell
set LIGHTGBM_USE_ERA5=1
set LIGHTGBM_COMPARE_ERA5=1
python src\Codes\LightGBM_model.py
```

## What Each Run Produces

Each model writes outputs into `data/processed/`.

Typical outputs include:

- `*_eval_metrics.csv`
- `*_predictions_2023.csv`
- `*_feature_importance.csv` for tree-based models
- `*_era5_comparison_metrics.csv` when comparison mode is enabled
- `*_era5_comparison.png` when comparison mode saves a plot
- `*_model_results.png` when the main plot is enabled

Examples:

- `lr_eval_metrics.csv`
- `xgb_eval_metrics.csv`
- `catboost_eval_metrics.csv`
- `lightgbm_eval_metrics.csv`
- `xgb_predictions_2023.csv`
- `catboost_predictions_2023.csv`
- `lightgbm_predictions_2023.csv`
- `lr_predictions_2023.csv`

## Important Forecasting Note

For strict recursive forecasting, the no-ERA baseline runs are the cleanest forecast-ready path.

Why:

- same-month ERA5 reanalysis is not a true future predictor
- the scripts intentionally skip strict forecast generation in ERA5 comparison mode
- the baseline runs are the ones to use for the main 2023 recursive forecast outputs

## Figure Scripts

Figure and results scripts live in:

`src/Codes/Figure and other codes/`

Examples:

- `Time Series.py`
- `Top Features.py`
- `All Model Results Table.py`
- `XGB vs CatBoost.py`
- `create_era5_comparison_figures.py`

Run them from the project root after the model outputs have been saved.

## Exact Final Reproduction Workflow

If you want to reproduce the full final workflow from scratch on a clean machine:

1. Clone the repository.
2. Create and activate a fresh virtual environment.
3. Install dependencies with:
   - `python -m pip install -r requirements.txt`
   - `python -m pip install matplotlib`
4. Place `na_pm25_cells_clean.csv` into `data/processed/`.
5. Place `meteorlogy-data.nc` and `meteorlogy-data-other.nc` into `data/raw/`.
6. Run all four baseline model scripts.
7. Run any ERA5 comparison scripts you need with the `*_USE_ERA5=1` and `*_COMPARE_ERA5=1` flags.
8. Regenerate any final figures from `src/Codes/Figure and other codes/`.

## Authors

Alfred Liljas  
The Pennsylvania State University

Otis Murray  
The Pennsylvania State University  
Major in Statistical Modeling Data Science, minor in Mathematics
