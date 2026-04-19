import os
import tempfile
from pathlib import Path

#Cross-platform Matplotlib cache
#Send Matplotlib's cache into a temporary folder so the script runs on other machines too.
MPLCONFIGDIR = Path(tempfile.gettempdir()) / "matplotlib-cache"
#Create the cache folder if it does not already exist.
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
#Tell Matplotlib to use that cache location.
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

#Use a non-interactive backend so plots save cleanly in scripts and terminals.
import matplotlib
matplotlib.use("Agg")
#Import plotting after the backend is fixed.
import matplotlib.pyplot as plt
#Use NumPy for bar positions and layout calculations.
import numpy as np
#Use pandas for comparison tables.
import pandas as pd


#Paths
#Anchor every output path from the project root.
BASE_DIR = Path(__file__).resolve().parents[3]
#Point to the processed data folder where model metrics already live.
PROCESSED_DIR = BASE_DIR / "data" / "processed"
#Make sure the processed folder exists before saving files into it.
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

#Store the CatBoost-only comparison table here.
CATBOOST_COMPARISON_FILE = PROCESSED_DIR / "catboost_era5_comparison_metrics.csv"
#Store the CatBoost-only comparison figure here.
CATBOOST_COMPARISON_PLOT = PROCESSED_DIR / "catboost_era5_comparison.png"
#Store the combined all-model ERA comparison table here.
MODEL_COMPARISON_FILE = PROCESSED_DIR / "era5_model_comparison_full.csv"
#Store the combined all-model ERA comparison figure here.
MODEL_COMPARISON_PLOT = PROCESSED_DIR / "era5_model_comparison_full.png"


#Saved full CatBoost comparison
#Hard-code the full CatBoost baseline and ERA results so this script can build
#the paper-ready comparison without rerunning CatBoost.
CATBOOST_FULL_RESULTS = [
    {
        #Mark this row as the no-ERA baseline.
        "Scenario": "Without_ERA5",
        #This row belongs to the validation split.
        "Dataset": "Validation",
        #CatBoost used the base feature set for the winning baseline.
        "FeatureSet": "base",
        #Validation RMSE on the original PM2.5 scale.
        "RMSE": 3.227361,
        #Validation MAE on the original PM2.5 scale.
        "MAE": 1.488512,
        #Validation R-squared for the baseline run.
        "R2": 0.520329,
        #Validation median absolute error.
        "MedianAE": 0.750093,
        #Validation bias = average prediction minus actual value.
        "Bias": -0.515646,
    },
    {
        #Keep the no-ERA baseline label for the held-out test row.
        "Scenario": "Without_ERA5",
        #This row belongs to the 2022 test split.
        "Dataset": "Test",
        #Use the same CatBoost feature set label for consistency.
        "FeatureSet": "base",
        #Test RMSE for the CatBoost baseline.
        "RMSE": 1.792935,
        #Test MAE for the CatBoost baseline.
        "MAE": 1.158893,
        #Test R-squared for the CatBoost baseline.
        "R2": 0.632510,
        #Test median absolute error.
        "MedianAE": 0.834436,
        #Test bias for the CatBoost baseline.
        "Bias": 0.402567,
    },
    {
        #Mark this row as the ERA-enhanced CatBoost run.
        "Scenario": "With_ERA5",
        #This row belongs to the validation split.
        "Dataset": "Validation",
        #The ERA comparison also used the base CatBoost feature set.
        "FeatureSet": "base",
        #Validation RMSE with ERA5 features added.
        "RMSE": 3.210840,
        #Validation MAE with ERA5 features added.
        "MAE": 1.494055,
        #Validation R-squared with ERA5 features added.
        "R2": 0.525228,
        #Validation median absolute error with ERA5.
        "MedianAE": 0.771176,
        #Validation bias with ERA5.
        "Bias": -0.481599,
    },
    {
        #Keep the ERA label for the held-out test row.
        "Scenario": "With_ERA5",
        #This row belongs to the 2022 test split.
        "Dataset": "Test",
        #Use the same feature-set label for a fair comparison.
        "FeatureSet": "base",
        #Test RMSE with ERA5 added.
        "RMSE": 1.875791,
        #Test MAE with ERA5 added.
        "MAE": 1.230471,
        #Test R-squared with ERA5 added.
        "R2": 0.597760,
        #Test median absolute error with ERA5.
        "MedianAE": 0.876818,
        #Test bias with ERA5.
        "Bias": 0.471310,
    },
]


#Load or build CatBoost comparison
#Turn the saved CatBoost metrics into a dataframe we can write and plot.
def load_catboost_comparison():
    #Build a dataframe from the fixed CatBoost result records above.
    comparison_df = pd.DataFrame(CATBOOST_FULL_RESULTS)
    #Save the dataframe so other scripts and the paper can use the same table.
    comparison_df.to_csv(CATBOOST_COMPARISON_FILE, index=False)
    #Return the table so the plotting function can reuse it immediately.
    return comparison_df


#Save CatBoost figure
#Create the CatBoost-only before/after ERA figure for the paper.
def save_catboost_comparison_plot(comparison_df):
    #Split the table into baseline rows so each metric can be plotted cleanly.
    baseline = comparison_df[comparison_df["Scenario"] == "Without_ERA5"].set_index("Dataset")
    #Split the table into ERA rows so the two scenarios can be compared directly.
    enhanced = comparison_df[comparison_df["Scenario"] == "With_ERA5"].set_index("Dataset")

    #Start a 2x2 figure that shows the main metric comparisons.
    fig = plt.figure(figsize=(12, 10))
    #Use validation and test in the same order for every subplot.
    labels = ["Validation", "Test"]
    #Create x positions for the grouped bars.
    x = np.arange(len(labels))
    #Use the same bar width on every panel.
    width = 0.35

    #Plot the before/after R-squared values.
    plt.subplot(2, 2, 1)
    #Draw the baseline R-squared bars.
    plt.bar(x - width / 2, baseline.loc[labels, "R2"], width=width, label="Without ERA5")
    #Draw the ERA R-squared bars.
    plt.bar(x + width / 2, enhanced.loc[labels, "R2"], width=width, label="With ERA5")
    #Label the x-axis groups as validation and test.
    plt.xticks(x, labels)
    #Label the y-axis with the plotted metric.
    plt.ylabel("R^2")
    #Give the subplot a descriptive title.
    plt.title("CatBoost R^2 Before vs After ERA5")
    #Show the legend so the two bar colors are clear.
    plt.legend()

    #Plot the before/after RMSE values.
    plt.subplot(2, 2, 2)
    #Draw the baseline RMSE bars.
    plt.bar(x - width / 2, baseline.loc[labels, "RMSE"], width=width, label="Without ERA5")
    #Draw the ERA RMSE bars.
    plt.bar(x + width / 2, enhanced.loc[labels, "RMSE"], width=width, label="With ERA5")
    #Reuse the same validation/test x labels.
    plt.xticks(x, labels)
    #Label the y-axis with RMSE.
    plt.ylabel("RMSE")
    #Title the subplot so the reader knows this panel is RMSE.
    plt.title("CatBoost RMSE Before vs After ERA5")
    #Keep the legend for consistency.
    plt.legend()

    #Plot the before/after MAE values.
    plt.subplot(2, 2, 3)
    #Draw the baseline MAE bars.
    plt.bar(x - width / 2, baseline.loc[labels, "MAE"], width=width, label="Without ERA5")
    #Draw the ERA MAE bars.
    plt.bar(x + width / 2, enhanced.loc[labels, "MAE"], width=width, label="With ERA5")
    #Reuse the same validation/test x labels.
    plt.xticks(x, labels)
    #Label the y-axis with MAE.
    plt.ylabel("MAE")
    #Title the subplot so it is easy to discuss in the presentation.
    plt.title("CatBoost MAE Before vs After ERA5")
    #Keep the legend visible here too.
    plt.legend()

    #Plot the before/after bias values.
    plt.subplot(2, 2, 4)
    #Draw the baseline bias bars.
    plt.bar(x - width / 2, baseline.loc[labels, "Bias"], width=width, label="Without ERA5")
    #Draw the ERA bias bars.
    plt.bar(x + width / 2, enhanced.loc[labels, "Bias"], width=width, label="With ERA5")
    #Reuse the same validation/test x labels.
    plt.xticks(x, labels)
    #Label the y-axis with bias.
    plt.ylabel("Bias")
    #Title the bias comparison panel.
    plt.title("CatBoost Bias Before vs After ERA5")
    #Show the legend so the two scenarios are still obvious.
    plt.legend()

    #Tighten spacing so titles and labels do not overlap.
    plt.tight_layout()
    #Save the completed CatBoost figure into the processed folder.
    plt.savefig(CATBOOST_COMPARISON_PLOT, dpi=150)
    #Close the figure so repeated runs do not keep it in memory.
    plt.close()


#Build cross-model table
#Combine the full before/after ERA comparison tables across all models that
#already have finished full runs.
def build_model_comparison_table():
    #Point to the per-model comparison CSV files we want to combine.
    comparison_sources = {
        "CatBoost": CATBOOST_COMPARISON_FILE,
        "LightGBM": PROCESSED_DIR / "lightgbm_era5_comparison_metrics.csv",
        "LinearRegression": PROCESSED_DIR / "lr_era5_comparison_metrics.csv",
        "XGBoost": PROCESSED_DIR / "xgb_era5_comparison_metrics.csv",
    }

    #Collect the available model comparison tables here.
    frames = []
    #Loop through each model and try to add its comparison CSV.
    for model_name, path in comparison_sources.items():
        #Skip the model if its full comparison file does not exist yet.
        if not path.exists():
            continue
        #Read that model's comparison table.
        df = pd.read_csv(path)
        #Skip malformed files that do not have the needed columns.
        if "Scenario" not in df.columns or "Dataset" not in df.columns:
            continue
        #Copy so we can safely add a model-name column.
        df = df.copy()
        #Add the model name so the combined table knows which rows belong together.
        df["Model"] = model_name
        #Store this model table for the final concat.
        frames.append(df)

    #Stop with a clear message if no full comparison tables are available.
    if not frames:
        raise SystemExit("No full ERA5 comparison CSV files were found to combine.")

    #Stack all available model tables into one long dataframe.
    combined = pd.concat(frames, ignore_index=True)
    #Write the combined table so the paper has one master comparison file.
    combined.to_csv(MODEL_COMPARISON_FILE, index=False)
    #Return the merged table for plotting.
    return combined


#Save cross-model figure
#Create one figure that compares ERA effects across models on the test split.
def save_model_comparison_plot(combined_df):
    #Keep only the held-out test rows because that is the fairest model comparison.
    test_df = combined_df[combined_df["Dataset"] == "Test"].copy()
    #Pivot R-squared so each model has baseline and ERA columns side by side.
    pivot_r2 = test_df.pivot(index="Model", columns="Scenario", values="R2")
    #Pivot RMSE the same way for the second panel.
    pivot_rmse = test_df.pivot(index="Model", columns="Scenario", values="RMSE")

    #Use the model names as the x-axis categories.
    models = pivot_r2.index.tolist()
    #Create bar positions for the model groups.
    x = np.arange(len(models))
    #Use one shared bar width in both subplots.
    width = 0.35

    #Start a side-by-side figure with one panel for R-squared and one for RMSE.
    fig = plt.figure(figsize=(13, 5))

    #Plot the test R-squared comparison by model.
    plt.subplot(1, 2, 1)
    #Add the baseline bars if that scenario exists in the table.
    if "Without_ERA5" in pivot_r2.columns:
        plt.bar(x - width / 2, pivot_r2["Without_ERA5"], width=width, label="Without ERA5")
    #Add the ERA bars if that scenario exists in the table.
    if "With_ERA5" in pivot_r2.columns:
        plt.bar(x + width / 2, pivot_r2["With_ERA5"], width=width, label="With ERA5")
    #Label each grouped position with its model name.
    plt.xticks(x, models)
    #Label the y-axis with the plotted test metric.
    plt.ylabel("Test R^2")
    #Title the subplot as the main ERA effect summary.
    plt.title("ERA5 Effect by Model")
    #Show the legend for the two scenarios.
    plt.legend()

    #Plot the test RMSE comparison by model.
    plt.subplot(1, 2, 2)
    #Add baseline RMSE bars when available.
    if "Without_ERA5" in pivot_rmse.columns:
        plt.bar(x - width / 2, pivot_rmse["Without_ERA5"], width=width, label="Without ERA5")
    #Add ERA RMSE bars when available.
    if "With_ERA5" in pivot_rmse.columns:
        plt.bar(x + width / 2, pivot_rmse["With_ERA5"], width=width, label="With ERA5")
    #Reuse the model names on the x-axis.
    plt.xticks(x, models)
    #Label the y-axis with test RMSE.
    plt.ylabel("Test RMSE")
    #Title the second panel so the reader sees the error comparison too.
    plt.title("Test RMSE by Model")
    #Show the legend here as well.
    plt.legend()

    #Tighten subplot spacing before saving.
    plt.tight_layout()
    #Save the cross-model figure into the processed folder.
    plt.savefig(MODEL_COMPARISON_PLOT, dpi=150)
    #Close the figure so later scripts do not accumulate open figures.
    plt.close()


#Main
#Build the CatBoost-only comparison outputs first.
if __name__ == "__main__":
    #Load the stored full CatBoost metrics and rewrite the CSV for consistency.
    catboost_df = load_catboost_comparison()
    #Create the CatBoost-only before/after ERA figure.
    save_catboost_comparison_plot(catboost_df)
    #Report where the CatBoost comparison table was saved.
    print("Saved CatBoost ERA5 comparison metrics to:", CATBOOST_COMPARISON_FILE)
    #Report where the CatBoost comparison figure was saved.
    print("Saved CatBoost ERA5 comparison plot to:", CATBOOST_COMPARISON_PLOT)

    #Combine all available full comparison CSVs into one table.
    combined_df = build_model_comparison_table()
    #Create the cross-model figure from that combined table.
    save_model_comparison_plot(combined_df)
    #Report where the combined comparison table was saved.
    print("Saved cross-model ERA5 comparison table to:", MODEL_COMPARISON_FILE)
    #Report where the combined comparison figure was saved.
    print("Saved cross-model ERA5 comparison plot to:", MODEL_COMPARISON_PLOT)
