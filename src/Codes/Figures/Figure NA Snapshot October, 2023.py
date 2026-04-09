from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ======================
# PATHS
# ======================
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_FILE = BASE_DIR / "data" / "processed" / "catboost_predictions_2023.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "figure data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ======================
# SETTINGS
# ======================
TARGET_DATE = "2023-10-01"   # monthly forecast date from your CatBoost output
OUTPUT_FILE = OUTPUT_DIR / "Figure_North_America_2023_10.png"

# North America bounds
LON_MIN, LON_MAX = -168, -52
LAT_MIN, LAT_MAX = 7, 84

# ======================
# LOAD DATA
# ======================
print("Reading from:", DATA_FILE)
print("File exists:", DATA_FILE.exists())

df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")

df = df.dropna(subset=["date", "lat", "lon", "pm25"]).copy()

# ======================
# FILTER TO TARGET MONTH
# ======================
target_date = pd.to_datetime(TARGET_DATE)
plot_df = df[df["date"] == target_date].copy()

# North America subset
plot_df = plot_df[
    (plot_df["lon"] >= LON_MIN) & (plot_df["lon"] <= LON_MAX) &
    (plot_df["lat"] >= LAT_MIN) & (plot_df["lat"] <= LAT_MAX)
].copy()

print("Rows for plot:", len(plot_df))

if len(plot_df) == 0:
    raise ValueError(
        f"No rows found for {TARGET_DATE}. Check the exact dates in catboost_predictions_2023.csv"
    )

# ======================
# PLOT
# ======================
fig = plt.figure(figsize=(11, 7))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="none")
ax.add_feature(cfeature.OCEAN, facecolor="white")
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS, linewidth=0.4)
ax.add_feature(cfeature.STATES, linewidth=0.2, alpha=0.5)

sc = ax.scatter(
    plot_df["lon"],
    plot_df["lat"],
    c=plot_df["pm25"],
    s=8,
    cmap="RdYlBu_r",
    vmin=0,
    vmax=max(40, min(80, plot_df["pm25"].quantile(0.99))),
    transform=ccrs.PlateCarree()
)

gl = ax.gridlines(
    draw_labels=True,
    linewidth=0.2,
    color="gray",
    alpha=0.5,
    linestyle="--"
)
gl.top_labels = False
gl.right_labels = False

cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.06, fraction=0.05)
cbar.set_label("Forecasted PM$_{2.5}$ (μg m$^{-3}$)", fontsize=12)
cbar.ax.tick_params(labelsize=10)

ax.set_title(
    "Forecasted North America PM$_{2.5}$ map using CatBoost\nOctober 2023",
    fontsize=20,
    fontweight="bold",
    pad=20,
    loc="center"
)

plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
print("Saved to:", OUTPUT_FILE)
plt.close()
