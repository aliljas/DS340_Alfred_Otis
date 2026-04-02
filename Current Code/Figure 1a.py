import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# -------------------------------
# PATHS
# -------------------------------
BASE_DIR = Path(__file__).resolve().parents[3]   # Project/
DATA_FILE = BASE_DIR / "data" / "processed" / "figure data" / "Figure 1.xlsx"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "figure data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print("Reading from:", DATA_FILE)
    print("File exists:", DATA_FILE.exists())

    df = pd.read_excel(DATA_FILE)

    lon = pd.to_numeric(df["Lon"], errors="coerce")
    lat = pd.to_numeric(df["Lat"], errors="coerce")
    r2 = pd.to_numeric(df["R2"], errors="coerce")

    mask = (~lon.isna()) & (~lat.isna()) & (~r2.isna())
    lon = lon[mask]
    lat = lat[mask]
    r2 = r2[mask]

    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent([-180, 180, -60, 80], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4)
    ax.add_feature(cfeature.LAND, facecolor="none", edgecolor="black", linewidth=0.2)

    sc = ax.scatter(
        lon,
        lat,
        c=r2,
        s=12,
        cmap="RdYlBu_r",
        vmin=0.3,
        vmax=0.75,
        transform=ccrs.PlateCarree()
    )

    cbar = plt.colorbar(
        sc,
        ax=ax,
        orientation="horizontal",
        pad=0.05,
        fraction=0.05,
        ticks=[0.3, 0.4, 0.5, 0.6, 0.7]
    )
    cbar.set_label("CV R²", fontsize=14)
    cbar.ax.tick_params(labelsize=14, width=2)
    cbar.outline.set_linewidth(2)

    for tick in cbar.ax.get_xticklabels():
        tick.set_fontweight("bold")

    ax.text(
        -175, 75, "(a) CV-R²",
        fontsize=18,
        fontweight="bold",
        transform=ccrs.PlateCarree()
    )

    output_file = OUTPUT_DIR / "Figure_1a_R2_map.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print("Saved to:", output_file)
    plt.close()
