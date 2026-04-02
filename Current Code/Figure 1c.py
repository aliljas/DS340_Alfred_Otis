from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from math import sqrt
from sklearn import linear_model

# -------------------------------
# PATHS
# -------------------------------
BASE_DIR = Path(__file__).resolve().parents[3]   # Project/
DATA_FILE = BASE_DIR / "data" / "processed" / "figure data" / "Figure 1.xlsx"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "figure data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# HELPERS
# -------------------------------
def getdata(filepath, sheet_name="Figure 1c"):
    df = pd.read_excel(filepath, sheet_name=sheet_name)

    x = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
    y = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()

    mask = (~np.isnan(x)) & (~np.isnan(y))
    x = x[mask]
    y = y[mask]

    return x, y


def RMSE(x, y):
    return f"{np.sqrt(np.mean((x - y) ** 2)):.2f}"


def NRMSE(x, y):
    rmse = np.sqrt(np.mean((x - y) ** 2))
    return f"{rmse / np.mean(x):.2f}"


def MAE(x, y):
    return f"{np.mean(np.abs(x - y)):.2f}"


def multipl(a, b):
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total


def correlation(x, y):
    n = len(x)
    sum1 = sum(x)
    sum2 = sum(y)
    sumofxy = multipl(x, y)
    sumofx2 = sum([pow(i, 2) for i in x])
    sumofy2 = sum([pow(j, 2) for j in y])

    num = sumofxy - (float(sum1) * float(sum2) / n)
    den = sqrt((sumofx2 - float(sum1**2) / n) * (sumofy2 - float(sum2**2) / n))
    r = num / den
    return f"{(r * r):.2f}"


def regress(x, y):
    regr = linear_model.LinearRegression()
    t = np.array([x]).T
    regr.fit(t, y)
    slope = float(regr.coef_[0])
    intercept = float(regr.intercept_)
    return slope, intercept


def plot_data(x, y, ax):
    z_min = 0
    z_max = 200
    nbins = 140

    global plot
    H, xedges, yedges = np.histogram2d(x, y, bins=nbins)
    H = np.rot90(H)
    H = np.flipud(H)
    Hmasked = np.ma.masked_where(H == 0, H)

    plot = ax.pcolormesh(
        xedges,
        yedges,
        Hmasked,
        cmap="RdYlBu_r",
        vmin=z_min,
        vmax=z_max,
        shading="auto"
    )

    xx = [0, 1000]
    yy = [0, 1000]
    ax.plot(xx, yy, "k--", linewidth=1.8)

    slope, intercept = regress(x, y)
    x_line = np.array([0, 1000])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, "r", linewidth=1.2)

    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)

    ax.text(85, 905, f"Y = {slope:.2f}X + {intercept:.2f}",
            color="k", fontsize=16, fontweight="bold")
    ax.text(85, 835, r"R$^2$ = " + correlation(x, y),
            color="k", fontsize=16, fontweight="bold")
    ax.text(85, 765, "RMSE = " + RMSE(x, y),
            color="k", fontsize=16, fontweight="bold")
    ax.text(85, 695, "NRMSE = " + NRMSE(x, y),
            color="k", fontsize=16, fontweight="bold")
    ax.text(85, 625, "MAE = " + MAE(x, y),
            color="k", fontsize=16, fontweight="bold")


if __name__ == "__main__":
    print("Reading from:", DATA_FILE)
    print("File exists:", DATA_FILE.exists())

    x1, y1 = getdata(DATA_FILE, sheet_name="Figure 1c")

    fig1 = plt.figure(figsize=(4.8, 5.6))
    G = gridspec.GridSpec(1, 1)

    ax1 = plt.subplot(G[0, 0])
    plot_data(x1, y1, ax1)

    ax1.set_xlabel(r"Measured PM$_{2.5}$ (μg m$^{-3}$)", fontsize=22, fontweight="bold")
    ax1.set_ylabel(r"Estimated PM$_{2.5}$ (μg m$^{-3}$)", fontsize=14, fontweight="bold")

    ax1.set_xticks([0, 200, 400, 600, 800, 1000])
    ax1.set_yticks([0, 200, 400, 600, 800, 1000])
    ax1.tick_params(axis="both", labelsize=14, width=1.0, length=5)

    ax1.text(620, 55, "(c) Daily", color="k", fontsize=22, fontweight="bold")

    fig1.subplots_adjust(right=0.80)
    cbar_ax = fig1.add_axes([0.83, 0.17, 0.035, 0.66])
    cbar = fig1.colorbar(plot, cax=cbar_ax)
    cbar.set_ticks([0, 25, 50, 75, 100, 125, 150, 175, 200])
    cbar.ax.tick_params(labelsize=14)

    savepath = OUTPUT_DIR / "Figure_1c_Daily.png"
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    print("Saved to:", savepath)
    plt.show()
