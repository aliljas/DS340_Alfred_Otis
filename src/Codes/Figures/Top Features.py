from pathlib import Path

from figure_path_utils import (
    PROCESSED_DIR,
    TOP_FEATURES_FIGURE_FILE,
    TOP_FEATURES_TABLE_FILE,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


MODEL_SOURCES = [
    {
        "model": "Ridge Regression",
        "file": PROCESSED_DIR / "lr_coefficients.csv",
        "value_col": "abs_coefficient",
        "label_col": "feature",
        "color": "#1b9e77",
        "title": "Top 5 Ridge Features",
    },
    {
        "model": "XGBoost",
        "file": PROCESSED_DIR / "xgb_feature_importance.csv",
        "value_col": "importance",
        "label_col": "feature",
        "color": "#7570b3",
        "title": "Top 5 XGBoost Features",
    },
    {
        "model": "CatBoost",
        "file": PROCESSED_DIR / "catboost_feature_importance.csv",
        "value_col": "importance",
        "label_col": "feature",
        "color": "#d95f02",
        "title": "Top 5 CatBoost Features",
    },
    {
        "model": "LightGBM",
        "file": PROCESSED_DIR / "lightgbm_feature_importance.csv",
        "value_col": "importance",
        "label_col": "feature",
        "color": "#e7298a",
        "title": "Top 5 LightGBM Features",
    },
]

MANUAL_TOP_FEATURES = {
    "XGBoost": [
        ("pm25_lag1", 936906.6875),
        ("neighbor_max_lag1", 336187.03125),
        ("lat_lon_p90", 227377.734375),
        ("neighbor_mean_lag1", 191374.65625),
        ("lat_lon_mean", 101760.734375),
    ]
}


def load_top_features(source, top_n=5):
    feature_file = Path(source["file"])
    if not feature_file.exists():
        manual_rows = MANUAL_TOP_FEATURES.get(source["model"])
        if manual_rows is None:
            return None

        top_df = pd.DataFrame(manual_rows, columns=["feature", "strength"])
        top_df["model"] = source["model"]
        return top_df.head(top_n)

    df = pd.read_csv(feature_file)
    top_df = (
        df.loc[:, [source["label_col"], source["value_col"]]]
        .dropna()
        .sort_values(source["value_col"], ascending=False)
        .head(top_n)
        .copy()
    )
    top_df["model"] = source["model"]
    top_df = top_df.rename(
        columns={
            source["label_col"]: "feature",
            source["value_col"]: "strength",
        }
    )
    return top_df


def add_missing_panel(ax, source):
    ax.axis("off")
    ax.set_title(source["title"])
    ax.text(
        0.5,
        0.5,
        "Feature importance file\nnot available yet",
        ha="center",
        va="center",
        fontsize=11,
    )


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    axes = axes.flatten()

    collected_tables = []

    for ax, source in zip(axes, MODEL_SOURCES):
        top_df = load_top_features(source)

        if top_df is None or top_df.empty:
            add_missing_panel(ax, source)
            continue

        collected_tables.append(top_df)

        plot_df = top_df.sort_values("strength", ascending=True)
        ax.barh(plot_df["feature"], plot_df["strength"], color=source["color"])
        ax.set_title(source["title"])
        ax.set_xlabel("Feature Strength")
        ax.set_xticks([])
        ax.grid(axis="x", alpha=0.25)

    plt.suptitle("Top 5 Strongest Features Across Forecasting Models", y=0.98, fontsize=15)
    plt.tight_layout()
    plt.savefig(TOP_FEATURES_FIGURE_FILE, bbox_inches="tight")
    plt.close()

    if collected_tables:
        combined_df = pd.concat(collected_tables, ignore_index=True)
        combined_df.to_csv(TOP_FEATURES_TABLE_FILE, index=False)


if __name__ == "__main__":
    main()
