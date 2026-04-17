from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score


def compute_metrics(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MedianAE": float(median_absolute_error(y_true, y_pred)),
        "Bias": float(np.mean(y_pred - y_true)),
    }


def print_metrics(name, metrics):
    print(f"\n{name}")
    print(f"RMSE:      {metrics['RMSE']:.6f}")
    print(f"MAE:       {metrics['MAE']:.6f}")
    print(f"R^2:       {metrics['R2']:.6f}")
    print(f"Median AE: {metrics['MedianAE']:.6f}")
    print(f"Bias:      {metrics['Bias']:.6f}")


def transform_target(values, transform_name, env_var_name):
    if transform_name == "log1p":
        return np.log1p(values)
    if transform_name == "none":
        return values
    raise ValueError(f"{env_var_name} must be 'log1p' or 'none'")


def inverse_target(values, transform_name):
    if transform_name == "log1p":
        return np.expm1(values)
    return values


def split_train_val_test(df, train_end, val_end):
    train = df[df["date"] < train_end].copy()
    val = df[(df["date"] >= train_end) & (df["date"] < val_end)].copy()
    test = df[df["date"] >= val_end].copy()
    return train, val, test


def print_run_configuration(feature_set, active_features, target_transform, era5_feature_level, train, val, test):
    print(f"\nFeature set: {feature_set}")
    print(f"Feature count: {len(active_features)}")
    print(f"Target transform: {target_transform}")
    print(f"ERA5 feature level: {era5_feature_level}")
    print(f"\nTrain: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")


def build_metrics_table(model_name, feature_set, scenario_label, val_metrics, test_metrics, naive_metrics):
    return pd.DataFrame(
        [
            {"Dataset": "Validation", "Model": model_name, "FeatureSet": feature_set, "Scenario": scenario_label, **val_metrics},
            {"Dataset": "Test", "Model": model_name, "FeatureSet": feature_set, "Scenario": scenario_label, **test_metrics},
            {"Dataset": "Naive_Test", "Model": "pm25_lag1", "Scenario": "lag1", **naive_metrics},
        ]
    )


def build_comparison_table(feature_set, baseline_result, enhanced_result):
    return pd.DataFrame(
        [
            {"Scenario": "Without_ERA5", "Dataset": "Validation", "FeatureSet": feature_set, **baseline_result["val_metrics"]},
            {"Scenario": "Without_ERA5", "Dataset": "Test", "FeatureSet": feature_set, **baseline_result["test_metrics"]},
            {"Scenario": "With_ERA5", "Dataset": "Validation", "FeatureSet": feature_set, **enhanced_result["val_metrics"]},
            {"Scenario": "With_ERA5", "Dataset": "Test", "FeatureSet": feature_set, **enhanced_result["test_metrics"]},
        ]
    )


def save_main_results_plot(
    result,
    output_file,
    enabled,
    skip_message,
    ranked_df_key,
    value_column,
    title,
    random_seed,
    sample_size=100_000,
    sort_column=None,
    label_column="feature",
):
    if not enabled:
        print(skip_message)
        return

    import matplotlib.pyplot as plt

    ranked_df = result[ranked_df_key]
    sort_column = sort_column or value_column

    fig = plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    top_ranked = ranked_df.head(15).sort_values(sort_column)
    plt.barh(top_ranked[label_column], top_ranked[value_column])
    plt.title(title)

    plt.subplot(1, 3, 2)
    plot_sample = min(sample_size, len(result["y_test"]))
    sample_idx = np.random.default_rng(random_seed).choice(
        len(result["y_test"]),
        size=plot_sample,
        replace=False,
    )
    plt.scatter(result["y_test"][sample_idx], result["test_pred"][sample_idx], alpha=0.25, s=5)
    lims = [
        min(float(result["y_test"].min()), float(np.min(result["test_pred"]))),
        max(float(result["y_test"].max()), float(np.max(result["test_pred"]))),
    ]
    plt.plot(lims, lims, "r--", linewidth=1)
    plt.xlabel("Actual PM2.5")
    plt.ylabel("Predicted PM2.5")
    plt.title("Predicted vs Actual (Test)")

    plt.subplot(1, 3, 3)
    residuals = result["y_test"] - result["test_pred"]
    plt.hist(residuals, bins=60)
    plt.axvline(0, linewidth=1)
    plt.xlabel("Residual (Actual - Predicted)")
    plt.title("Residual Distribution")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print("Saved plot to:", output_file)


def save_era5_comparison_plot(
    baseline_result,
    enhanced_result,
    output_file,
    enabled,
    skip_message,
    model_name,
    random_seed,
    sample_size=75_000,
):
    if not enabled:
        print(skip_message)
        return

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 10))

    labels = ["Validation", "Test"]
    x = np.arange(len(labels))
    width = 0.35

    plt.subplot(2, 2, 1)
    baseline_r2 = [baseline_result["val_metrics"]["R2"], baseline_result["test_metrics"]["R2"]]
    enhanced_r2 = [enhanced_result["val_metrics"]["R2"], enhanced_result["test_metrics"]["R2"]]
    plt.bar(x - width / 2, baseline_r2, width=width, label="Without ERA5")
    plt.bar(x + width / 2, enhanced_r2, width=width, label="With ERA5")
    plt.xticks(x, labels)
    plt.ylabel("R^2")
    plt.title(f"{model_name} R^2 Before vs After ERA5")
    plt.legend()

    plt.subplot(2, 2, 2)
    baseline_rmse = [baseline_result["val_metrics"]["RMSE"], baseline_result["test_metrics"]["RMSE"]]
    enhanced_rmse = [enhanced_result["val_metrics"]["RMSE"], enhanced_result["test_metrics"]["RMSE"]]
    plt.bar(x - width / 2, baseline_rmse, width=width, label="Without ERA5")
    plt.bar(x + width / 2, enhanced_rmse, width=width, label="With ERA5")
    plt.xticks(x, labels)
    plt.ylabel("RMSE")
    plt.title(f"{model_name} RMSE Before vs After ERA5")
    plt.legend()

    plot_sample = min(sample_size, len(baseline_result["y_test"]))
    sample_idx = np.random.default_rng(random_seed).choice(
        len(baseline_result["y_test"]),
        size=plot_sample,
        replace=False,
    )

    plt.subplot(2, 2, 3)
    plt.scatter(
        baseline_result["y_test"][sample_idx],
        baseline_result["test_pred"][sample_idx],
        alpha=0.2,
        s=5,
    )
    baseline_lims = [
        min(float(baseline_result["y_test"].min()), float(np.min(baseline_result["test_pred"]))),
        max(float(baseline_result["y_test"].max()), float(np.max(baseline_result["test_pred"]))),
    ]
    plt.plot(baseline_lims, baseline_lims, "r--", linewidth=1)
    plt.xlabel("Actual PM2.5")
    plt.ylabel("Predicted PM2.5")
    plt.title("Without ERA5")

    plt.subplot(2, 2, 4)
    plt.scatter(
        enhanced_result["y_test"][sample_idx],
        enhanced_result["test_pred"][sample_idx],
        alpha=0.2,
        s=5,
    )
    enhanced_lims = [
        min(float(enhanced_result["y_test"].min()), float(np.min(enhanced_result["test_pred"]))),
        max(float(enhanced_result["y_test"].max()), float(np.max(enhanced_result["test_pred"]))),
    ]
    plt.plot(enhanced_lims, enhanced_lims, "r--", linewidth=1)
    plt.xlabel("Actual PM2.5")
    plt.ylabel("Predicted PM2.5")
    plt.title("With ERA5")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print("Saved ERA5 comparison plot to:", output_file)


def run_recursive_forecast(
    model,
    history_frame,
    train_frame,
    active_features,
    feature_set_name,
    output_file,
    target,
    train_end,
    raw_dir,
    build_feature_sets_fn,
    add_history_features_fn,
    add_train_only_climatology_fn,
    add_experimental_features_fn,
    add_era5_features_fn,
    predict_fn,
    inverse_transform_fn,
    fill_values,
    forecast_months=12,
    era5_feature_level="core",
    prepare_model_frame_fn=None,
    clip_to_train_range=True,
):
    print("\n--- FORECASTING 2023 ---")

    history = history_frame[["lat", "lon", "date", target]].copy()
    history["date"] = pd.to_datetime(history["date"])
    history = history.sort_values(["lat", "lon", "date"]).reset_index(drop=True)
    future_preds = []
    latest_date = history["date"].max()
    train_min = float(train_frame[target].min())
    train_max = float(train_frame[target].max())

    for step in range(forecast_months):
        next_date = latest_date + pd.DateOffset(months=1)
        base = history[history["date"] == latest_date][["lat", "lon"]].copy()
        if base.empty:
            raise ValueError(
                f"No grid cells found for latest_date={latest_date:%Y-%m-%d}. "
                "History does not contain a usable latest month."
            )

        base["date"] = next_date
        temp = pd.concat([history, base.assign(**{target: np.nan})], ignore_index=True)
        temp = temp.sort_values(["lat", "lon", "date"]).reset_index(drop=True)
        temp = add_history_features_fn(temp, target=target)
        temp = add_train_only_climatology_fn(temp, train_end=train_end, target=target)
        temp = add_experimental_features_fn(temp)
        temp, future_era5_feature_names = add_era5_features_fn(
            temp,
            raw_dir=raw_dir,
            train_end=train_end,
            use_era5=False,
            feature_level=era5_feature_level,
        )

        future_feature_sets = build_feature_sets_fn(future_era5_feature_names)
        if feature_set_name not in future_feature_sets:
            raise ValueError(f"Unknown forecast feature set: {feature_set_name}")

        future_rows = temp[temp["date"] == next_date].copy()
        if future_rows.empty:
            raise ValueError(f"Forecast rows are empty for {next_date:%Y-%m-%d}.")

        missing_feature_columns = [col for col in active_features if col not in future_rows.columns]
        if missing_feature_columns:
            raise ValueError(
                "Forecast feature mismatch. Missing columns: "
                + ", ".join(missing_feature_columns)
            )

        print(f"\nForecast step {step + 1}/{forecast_months}: {next_date.strftime('%Y-%m-%d')}")

        missing_counts = future_rows[active_features].isna().sum()
        missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
        if not missing_counts.empty:
            print("Missing values before fill:")
            print(missing_counts.to_string())

        X_future = future_rows.loc[:, active_features].copy()
        X_future = X_future.fillna(fill_values)
        X_future = X_future.fillna(0.0)

        if X_future.empty:
            raise ValueError(f"Forecast matrix is empty for {next_date:%Y-%m-%d} after feature filling.")

        if X_future.isna().any().any():
            remaining_missing = X_future.isna().sum()
            remaining_missing = remaining_missing[remaining_missing > 0]
            raise ValueError(
                "Forecast matrix still contains NaNs after fill:\n"
                + remaining_missing.to_string()
            )

        if prepare_model_frame_fn is not None:
            X_future = prepare_model_frame_fn(X_future, active_features)

        preds = inverse_transform_fn(predict_fn(model, X_future))
        if clip_to_train_range:
            preds = np.clip(preds, train_min, train_max)

        future_rows[target] = preds
        next_month_output = future_rows[["lat", "lon", "date", target]].copy()
        history = pd.concat([history, next_month_output], ignore_index=True)
        future_preds.append(next_month_output)
        latest_date = next_date

    final_df = pd.concat(future_preds, ignore_index=True)
    final_df.to_csv(output_file, index=False)
    print("\nSaved predictions to:", output_file)
