#!/usr/bin/env python3
"""Exploratory wave-specific sensitivity analysis for Reviewer 2 Comment 8.

This script does not modify the pooled main model or any manuscript artifact.
It fits the existing XGBoost specification separately in the 2016 and 2022
samples, using outcome-stratified 10-fold cross-validation, and writes all
outputs to an isolated directory.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


OUTCOME = "HumanDiseaseIncreasePast25_Dummy"
DISASTER = "DisasterExpInd"
KNOWLEDGE = "HeardClimate_Dummy"
GRID = np.arange(0, 11, dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Rev/analysis/reviewer-2-comment-8-wave-sensitivity"),
    )
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument(
        "--csv-cache-dir",
        type=Path,
        default=None,
        help="Optional directory containing 2016.csv and 2022.csv when the modelling environment lacks a Parquet engine.",
    )
    return parser.parse_args()


def load_project_data(
    root: Path, csv_cache_dir: Path | None = None
) -> tuple[pd.DataFrame, list[str]]:
    # SettingForFeatures imports geopandas only for an unrelated spatial helper.
    # A minimal stub lets this experiment reuse the project's exact analytical
    # data construction without adding a spatial dependency.
    if "geopandas" not in sys.modules:
        stub = types.ModuleType("geopandas")
        stub.GeoDataFrame = pd.DataFrame
        sys.modules["geopandas"] = stub

    sys.path.insert(0, str(root / "notebooks"))
    import SettingForFeatures  # pylint: disable=import-error,import-outside-toplevel

    original_read_parquet = pd.read_parquet
    if csv_cache_dir is not None:
        cache = csv_cache_dir.resolve()

        def read_cached_parquet(path: str | Path, *args, **kwargs) -> pd.DataFrame:
            del args, kwargs
            name = Path(path).name
            if "2016" in name:
                return pd.read_csv(cache / "2016.csv")
            if "2022" in name:
                return pd.read_csv(cache / "2022.csv")
            raise RuntimeError(f"Unexpected Parquet path: {path}")

        pd.read_parquet = read_cached_parquet
    try:
        data = SettingForFeatures.data_load_combine_dataset()
    finally:
        pd.read_parquet = original_read_parquet
    predictors = SettingForFeatures.return_input_variables()
    required = predictors + [OUTCOME]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise RuntimeError(f"Missing analytical variables: {missing}")
    analytical = data[required].dropna().copy()
    if analytical[required].isna().any().any():
        raise RuntimeError("Analytical matrix still contains missing values")
    return analytical, predictors


def load_params(root: Path, seed: int, n_jobs: int) -> dict:
    with (root / f"{OUTCOME}_params.yaml").open("r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)
    params = dict(params)
    # CPU/GPU choice is computational, not a change to the fitted specification.
    params["device"] = "cpu"
    params["tree_method"] = "hist"
    params["random_state"] = seed
    params["n_jobs"] = n_jobs
    params.setdefault("eval_metric", "logloss")
    return params


def classification_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict:
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "prevalence": float(np.mean(y_true)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "sensitivity": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def fold_curve(model: xgb.XGBClassifier, x_test: pd.DataFrame) -> dict[str, np.ndarray]:
    disaster_curve = []
    knowledge_no_curve = []
    knowledge_yes_curve = []
    for value in GRID:
        adjusted = x_test.copy()
        adjusted[DISASTER] = value
        disaster_curve.append(float(model.predict_proba(adjusted)[:, 1].mean()))

        adjusted[KNOWLEDGE] = 0
        knowledge_no_curve.append(float(model.predict_proba(adjusted)[:, 1].mean()))

        adjusted[KNOWLEDGE] = 1
        knowledge_yes_curve.append(float(model.predict_proba(adjusted)[:, 1].mean()))

    no_knowledge = x_test.copy()
    no_knowledge[KNOWLEDGE] = 0
    yes_knowledge = x_test.copy()
    yes_knowledge[KNOWLEDGE] = 1
    average_knowledge_contrast = float(
        (
            model.predict_proba(yes_knowledge)[:, 1]
            - model.predict_proba(no_knowledge)[:, 1]
        ).mean()
    )
    return {
        "disaster_curve": np.asarray(disaster_curve),
        "knowledge_no_curve": np.asarray(knowledge_no_curve),
        "knowledge_yes_curve": np.asarray(knowledge_yes_curve),
        "knowledge_contrast_curve": np.asarray(knowledge_yes_curve)
        - np.asarray(knowledge_no_curve),
        "average_knowledge_contrast": np.asarray([average_knowledge_contrast]),
    }


def fit_wave(
    year: int,
    data: pd.DataFrame,
    predictors: list[str],
    params: dict,
    folds: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wave = data.loc[data["Year"].eq(year)].copy()
    wave_predictors = [name for name in predictors if name != "Year"]
    x_data = wave[wave_predictors]
    y_data = wave[OUTCOME].astype(int)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    oof_probability = np.full(len(wave), np.nan)
    oof_fold = np.full(len(wave), -1, dtype=int)
    fold_metrics: list[dict] = []
    curves: dict[str, list[np.ndarray]] = {
        "disaster_curve": [],
        "knowledge_no_curve": [],
        "knowledge_yes_curve": [],
        "knowledge_contrast_curve": [],
        "average_knowledge_contrast": [],
    }

    for fold, (train_index, test_index) in enumerate(
        splitter.split(x_data, y_data), start=1
    ):
        print(f"year={year} fold={fold}/{folds}", flush=True)
        model = xgb.XGBClassifier(**params)
        model.fit(x_data.iloc[train_index], y_data.iloc[train_index])
        probability = model.predict_proba(x_data.iloc[test_index])[:, 1]
        oof_probability[test_index] = probability
        oof_fold[test_index] = fold

        metrics = classification_metrics(y_data.iloc[test_index].to_numpy(), probability)
        metrics.update({"year": year, "fold": fold})
        fold_metrics.append(metrics)

        current_curves = fold_curve(model, x_data.iloc[test_index])
        for name, values in current_curves.items():
            curves[name].append(values)

    if np.isnan(oof_probability).any() or (oof_fold < 1).any():
        raise RuntimeError(f"Incomplete OOF predictions for {year}")

    oof = pd.DataFrame(
        {
            "year": year,
            "row_position": np.arange(len(wave)),
            "fold": oof_fold,
            "y_true": y_data.to_numpy(),
            "y_probability": oof_probability,
            "y_prediction": (oof_probability >= 0.5).astype(int),
        }
    )

    curve_rows: list[dict] = []
    for name in (
        "disaster_curve",
        "knowledge_no_curve",
        "knowledge_yes_curve",
        "knowledge_contrast_curve",
    ):
        array = np.vstack(curves[name])
        for index, grid_value in enumerate(GRID):
            values = array[:, index]
            curve_rows.append(
                {
                    "year": year,
                    "curve": name,
                    "disaster_count": int(grid_value),
                    "mean": float(values.mean()),
                    "sd_across_folds": float(values.std(ddof=1)),
                    "fold_q025": float(np.quantile(values, 0.025)),
                    "fold_q975": float(np.quantile(values, 0.975)),
                }
            )
    average_contrast = np.concatenate(curves["average_knowledge_contrast"])
    curve_rows.append(
        {
            "year": year,
            "curve": "average_knowledge_contrast",
            "disaster_count": -1,
            "mean": float(average_contrast.mean()),
            "sd_across_folds": float(average_contrast.std(ddof=1)),
            "fold_q025": float(np.quantile(average_contrast, 0.025)),
            "fold_q975": float(np.quantile(average_contrast, 0.975)),
        }
    )
    return oof, pd.DataFrame(fold_metrics), pd.DataFrame(curve_rows)


def curve_values(curves: pd.DataFrame, year: int, name: str) -> np.ndarray:
    selected = curves.loc[(curves["year"] == year) & (curves["curve"] == name)]
    return selected.sort_values("disaster_count")["mean"].to_numpy()


def build_diagnostics(curves: pd.DataFrame, support: pd.DataFrame) -> dict:
    supported = support.pivot(index="disaster_count", columns="year", values="n")
    supported_grid = [
        int(value)
        for value in GRID
        if value in supported.index and (supported.loc[value] >= 30).all()
    ]
    if not supported_grid:
        supported_grid = [int(value) for value in GRID]

    result: dict[str, object] = {"supported_grid_minimum_n_30_both_waves": supported_grid}
    supported_min = min(supported_grid)
    supported_max = max(supported_grid)
    low_supported = [value for value in supported_grid if value <= 3]
    high_supported = [value for value in supported_grid if value >= 4]
    for year in (2016, 2022):
        disaster = curve_values(curves, year, "disaster_curve")
        contrast = curve_values(curves, year, "knowledge_contrast_curve")
        average = curves.loc[
            (curves["year"] == year)
            & (curves["curve"] == "average_knowledge_contrast"),
            "mean",
        ].iloc[0]
        low = contrast[low_supported].mean()
        high = contrast[high_supported].mean()
        result[str(year)] = {
            "disaster_delta_0_to_3": float(disaster[3] - disaster[0]),
            "disaster_delta_3_to_8": float(disaster[8] - disaster[3]),
            "disaster_delta_8_to_10": float(disaster[10] - disaster[8]),
            "disaster_delta_0_to_10": float(disaster[10] - disaster[0]),
            "disaster_delta_supported_min_to_max": float(
                disaster[supported_max] - disaster[supported_min]
            ),
            "average_knowledge_contrast_yes_minus_no": float(average),
            "knowledge_contrast_negative_supported_points": int(
                sum(contrast[value] < 0 for value in supported_grid)
            ),
            "knowledge_contrast_supported_points": len(supported_grid),
            "mean_knowledge_contrast_low_supported_counts": float(low),
            "mean_knowledge_contrast_high_supported_counts": float(high),
            "knowledge_contrast_more_negative_at_high_counts": bool(high < low),
        }

    disaster_2016 = curve_values(curves, 2016, "disaster_curve")
    disaster_2022 = curve_values(curves, 2022, "disaster_curve")
    contrast_2016 = curve_values(curves, 2016, "knowledge_contrast_curve")
    contrast_2022 = curve_values(curves, 2022, "knowledge_contrast_curve")
    result["between_wave"] = {
        "disaster_curve_pearson": float(np.corrcoef(disaster_2016, disaster_2022)[0, 1]),
        "knowledge_contrast_curve_pearson": float(
            np.corrcoef(contrast_2016, contrast_2022)[0, 1]
        ),
        "core_direction_consistent": bool(
            result["2016"]["disaster_delta_supported_min_to_max"] > 0
            and result["2022"]["disaster_delta_supported_min_to_max"] > 0
            and result["2016"]["average_knowledge_contrast_yes_minus_no"] < 0
            and result["2022"]["average_knowledge_contrast_yes_minus_no"] < 0
        ),
        "widening_pattern_consistent": bool(
            result["2016"]["knowledge_contrast_more_negative_at_high_counts"]
            and result["2022"]["knowledge_contrast_more_negative_at_high_counts"]
        ),
    }
    return result


def plot_curves(curves: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    colors = {2016: "#2c7fb8", 2022: "#d95f0e"}
    for year in (2016, 2022):
        disaster = curves.loc[
            (curves["year"] == year) & (curves["curve"] == "disaster_curve")
        ].sort_values("disaster_count")
        axes[0].plot(
            disaster["disaster_count"], disaster["mean"], marker="o", color=colors[year], label=str(year)
        )
        axes[0].fill_between(
            disaster["disaster_count"].to_numpy(),
            disaster["fold_q025"].to_numpy(),
            disaster["fold_q975"].to_numpy(),
            color=colors[year],
            alpha=0.15,
        )

        contrast = curves.loc[
            (curves["year"] == year)
            & (curves["curve"] == "knowledge_contrast_curve")
        ].sort_values("disaster_count")
        axes[1].plot(
            contrast["disaster_count"], contrast["mean"], marker="o", color=colors[year], label=str(year)
        )
        axes[1].fill_between(
            contrast["disaster_count"].to_numpy(),
            contrast["fold_q025"].to_numpy(),
            contrast["fold_q975"].to_numpy(),
            color=colors[year],
            alpha=0.15,
        )

    axes[0].set_title("Cumulative disaster exposure")
    axes[0].set_xlabel("Natural disaster count")
    axes[0].set_ylabel("Mean predicted probability")
    axes[1].set_title("Climate knowledge contrast")
    axes[1].set_xlabel("Natural disaster count")
    axes[1].set_ylabel("Prediction difference: yes minus no")
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(title="Survey wave")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data, predictors = load_project_data(root, args.csv_cache_dir)
    params = load_params(root, args.seed, args.n_jobs)
    print(
        f"analytical_n={len(data)} predictors={len(predictors)} params={params}",
        flush=True,
    )

    all_oof = []
    all_fold_metrics = []
    all_curves = []
    for year in (2016, 2022):
        oof, fold_metrics, curves = fit_wave(
            year, data, predictors, params, args.folds, args.seed
        )
        all_oof.append(oof)
        all_fold_metrics.append(fold_metrics)
        all_curves.append(curves)

    oof = pd.concat(all_oof, ignore_index=True)
    fold_metrics = pd.concat(all_fold_metrics, ignore_index=True)
    curves = pd.concat(all_curves, ignore_index=True)

    summary_rows = []
    for year in (2016, 2022):
        current = oof.loc[oof["year"] == year]
        metrics = classification_metrics(
            current["y_true"].to_numpy(), current["y_probability"].to_numpy()
        )
        metrics["year"] = year
        for metric in (
            "roc_auc",
            "accuracy",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "precision",
            "f1",
            "brier_score",
        ):
            metric_values = fold_metrics.loc[fold_metrics["year"] == year, metric]
            metrics[f"{metric}_fold_mean"] = float(metric_values.mean())
            metrics[f"{metric}_fold_sd"] = float(metric_values.std(ddof=1))
        summary_rows.append(metrics)
    metrics_summary = pd.DataFrame(summary_rows)

    support = (
        data.reset_index()
        .groupby(["Year", DISASTER], dropna=False)
        .size()
        .rename("n")
        .reset_index()
        .rename(columns={"Year": "year", DISASTER: "disaster_count"})
    )
    diagnostics = build_diagnostics(curves, support)

    oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    fold_metrics.to_csv(output_dir / "metrics_by_fold.csv", index=False)
    metrics_summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    curves.to_csv(output_dir / "wave_curves.csv", index=False)
    support.to_csv(output_dir / "disaster_count_support.csv", index=False)
    with (output_dir / "diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "purpose": "exploratory wave sensitivity for reviewer-2/comment-8",
                "outcome": OUTCOME,
                "predictor_count_pooled": len(predictors),
                "predictor_count_wave_models": len(predictors) - 1,
                "year_removed_from_wave_models": True,
                "folds": args.folds,
                "splitter": "StratifiedKFold",
                "shuffle": True,
                "seed": args.seed,
                "threshold": 0.5,
                "parameters": params,
                "grid": GRID.astype(int).tolist(),
                "fold_quantiles_are_descriptive_not_confidence_intervals": True,
            },
            handle,
            indent=2,
        )
    plot_curves(curves, output_dir / "wave_sensitivity.png")
    print(json.dumps(diagnostics, indent=2), flush=True)
    print(f"outputs={output_dir}", flush=True)


if __name__ == "__main__":
    main()
