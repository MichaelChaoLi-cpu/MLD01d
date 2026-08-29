#!/usr/bin/env python3
"""Final fixed-parameter out-of-fold batch for Reviewer 2 Comments 4–6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from sklearn.model_selection import StratifiedKFold

import reviewer2_comment5_logistic_comparison as base


DISPLAY_NAME_OVERRIDES = {
    "DisasterExpInd": "Multi-Hazard Exposure Count",
    "HeardClimate_Dummy": "Climate Change Knowledge",
    "HouseHead_AgriExpYear": "Household Head Agricultural Experience",
    "DisasterFoodShortageInd": "Disaster-Related Food Shortage Count",
    "Dist_AgriSupport": "Distance to Agricultural Support Centre",
    "Respon_Age": "Respondent Age",
    "Prov_Bagmati": "Bagmati Province",
    "Prov_Lumbini": "Lumbini Province",
    "Dist_Market": "Distance to Market",
    "Year": "Survey Year",
    "Dist_Road": "Distance to Motorable Road",
    "LivingYear": "Years Living in Community",
    "FramMechan": "Farm Mechanization",
    "Dist_SecondarySchool": "Distance to Secondary School",
    "EcoBelt_Mountain": "Mountain Ecological Belt",
    "AgriSupport": "Agricultural Support",
    "Prov_Karnali": "Karnali Province",
    "Female_Ratio": "Female Household-Member Ratio",
    "IncomeResOthers_dummy": "Other Income Source",
    "Dist_HealthCenter": "Distance to Health Centre",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Rev/analysis/reviewer-2-batch-a-final"),
    )
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--logistic-max-iter", type=int, default=5000)
    parser.add_argument("--csv-cache-dir", type=Path, default=None)
    parser.add_argument("--beeswarm-sample", type=int, default=2500)
    return parser.parse_args()


def build_shap_summary(
    predictors: list[str], shap_values: np.ndarray
) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "feature": predictors,
            "mean_absolute_shap": np.mean(np.abs(shap_values), axis=0),
            "mean_shap": np.mean(shap_values, axis=0),
            "sd_shap": np.std(shap_values, axis=0, ddof=1),
            "positive_fraction": np.mean(shap_values > 0, axis=0),
        }
    ).sort_values("mean_absolute_shap", ascending=False)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    return summary


def plot_shap_summary(
    x_data: pd.DataFrame,
    predictors: list[str],
    shap_values: np.ndarray,
    summary: pd.DataFrame,
    display_names: dict[str, str],
    bar_output_path: Path,
    beeswarm_output_path: Path,
    seed: int,
    sample_size: int,
) -> None:
    top_features = summary.head(20)["feature"].tolist()
    labels = [display_names.get(name, name) for name in top_features]
    bar_values = summary.set_index("feature").loc[top_features, "mean_absolute_shap"]

    bar_figure, bar_axis = plt.subplots(figsize=(8.2, 7.6), constrained_layout=True)
    positions = np.arange(len(top_features))[::-1]
    bar_axis.barh(positions, bar_values.to_numpy()[::-1], color="#4472C4")
    bar_axis.set_yticks(positions, labels[::-1])
    bar_axis.set_xlabel("Mean absolute SHAP value (log-odds)")
    bar_axis.grid(axis="x", color="#D9D9D9", linewidth=0.7)
    bar_axis.set_axisbelow(True)
    for spine in bar_axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")
    bar_figure.savefig(
        bar_output_path, dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.close(bar_figure)

    swarm_figure, swarm_axis = plt.subplots(
        figsize=(8.2, 8.8), constrained_layout=True
    )
    rng = np.random.default_rng(seed)
    colour_map = plt.get_cmap("coolwarm")
    sample_count = min(sample_size, len(x_data))
    sample_index = np.sort(rng.choice(len(x_data), size=sample_count, replace=False))
    feature_to_index = {name: index for index, name in enumerate(predictors)}
    for position, feature in zip(positions, top_features[::-1]):
        feature_index = feature_to_index[feature]
        contributions = shap_values[sample_index, feature_index]
        values = x_data.iloc[sample_index][feature].to_numpy(dtype=float)
        lower, upper = np.quantile(values, [0.05, 0.95])
        if upper > lower:
            scaled = np.clip((values - lower) / (upper - lower), 0, 1)
        else:
            scaled = np.full_like(values, 0.5, dtype=float)
        jitter = rng.normal(0, 0.11, size=len(sample_index))
        swarm_axis.scatter(
            contributions,
            position + jitter,
            c=scaled,
            cmap=colour_map,
            vmin=0,
            vmax=1,
            s=7,
            alpha=0.42,
            linewidths=0,
            rasterized=True,
        )
    swarm_axis.axvline(0, color="black", linewidth=0.8)
    swarm_axis.set_yticks(positions, labels[::-1])
    swarm_axis.set_xlabel("SHAP value (change in predicted log-odds)")
    swarm_axis.grid(axis="both", color="#D9D9D9", linewidth=0.7)
    swarm_axis.set_axisbelow(True)
    colour_bar = swarm_figure.colorbar(
        ScalarMappable(norm=Normalize(0, 1), cmap=colour_map),
        ax=swarm_axis,
        pad=0.015,
        fraction=0.035,
    )
    colour_bar.set_label("Feature value")
    colour_bar.set_ticks([0, 1], labels=["Low", "High"])

    for spine in swarm_axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")
    swarm_figure.savefig(
        beeswarm_output_path, dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.close(swarm_figure)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data, predictors = base.load_project_data(root, args.csv_cache_dir)
    x_data = data[predictors]
    y_data = data[base.OUTCOME].astype(int)
    xgb_params = base.load_xgb_params(root, args.seed, args.n_jobs)

    import SettingForFeatures  # pylint: disable=import-error,import-outside-toplevel

    display_names = SettingForFeatures.return_beautiful_dict()
    display_names.update(DISPLAY_NAME_OVERRIDES)
    splitter = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    model_names = ("xgboost", "logistic_default", "logistic_extended")
    oof = {name: np.full(len(data), np.nan) for name in model_names}
    folds = np.full(len(data), -1, dtype=int)
    shap_values = np.full((len(data), len(predictors)), np.nan, dtype=np.float32)
    shap_bias = np.full(len(data), np.nan, dtype=np.float32)
    raw_margin = np.full(len(data), np.nan, dtype=np.float32)
    fold_rows: list[dict] = []
    convergence_rows: list[dict] = []
    additivity_rows: list[dict] = []

    for fold, (train_index, test_index) in enumerate(
        splitter.split(x_data, y_data), start=1
    ):
        print(f"fold={fold}/{args.folds}", flush=True)
        x_train, x_test = x_data.iloc[train_index], x_data.iloc[test_index]
        y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]
        folds[test_index] = fold

        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(x_train, y_train)
        probability = xgb_model.predict_proba(x_test)[:, 1]
        oof["xgboost"][test_index] = probability
        row = base.metrics(y_test.to_numpy(), probability)
        row.update({"model": "xgboost", "fold": fold})
        fold_rows.append(row)

        matrix = xgb.DMatrix(x_test, feature_names=predictors)
        booster = xgb_model.get_booster()
        contributions = booster.predict(matrix, pred_contribs=True)
        margins = booster.predict(matrix, output_margin=True)
        shap_values[test_index, :] = contributions[:, :-1]
        shap_bias[test_index] = contributions[:, -1]
        raw_margin[test_index] = margins
        errors = np.abs(contributions.sum(axis=1) - margins)
        additivity_rows.append(
            {
                "fold": fold,
                "n": int(len(test_index)),
                "maximum_absolute_error": float(errors.max()),
                "mean_absolute_error": float(errors.mean()),
            }
        )

        for model_name, max_iter in (
            ("logistic_default", 100),
            ("logistic_extended", args.logistic_max_iter),
        ):
            logistic_probability, convergence = base.fit_logistic(
                x_train, y_train, x_test, max_iter=max_iter
            )
            oof[model_name][test_index] = logistic_probability
            row = base.metrics(y_test.to_numpy(), logistic_probability)
            row.update({"model": model_name, "fold": fold})
            fold_rows.append(row)
            convergence.update({"model": model_name, "fold": fold})
            convergence_rows.append(convergence)

    if (
        (folds < 1).any()
        or any(np.isnan(values).any() for values in oof.values())
        or np.isnan(shap_values).any()
        or np.isnan(shap_bias).any()
        or np.isnan(raw_margin).any()
    ):
        raise RuntimeError("Incomplete out-of-fold output")

    fold_frame = pd.DataFrame(fold_rows)
    convergence_frame = pd.DataFrame(convergence_rows)
    additivity_frame = pd.DataFrame(additivity_rows)
    summary: dict[str, dict] = {}
    for name, probability in oof.items():
        current = base.metrics(y_data.to_numpy(), probability)
        current.update(base.calibration_diagnostics(y_data.to_numpy(), probability))
        subset = fold_frame.loc[fold_frame["model"].eq(name)]
        for metric_name in (
            "roc_auc",
            "accuracy",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "precision",
            "f1",
            "brier_score",
            "log_loss",
        ):
            current[f"{metric_name}_fold_mean"] = float(subset[metric_name].mean())
            current[f"{metric_name}_fold_sd"] = float(subset[metric_name].std(ddof=1))
        if name.startswith("logistic"):
            convergence_subset = convergence_frame.loc[
                convergence_frame["model"].eq(name)
            ]
            current["folds_with_convergence_warning"] = int(
                convergence_subset["convergence_warning"].sum()
            )
            current["folds_reaching_iteration_limit"] = int(
                convergence_subset["reached_iteration_limit"].sum()
            )
            current["n_iter_min"] = int(convergence_subset["n_iter"].min())
            current["n_iter_max"] = int(convergence_subset["n_iter"].max())
            current["n_iter_mean"] = float(convergence_subset["n_iter"].mean())
        summary[name] = current

    shap_summary = build_shap_summary(predictors, shap_values)
    plot_shap_summary(
        x_data,
        predictors,
        shap_values,
        shap_summary,
        display_names,
        output_dir / "figure_s3.png",
        output_dir / "figure_s4.png",
        args.seed,
        args.beeswarm_sample,
    )

    oof_frame = pd.DataFrame(
        {
            "row": np.arange(len(data)),
            "fold": folds,
            "outcome": y_data.to_numpy(),
            **{f"probability_{name}": values for name, values in oof.items()},
            "xgboost_raw_margin": raw_margin,
            "shap_bias": shap_bias,
        }
    )
    shap_frame = pd.DataFrame(shap_values, columns=predictors)
    shap_frame.insert(0, "row", np.arange(len(data)))
    diagnostics = {
        "purpose": "final fixed-parameter out-of-fold batch for reviewer-2/comments-4-6",
        "sample_size": int(len(data)),
        "outcome_prevalence": float(y_data.mean()),
        "outcome_negative_fraction": float(1 - y_data.mean()),
        "predictor_count": int(len(predictors)),
        "splitter": "StratifiedKFold",
        "folds": int(args.folds),
        "training_fraction_each_fold": 0.9,
        "testing_fraction_each_fold": 0.1,
        "seed": int(args.seed),
        "threshold": 0.5,
        "resampling": "none",
        "class_weighting": "none",
        "independent_validation_claim": False,
        "hyperparameters_fixed_before_this_evaluation": True,
        "xgboost_parameters": xgb_params,
        "logistic_interaction_terms": "none",
        "logistic_standardization": "none; existing analytical matrix used unchanged",
        "design_diagnostics": base.design_diagnostics(x_data),
        "model_summary": summary,
        "shap": {
            "method": "XGBoost exact TreeSHAP via pred_contribs=True",
            "scale": "raw margin / log-odds",
            "held_out_only": True,
            "rows": int(len(shap_values)),
            "features": int(shap_values.shape[1]),
            "maximum_additivity_error": float(
                additivity_frame["maximum_absolute_error"].max()
            ),
            "mean_additivity_error_across_folds": float(
                additivity_frame["mean_absolute_error"].mean()
            ),
            "top_20_features": shap_summary.head(20)["feature"].tolist(),
        },
        "hosmer_lemeshow_note": "Large-sample sensitive; interpret with Brier score and calibration intercept/slope.",
    }

    oof_frame.to_csv(output_dir / "oof_predictions.csv", index=False)
    fold_frame.to_csv(output_dir / "metrics_by_fold.csv", index=False)
    convergence_frame.to_csv(output_dir / "logistic_convergence_by_fold.csv", index=False)
    additivity_frame.to_csv(output_dir / "shap_additivity_by_fold.csv", index=False)
    shap_summary.to_csv(output_dir / "shap_summary.csv", index=False)
    shap_frame.to_csv(output_dir / "shap_values.csv.gz", index=False, compression="gzip")
    with (output_dir / "diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, allow_nan=False)
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "script": "scripts/reviewer2_batch_a_final.py",
                "analysis_boundary": "fixed selected hyperparameters; cross-validated out-of-fold performance; not independent validation",
                "figure_s3_content": "top-20 mean absolute SHAP values",
                "figure_s4_content": "top-20 SHAP beeswarm distributions",
                "figure_dpi": 300,
            },
            handle,
            indent=2,
        )
    print(json.dumps(diagnostics, indent=2, allow_nan=False), flush=True)
    print(f"outputs={output_dir}", flush=True)


if __name__ == "__main__":
    main()
