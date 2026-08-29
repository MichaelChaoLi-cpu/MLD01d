#!/usr/bin/env python3
"""Isolated XGBoost versus ordinary-logistic comparison for R2 Comment 5."""

from __future__ import annotations

import argparse
import json
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from scipy.stats import chi2
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


OUTCOME = "HumanDiseaseIncreasePast25_Dummy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Rev/analysis/reviewer-2-comment-5-logistic-comparison"),
    )
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--extended-max-iter", type=int, default=5000)
    parser.add_argument("--csv-cache-dir", type=Path, default=None)
    return parser.parse_args()


def load_project_data(
    root: Path, csv_cache_dir: Path | None
) -> tuple[pd.DataFrame, list[str]]:
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


def load_xgb_params(root: Path, seed: int, n_jobs: int) -> dict:
    with (root / f"{OUTCOME}_params.yaml").open("r", encoding="utf-8") as handle:
        params = dict(yaml.safe_load(handle))
    params.update(
        {
            "device": "cpu",
            "tree_method": "hist",
            "random_state": seed,
            "n_jobs": n_jobs,
            "eval_metric": "logloss",
        }
    )
    return params


def metrics(y_true: np.ndarray, probability: np.ndarray) -> dict:
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
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def fit_logistic(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    max_iter: int,
) -> tuple[np.ndarray, dict]:
    model = LogisticRegression(max_iter=max_iter)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x_train, y_train)
    convergence = [
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    ]
    probability = model.predict_proba(x_test)[:, 1]
    return probability, {
        "max_iter": int(max_iter),
        "n_iter": int(np.max(model.n_iter_)),
        "reached_iteration_limit": bool(np.max(model.n_iter_) >= max_iter),
        "convergence_warning": bool(convergence),
        "warning_messages": convergence,
        "solver": model.solver,
        "penalty": str(model.penalty),
        "C": float(model.C),
        "fit_intercept": bool(model.fit_intercept),
    }


def calibration_diagnostics(y_true: np.ndarray, probability: np.ndarray) -> dict:
    eps = np.finfo(float).eps
    clipped = np.clip(probability, eps, 1 - eps)
    logit_probability = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibration = LogisticRegression(penalty=None, max_iter=5000)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calibration.fit(logit_probability, y_true)
    warning_messages = [
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    ]

    frame = pd.DataFrame({"y": y_true, "p": probability})
    frame["group"] = pd.qcut(frame["p"], q=10, duplicates="drop")
    grouped = frame.groupby("group", observed=True).agg(
        observed=("y", "sum"), expected=("p", "sum"), n=("y", "size")
    )
    grouped["expected_no"] = grouped["n"] - grouped["expected"]
    grouped["observed_no"] = grouped["n"] - grouped["observed"]
    denom_yes = grouped["expected"].clip(lower=eps)
    denom_no = grouped["expected_no"].clip(lower=eps)
    hl_stat = float(
        (((grouped["observed"] - grouped["expected"]) ** 2) / denom_yes).sum()
        + (((grouped["observed_no"] - grouped["expected_no"]) ** 2) / denom_no).sum()
    )
    groups = int(len(grouped))
    degrees_freedom = max(groups - 2, 1)
    return {
        "calibration_intercept": float(calibration.intercept_[0]),
        "calibration_slope": float(calibration.coef_[0, 0]),
        "calibration_n_iter": int(np.max(calibration.n_iter_)),
        "calibration_convergence_warning": bool(warning_messages),
        "hosmer_lemeshow_groups": groups,
        "hosmer_lemeshow_statistic": hl_stat,
        "hosmer_lemeshow_df": degrees_freedom,
        "hosmer_lemeshow_p": float(chi2.sf(hl_stat, degrees_freedom)),
    }


def design_diagnostics(x_data: pd.DataFrame) -> dict:
    values = x_data.to_numpy(dtype=float)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    nonconstant = std > 0
    standardized = (values[:, nonconstant] - mean[nonconstant]) / std[nonconstant]
    design = np.column_stack([np.ones(len(standardized)), standardized])
    singular_values = np.linalg.svd(design, compute_uv=False)
    tolerance = singular_values.max() * max(design.shape) * np.finfo(float).eps
    rank = int(np.sum(singular_values > tolerance))
    smallest = float(singular_values.min())
    condition_number = (
        float(singular_values.max() / smallest) if smallest > 0 else float("inf")
    )
    return {
        "observations": int(design.shape[0]),
        "predictors": int(x_data.shape[1]),
        "design_columns_with_intercept": int(design.shape[1]),
        "nonconstant_predictors": int(nonconstant.sum()),
        "matrix_rank": rank,
        "rank_deficiency": int(design.shape[1] - rank),
        "condition_number_standardized_design": condition_number,
        "smallest_singular_value": smallest,
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data, predictors = load_project_data(root, args.csv_cache_dir)
    x_data = data[predictors]
    y_data = data[OUTCOME].astype(int)
    xgb_params = load_xgb_params(root, args.seed, args.n_jobs)
    splitter = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )

    model_names = ("xgboost", "logistic_default", "logistic_extended")
    oof = {name: np.full(len(data), np.nan) for name in model_names}
    fold_assignment = np.full(len(data), -1, dtype=int)
    fold_rows: list[dict] = []
    convergence_rows: list[dict] = []

    for fold, (train_index, test_index) in enumerate(
        splitter.split(x_data, y_data), start=1
    ):
        print(f"fold={fold}/{args.folds}", flush=True)
        x_train, x_test = x_data.iloc[train_index], x_data.iloc[test_index]
        y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]
        fold_assignment[test_index] = fold

        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(x_train, y_train)
        xgb_probability = xgb_model.predict_proba(x_test)[:, 1]
        oof["xgboost"][test_index] = xgb_probability
        row = metrics(y_test.to_numpy(), xgb_probability)
        row.update({"model": "xgboost", "fold": fold})
        fold_rows.append(row)

        for model_name, max_iter in (
            ("logistic_default", 100),
            ("logistic_extended", args.extended_max_iter),
        ):
            probability, convergence = fit_logistic(
                x_train, y_train, x_test, max_iter=max_iter
            )
            oof[model_name][test_index] = probability
            row = metrics(y_test.to_numpy(), probability)
            row.update({"model": model_name, "fold": fold})
            fold_rows.append(row)
            convergence.update({"model": model_name, "fold": fold})
            convergence_rows.append(convergence)

    if (fold_assignment < 1).any() or any(np.isnan(values).any() for values in oof.values()):
        raise RuntimeError("Incomplete out-of-fold predictions")

    oof_frame = pd.DataFrame(
        {
            "row": np.arange(len(data)),
            "fold": fold_assignment,
            "outcome": y_data.to_numpy(),
            **{f"probability_{name}": values for name, values in oof.items()},
        }
    )
    fold_frame = pd.DataFrame(fold_rows)
    convergence_frame = pd.DataFrame(convergence_rows)

    summary: dict[str, dict] = {}
    for name, probability in oof.items():
        current = metrics(y_data.to_numpy(), probability)
        current.update(calibration_diagnostics(y_data.to_numpy(), probability))
        fold_subset = fold_frame.loc[fold_frame["model"].eq(name)]
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
            current[f"{metric_name}_fold_mean"] = float(fold_subset[metric_name].mean())
            current[f"{metric_name}_fold_sd"] = float(
                fold_subset[metric_name].std(ddof=1)
            )
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

    diagnostics = {
        "purpose": "reviewer-2/comment-5 XGBoost versus ordinary logistic comparison",
        "sample_size": int(len(data)),
        "outcome_prevalence": float(y_data.mean()),
        "predictor_count": int(len(predictors)),
        "same_sample_predictors_and_folds": True,
        "splitter": "StratifiedKFold",
        "folds": int(args.folds),
        "seed": int(args.seed),
        "classification_threshold": 0.5,
        "logistic_interaction_terms": "none",
        "logistic_standardization": "none; existing analytical matrix used unchanged",
        "logistic_default": {
            "solver": "lbfgs",
            "penalty": "l2",
            "C": 1.0,
            "max_iter": 100,
        },
        "logistic_extended_diagnostic_refit": {
            "solver": "lbfgs",
            "penalty": "l2",
            "C": 1.0,
            "max_iter": int(args.extended_max_iter),
            "purpose": "test whether default-iteration non-convergence persists without changing the model specification",
        },
        "xgboost_parameters": xgb_params,
        "design_diagnostics": design_diagnostics(x_data),
        "model_summary": summary,
        "hosmer_lemeshow_note": "Large-sample sensitive; interpret with Brier score and calibration intercept/slope.",
    }

    oof_frame.to_csv(output_dir / "oof_predictions.csv", index=False)
    fold_frame.to_csv(output_dir / "metrics_by_fold.csv", index=False)
    convergence_frame.to_csv(output_dir / "logistic_convergence_by_fold.csv", index=False)
    with (output_dir / "diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, allow_nan=False)
    print(json.dumps(diagnostics, indent=2, allow_nan=False), flush=True)
    print(f"outputs={output_dir}", flush=True)


if __name__ == "__main__":
    main()
