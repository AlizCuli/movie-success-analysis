"""Screen and evaluate point-in-time budget context without touching the benchmark."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

from src.evaluate_operational_franchise import inner_oof as control_inner_oof
from src.operational_budget_features import (
    BUDGET_CONTEXT_NUMERIC,
    BudgetContextHistoryBuilder,
    OperationalBudgetContextBuilder,
)
from src.pre_release_features import FORBIDDEN_PREDICTORS, load_pre_release_modeling_data
from src.reproduce_operational_ab_baseline import (
    INNER_SEED,
    PARAMS,
    SEED,
    choose_threshold,
    model,
    outer_assignments,
    preprocessor,
    weights,
)


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
PREFIX = "operational_budget_v1"
INNER_GATE_MIN_MEAN_DELTA = 0.003
INNER_GATE_MIN_WINS = 3
INNER_GATE_MIN_RECALL_0_DELTA = -0.01


def metric_values(target: pd.Series, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = (probability >= threshold).astype(int)
    return {
        "macro_f1": float(f1_score(target, prediction, average="macro", zero_division=0)),
        "f1_class_0": float(f1_score(target, prediction, pos_label=0, zero_division=0)),
        "f1_class_1": float(f1_score(target, prediction, pos_label=1, zero_division=0)),
        "recall_class_0": float(recall_score(target, prediction, pos_label=0, zero_division=0)),
        "recall_class_1": float(recall_score(target, prediction, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "accuracy": float(accuracy_score(target, prediction)),
    }


def candidate_inner_oof(
    raw: pd.DataFrame,
    target: pd.Series,
) -> tuple[np.ndarray, list[int]]:
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=INNER_SEED)
    probability = np.full(len(raw), np.nan)
    iterations: list[int] = []
    for inner_fold, (train_index, valid_index) in enumerate(splitter.split(raw, target), start=1):
        train_raw, valid_raw = raw.iloc[train_index], raw.iloc[valid_index]
        train_y, valid_y = target.iloc[train_index], target.iloc[valid_index]
        builder = OperationalBudgetContextBuilder()
        train_features = builder.fit_transform(
            train_raw.reset_index(drop=True),
            train_y.reset_index(drop=True),
        )
        valid_features = builder.transform(valid_raw.reset_index(drop=True))
        transformer = preprocessor(builder)
        train_matrix = transformer.fit_transform(train_features)
        valid_matrix = transformer.transform(valid_features)
        fitted = model(PARAMS["n_estimators"], inner_fold, early_stopping=True)
        fitted.fit(
            train_matrix,
            train_y,
            sample_weight=weights(train_y),
            eval_set=[(valid_matrix, valid_y)],
            verbose=False,
        )
        probability[valid_index] = fitted.predict_proba(valid_matrix)[:, 1]
        iterations.append(
            int(getattr(fitted, "best_iteration", PARAMS["n_estimators"] - 1)) + 1
        )
    if not np.isfinite(probability).all():
        raise ValueError("Candidate inner OOF contains NaN/Inf.")
    return probability, iterations


def feature_schema() -> pd.DataFrame:
    rows = [
        ("budget_prior_percentile", "(count(prior_log_budget <= current)+0.5)/(prior_count+1)", "0.5"),
        ("log_budget_minus_prior_median", "current_log_budget - median(all prior log_budget)", "0.0"),
        ("log_prior_budget_count", "log1p(count(all prior valid budgets))", "0.0"),
        ("log_budget_minus_prior_3y_median", "current_log_budget - median(prior 3-year log_budget)", "0.0"),
        ("log_prior_budget_3y_count", "log1p(count(prior valid budgets in 3 years))", "0.0"),
        ("budget_history_available", "1 if all-history budget reference is non-empty", "0.0"),
        ("budget_history_3y_available", "1 if 3-year budget reference is non-empty", "0.0"),
    ]
    return pd.DataFrame(rows, columns=["feature", "formula", "empty_history_fallback"]).assign(
        scope="pre_release_operational",
        reference_rule="fitted training rows with release_date < query release_date",
        uses_target=False,
    )


def coverage_for_fold(
    outer_fold: int,
    train_raw: pd.DataFrame,
    valid_raw: pd.DataFrame,
) -> dict[str, float | int]:
    builder = BudgetContextHistoryBuilder().fit(train_raw)
    features = builder.transform(valid_raw)
    prior_count = np.rint(np.expm1(features["log_prior_budget_count"])).astype(int)
    window_count = np.rint(np.expm1(features["log_prior_budget_3y_count"])).astype(int)
    return {
        "outer_fold": outer_fold,
        "validation_rows": len(valid_raw),
        "all_history_available_rate": float(features["budget_history_available"].mean()),
        "three_year_history_available_rate": float(features["budget_history_3y_available"].mean()),
        "all_history_at_least_30_rate": float((prior_count >= 30).mean()),
        "three_year_history_at_least_30_rate": float((window_count >= 30).mean()),
        "median_all_history_count": float(np.median(prior_count)),
        "median_three_year_history_count": float(np.median(window_count)),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    TABLES.mkdir(parents=True, exist_ok=True)
    raw, target = load_pre_release_modeling_data()
    folds = outer_assignments(raw, target)
    screening_rows: list[dict[str, float | int]] = []
    coverage_rows: list[dict[str, float | int]] = []
    candidate_cache: dict[int, dict[str, Any]] = {}

    for outer_fold in range(1, 6):
        valid_index = np.flatnonzero(folds.to_numpy() == outer_fold)
        train_index = np.flatnonzero(folds.to_numpy() != outer_fold)
        train_raw = raw.iloc[train_index].reset_index(drop=True)
        valid_raw = raw.iloc[valid_index].reset_index(drop=True)
        train_y = target.iloc[train_index].reset_index(drop=True)

        control_probability, _ = control_inner_oof(train_raw, train_y, outer_fold)
        control_threshold, _ = choose_threshold(train_y, control_probability)
        control_metrics = metric_values(train_y, control_probability, control_threshold)

        candidate_probability, candidate_iterations = candidate_inner_oof(train_raw, train_y)
        candidate_threshold, _ = choose_threshold(train_y, candidate_probability)
        candidate_metrics = metric_values(train_y, candidate_probability, candidate_threshold)
        candidate_cache[outer_fold] = {
            "iterations": candidate_iterations,
            "threshold": candidate_threshold,
            "inner_metrics": candidate_metrics,
        }
        screening_rows.append(
            {
                "outer_fold": outer_fold,
                "control_threshold": control_threshold,
                "control_macro_f1": control_metrics["macro_f1"],
                "control_recall_class_0": control_metrics["recall_class_0"],
                "candidate_threshold": candidate_threshold,
                "candidate_macro_f1": candidate_metrics["macro_f1"],
                "candidate_recall_class_0": candidate_metrics["recall_class_0"],
                "macro_f1_delta": candidate_metrics["macro_f1"] - control_metrics["macro_f1"],
                "recall_class_0_delta": candidate_metrics["recall_class_0"] - control_metrics["recall_class_0"],
            }
        )
        coverage_rows.append(coverage_for_fold(outer_fold, train_raw, valid_raw))
        print(f"Inner screening outer partition {outer_fold}/5: complete")

    screening = pd.DataFrame(screening_rows)
    coverage = pd.DataFrame(coverage_rows)
    mean_delta = float(screening["macro_f1_delta"].mean())
    wins = int((screening["macro_f1_delta"] > 0).sum())
    mean_recall_delta = float(screening["recall_class_0_delta"].mean())
    gate_passed = bool(
        mean_delta >= INNER_GATE_MIN_MEAN_DELTA
        and wins >= INNER_GATE_MIN_WINS
        and mean_recall_delta >= INNER_GATE_MIN_RECALL_0_DELTA
    )

    schema = feature_schema()
    forbidden = sorted(set(BUDGET_CONTEXT_NUMERIC) & FORBIDDEN_PREDICTORS)
    if forbidden:
        raise ValueError(f"Budget context contains forbidden predictors: {forbidden}")
    schema.to_csv(TABLES / f"{PREFIX}_feature_schema.csv", index=False)
    coverage.to_csv(TABLES / f"{PREFIX}_coverage.csv", index=False)
    screening.to_csv(TABLES / f"{PREFIX}_inner_screening.csv", index=False)

    metadata: dict[str, Any] = {
        "experiment": PREFIX,
        "scope": "pre_release_operational",
        "rows": len(raw),
        "outer_folds": 5,
        "inner_folds": 4,
        "outer_seed": SEED,
        "inner_seed": INNER_SEED,
        "fixed_xgboost_parameters": PARAMS,
        "feature_block": BUDGET_CONTEXT_NUMERIC,
        "inner_gate": {
            "minimum_mean_macro_f1_delta": INNER_GATE_MIN_MEAN_DELTA,
            "minimum_outer_training_partition_wins": INNER_GATE_MIN_WINS,
            "minimum_mean_recall_class_0_delta": INNER_GATE_MIN_RECALL_0_DELTA,
            "observed_mean_macro_f1_delta": mean_delta,
            "observed_wins": wins,
            "observed_mean_recall_class_0_delta": mean_recall_delta,
            "passed": gate_passed,
        },
        "forbidden_predictors_found": forbidden,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }

    if not gate_passed:
        metadata["status"] = "rejected_by_inner_gate_outer_not_run"
        metadata["runtime_seconds"] = time.perf_counter() - started
        write_json(TABLES / f"{PREFIX}_run_metadata.json", metadata)
        print("INNER GATE: FAIL. Outer evaluation was not run.")
        print(screening.round(6).to_string(index=False))
        return

    prediction_rows: list[dict[str, float | int]] = []
    parameter_rows: list[dict[str, float | int]] = []
    for outer_fold in range(1, 6):
        valid_index = np.flatnonzero(folds.to_numpy() == outer_fold)
        train_index = np.flatnonzero(folds.to_numpy() != outer_fold)
        train_raw, valid_raw = raw.iloc[train_index], raw.iloc[valid_index]
        train_y, valid_y = target.iloc[train_index], target.iloc[valid_index]
        selected_iterations = int(np.median(candidate_cache[outer_fold]["iterations"]))
        threshold = float(candidate_cache[outer_fold]["threshold"])

        builder = OperationalBudgetContextBuilder()
        train_features = builder.fit_transform(
            train_raw.reset_index(drop=True),
            train_y.reset_index(drop=True),
        )
        valid_features = builder.transform(valid_raw.reset_index(drop=True))
        transformer = preprocessor(builder)
        train_matrix = transformer.fit_transform(train_features)
        valid_matrix = transformer.transform(valid_features)
        fitted = model(selected_iterations, outer_fold, early_stopping=False)
        fitted.fit(train_matrix, train_y, sample_weight=weights(train_y), verbose=False)
        probability = fitted.predict_proba(valid_matrix)[:, 1]
        prediction = (probability >= threshold).astype(int)
        parameter_rows.append(
            {
                "outer_fold": outer_fold,
                "selected_iterations": selected_iterations,
                "inner_oof_threshold": threshold,
                "inner_oof_macro_f1": candidate_cache[outer_fold]["inner_metrics"]["macro_f1"],
                "raw_feature_count": len(builder.feature_columns_),
                "transformed_feature_count": len(transformer.get_feature_names_out()),
                **PARAMS,
            }
        )
        prediction_rows.extend(
            {
                "tmdb_id": int(raw.iloc[original_index]["tmdb_id"]),
                "success": int(target.iloc[original_index]),
                "outer_fold": outer_fold,
                "probability_success": float(probability[position]),
                "threshold": threshold,
                "prediction": int(prediction[position]),
            }
            for position, original_index in enumerate(valid_index)
        )
        print(f"Outer evaluation {outer_fold}/5: complete")

    predictions = pd.DataFrame(prediction_rows).sort_values("tmdb_id")
    if len(predictions) != 1646 or not predictions["tmdb_id"].is_unique:
        raise ValueError("Candidate outer OOF row/ID validation failed.")
    if not np.isfinite(predictions["probability_success"]).all():
        raise ValueError("Candidate outer OOF contains NaN/Inf.")

    overall = metric_values(
        predictions["success"],
        predictions["probability_success"].to_numpy(dtype=float),
        0.5,
    )
    # Threshold thay đổi theo fold, nên dùng prediction đã lưu cho metric chính thức.
    y_true = predictions["success"].to_numpy(dtype=int)
    y_pred = predictions["prediction"].to_numpy(dtype=int)
    overall.update(
        {
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_class_0": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
            "f1_class_1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "recall_class_0": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
            "recall_class_1": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
        }
    )
    summary = pd.DataFrame(
        [{"model": "xgboost_operational_budget_v1", "scope": "pre_release_operational", "rows": len(predictions), **overall}]
    )
    fold_rows = []
    for outer_fold, group in predictions.groupby("outer_fold", sort=True):
        fold_target = group["success"].to_numpy(dtype=int)
        fold_prediction = group["prediction"].to_numpy(dtype=int)
        fold_rows.append(
            {
                "outer_fold": int(outer_fold),
                "rows": len(group),
                "threshold": float(group["threshold"].iloc[0]),
                "macro_f1": float(f1_score(fold_target, fold_prediction, average="macro", zero_division=0)),
                "f1_class_0": float(f1_score(fold_target, fold_prediction, pos_label=0, zero_division=0)),
                "f1_class_1": float(f1_score(fold_target, fold_prediction, pos_label=1, zero_division=0)),
                "recall_class_0": float(recall_score(fold_target, fold_prediction, pos_label=0, zero_division=0)),
                "recall_class_1": float(recall_score(fold_target, fold_prediction, pos_label=1, zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(fold_target, fold_prediction)),
            }
        )
    fold_metrics = pd.DataFrame(fold_rows)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    matrix_frame = pd.DataFrame(
        matrix,
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    )

    assignments = pd.DataFrame(
        {"tmdb_id": raw["tmdb_id"], "success": target, "outer_fold": folds}
    )
    assignments.to_csv(TABLES / f"{PREFIX}_outer_fold_assignments.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(TABLES / f"{PREFIX}_parameters.csv", index=False)
    predictions.to_csv(TABLES / f"{PREFIX}_oof_predictions.csv", index=False)
    fold_metrics.to_csv(TABLES / f"{PREFIX}_fold_metrics.csv", index=False)
    summary.to_csv(TABLES / f"{PREFIX}_metrics.csv", index=False)
    matrix_frame.to_csv(TABLES / f"{PREFIX}_confusion_matrix.csv")
    metadata["status"] = "outer_evaluation_completed"
    metadata["outer_metrics"] = overall
    metadata["runtime_seconds"] = time.perf_counter() - started
    write_json(TABLES / f"{PREFIX}_run_metadata.json", metadata)
    print("INNER GATE: PASS. Outer evaluation completed.")
    print(screening.round(6).to_string(index=False))
    print(summary.round(6).to_string(index=False))
    print(fold_metrics.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
