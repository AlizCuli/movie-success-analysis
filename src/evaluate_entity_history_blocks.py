"""Inner-gate entity history blocks, then evaluate one locked combination once."""

from __future__ import annotations

import itertools
import hashlib
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

from src.entity_history_features import OperationalEntityHistoryBuilder, block_columns
from src.evaluate_operational_franchise import inner_oof as control_inner_oof
from src.pre_release_features import load_pre_release_modeling_data
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
PREFIX = "entity_history_v1"
GROUPS = ("director", "production_company", "cast")
MIN_MEAN_DELTA = 0.003
MIN_PARTITION_WINS = 3
MIN_RECALL_0_DELTA = -0.01


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_benchmark_integrity(raw: pd.DataFrame, target: pd.Series, folds: pd.Series) -> dict[str, Any]:
    assignment_path = TABLES / "operational_franchise_outer_fold_assignments.csv"
    metric_path = TABLES / "operational_franchise_metrics.csv"
    assignments = pd.read_csv(assignment_path).sort_values("tmdb_id").reset_index(drop=True)
    current = pd.DataFrame({
        "tmdb_id": raw["tmdb_id"].astype(int),
        "success": target.astype(int),
        "outer_fold": folds.astype(int),
    }).sort_values("tmdb_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(assignments[current.columns], current, check_dtype=False)
    benchmark = pd.read_csv(metric_path)
    macro_f1 = float(benchmark.iloc[0]["macro_f1"])
    if not np.isclose(macro_f1, 0.719483, atol=5e-7):
        raise ValueError(f"Official benchmark changed: {macro_f1}")
    return {
        "assignment_sha256": file_sha256(assignment_path),
        "metric_sha256": file_sha256(metric_path),
        "macro_f1": macro_f1,
        "rows": len(current),
    }


def metrics(target: pd.Series | np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(target, prediction, average="macro", zero_division=0)),
        "f1_class_0": float(f1_score(target, prediction, pos_label=0, zero_division=0)),
        "f1_class_1": float(f1_score(target, prediction, pos_label=1, zero_division=0)),
        "recall_class_0": float(recall_score(target, prediction, pos_label=0, zero_division=0)),
        "recall_class_1": float(recall_score(target, prediction, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "accuracy": float(accuracy_score(target, prediction)),
    }


def inner_oof(
    raw: pd.DataFrame,
    target: pd.Series,
    groups: tuple[str, ...],
) -> tuple[np.ndarray, list[int]]:
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=INNER_SEED)
    probability = np.full(len(raw), np.nan)
    iterations: list[int] = []
    for inner_fold, (train_index, valid_index) in enumerate(splitter.split(raw, target), start=1):
        train_raw = raw.iloc[train_index].reset_index(drop=True)
        valid_raw = raw.iloc[valid_index].reset_index(drop=True)
        train_y = target.iloc[train_index].reset_index(drop=True)
        valid_y = target.iloc[valid_index].reset_index(drop=True)
        builder = OperationalEntityHistoryBuilder(groups)
        train_features = builder.fit_transform(train_raw, train_y)
        valid_features = builder.transform(valid_raw)
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
        iterations.append(int(getattr(fitted, "best_iteration", PARAMS["n_estimators"] - 1)) + 1)
    if not np.isfinite(probability).all():
        raise ValueError(f"Inner OOF contains NaN/Inf for groups={groups}.")
    return probability, iterations


def gate_summary(screening: pd.DataFrame, candidate: str) -> dict[str, Any]:
    subset = screening[screening["candidate"] == candidate]
    mean_delta = float(subset["macro_f1_delta"].mean())
    wins = int((subset["macro_f1_delta"] > 0).sum())
    recall_delta = float(subset["recall_class_0_delta"].mean())
    passed = bool(
        mean_delta >= MIN_MEAN_DELTA
        and wins >= MIN_PARTITION_WINS
        and recall_delta >= MIN_RECALL_0_DELTA
    )
    return {
        "candidate": candidate,
        "mean_inner_macro_f1": float(subset["candidate_macro_f1"].mean()),
        "mean_macro_f1_delta": mean_delta,
        "partition_wins": wins,
        "mean_recall_class_0_delta": recall_delta,
        "gate_passed": passed,
    }


def candidate_name(groups: tuple[str, ...]) -> str:
    return "+".join(groups)


def evaluate_inner_candidates(
    raw: pd.DataFrame,
    target: pd.Series,
    folds: pd.Series,
    candidates: list[tuple[str, ...]],
) -> tuple[pd.DataFrame, dict[tuple[int, str], dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    cache: dict[tuple[int, str], dict[str, Any]] = {}
    for outer_fold in range(1, 6):
        train_index = np.flatnonzero(folds.to_numpy() != outer_fold)
        train_raw = raw.iloc[train_index].reset_index(drop=True)
        train_y = target.iloc[train_index].reset_index(drop=True)
        control_probability, _ = control_inner_oof(train_raw, train_y, outer_fold)
        control_threshold, _ = choose_threshold(train_y, control_probability)
        control_prediction = (control_probability >= control_threshold).astype(int)
        control_metrics = metrics(train_y, control_prediction)

        for groups in candidates:
            name = candidate_name(groups)
            probability, iterations = inner_oof(train_raw, train_y, groups)
            threshold, _ = choose_threshold(train_y, probability)
            candidate_metrics = metrics(train_y, (probability >= threshold).astype(int))
            rows.append({
                "outer_training_partition": outer_fold,
                "candidate": name,
                "groups": "|".join(groups),
                "control_threshold": control_threshold,
                "control_macro_f1": control_metrics["macro_f1"],
                "control_recall_class_0": control_metrics["recall_class_0"],
                "candidate_threshold": threshold,
                "candidate_macro_f1": candidate_metrics["macro_f1"],
                "candidate_f1_class_0": candidate_metrics["f1_class_0"],
                "candidate_f1_class_1": candidate_metrics["f1_class_1"],
                "candidate_recall_class_0": candidate_metrics["recall_class_0"],
                "candidate_recall_class_1": candidate_metrics["recall_class_1"],
                "macro_f1_delta": candidate_metrics["macro_f1"] - control_metrics["macro_f1"],
                "recall_class_0_delta": candidate_metrics["recall_class_0"] - control_metrics["recall_class_0"],
            })
            cache[(outer_fold, name)] = {
                "groups": groups,
                "threshold": threshold,
                "iterations": iterations,
                "inner_metrics": candidate_metrics,
            }
            print(f"Inner gate partition {outer_fold}/5, {name}: complete")
    return pd.DataFrame(rows), cache


def feature_schema(groups: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for group in groups:
        for feature in block_columns(group):
            rows.append({
                "entity_group": group,
                "feature": feature,
                "scope": "pre_release_operational_point_in_time",
                "maturity_lag_days_for_financial_outcomes": 365,
                "validation_updates_history": False,
                "target_movie_excluded_by_id": True,
            })
    return pd.DataFrame(rows)


def run_outer_once(
    raw: pd.DataFrame,
    target: pd.Series,
    folds: pd.Series,
    groups: tuple[str, ...],
    cache: dict[tuple[int, str], dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    name = candidate_name(groups)
    prediction_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    for outer_fold in range(1, 6):
        valid_index = np.flatnonzero(folds.to_numpy() == outer_fold)
        train_index = np.flatnonzero(folds.to_numpy() != outer_fold)
        train_raw = raw.iloc[train_index].reset_index(drop=True)
        valid_raw = raw.iloc[valid_index].reset_index(drop=True)
        train_y = target.iloc[train_index].reset_index(drop=True)
        selection = cache[(outer_fold, name)]
        iterations = int(np.median(selection["iterations"]))
        threshold = float(selection["threshold"])

        builder = OperationalEntityHistoryBuilder(groups)
        train_features = builder.fit_transform(train_raw, train_y)
        valid_features = builder.transform(valid_raw)
        transformer = preprocessor(builder)
        train_matrix = transformer.fit_transform(train_features)
        valid_matrix = transformer.transform(valid_features)
        fitted = model(iterations, outer_fold, early_stopping=False)
        fitted.fit(train_matrix, train_y, sample_weight=weights(train_y), verbose=False)
        probability = fitted.predict_proba(valid_matrix)[:, 1]
        prediction = (probability >= threshold).astype(int)
        parameter_rows.append({
            "outer_fold": outer_fold,
            "groups": "|".join(groups),
            "selected_iterations": iterations,
            "inner_oof_threshold": threshold,
            "raw_feature_count": len(builder.feature_columns_),
            "transformed_feature_count": len(transformer.get_feature_names_out()),
            **PARAMS,
        })
        prediction_rows.extend({
            "tmdb_id": int(raw.iloc[index]["tmdb_id"]),
            "success": int(target.iloc[index]),
            "outer_fold": outer_fold,
            "probability_success": float(probability[position]),
            "threshold": threshold,
            "prediction": int(prediction[position]),
        } for position, index in enumerate(valid_index))
        print(f"Locked outer evaluation {outer_fold}/5: complete")

    predictions = pd.DataFrame(prediction_rows).sort_values("tmdb_id")
    if len(predictions) != 1646 or not predictions["tmdb_id"].is_unique:
        raise ValueError("Outer OOF target IDs changed.")
    if not np.isfinite(predictions["probability_success"]).all():
        raise ValueError("Outer OOF probability contains NaN/Inf.")
    y = predictions["success"].to_numpy(dtype=int)
    pred = predictions["prediction"].to_numpy(dtype=int)
    summary = pd.DataFrame([{
        "model": f"xgboost_operational_franchise_{name}",
        "scope": "pre_release_operational_point_in_time_history",
        "rows": len(predictions),
        **metrics(y, pred),
    }])
    confusion = pd.DataFrame(
        confusion_matrix(y, pred, labels=[0, 1]),
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    )
    fold_rows = []
    for outer_fold, group in predictions.groupby("outer_fold", sort=True):
        fold_rows.append({
            "outer_fold": int(outer_fold),
            "rows": len(group),
            "threshold": float(group["threshold"].iloc[0]),
            **metrics(
                group["success"].to_numpy(dtype=int),
                group["prediction"].to_numpy(dtype=int),
            ),
        })
    return predictions, pd.DataFrame(parameter_rows), summary, pd.DataFrame(fold_rows), confusion


def main() -> None:
    started = time.perf_counter()
    TABLES.mkdir(parents=True, exist_ok=True)
    raw, target = load_pre_release_modeling_data()
    folds = outer_assignments(raw, target)
    benchmark_integrity = assert_benchmark_integrity(raw, target, folds)

    single_candidates = [(group,) for group in GROUPS]
    screening, cache = evaluate_inner_candidates(raw, target, folds, single_candidates)
    summaries = [gate_summary(screening, candidate_name(groups)) for groups in single_candidates]
    accepted_groups = tuple(row["candidate"] for row in summaries if row["gate_passed"])

    combination_candidates: list[tuple[str, ...]] = []
    if len(accepted_groups) >= 2:
        for size in range(2, len(accepted_groups) + 1):
            combination_candidates.extend(itertools.combinations(accepted_groups, size))
        combination_screening, combination_cache = evaluate_inner_candidates(
            raw, target, folds, combination_candidates
        )
        screening = pd.concat([screening, combination_screening], ignore_index=True)
        cache.update(combination_cache)
        summaries.extend(
            gate_summary(screening, candidate_name(groups)) for groups in combination_candidates
        )

    summary_frame = pd.DataFrame(summaries).sort_values(
        ["gate_passed", "mean_inner_macro_f1"], ascending=[False, False]
    )
    passing = summary_frame[summary_frame["gate_passed"]]
    locked_name = str(passing.iloc[0]["candidate"]) if len(passing) else None
    locked_groups = tuple(locked_name.split("+")) if locked_name else ()

    screening.to_csv(TABLES / f"{PREFIX}_inner_screening.csv", index=False)
    summary_frame.to_csv(TABLES / f"{PREFIX}_inner_gate_summary.csv", index=False)
    feature_schema(GROUPS).to_csv(TABLES / f"{PREFIX}_feature_schema.csv", index=False)

    metadata: dict[str, Any] = {
        "experiment": PREFIX,
        "rows": len(raw),
        "benchmark_macro_f1": 0.719483,
        "benchmark_integrity": benchmark_integrity,
        "control": "A+B + franchise history",
        "outer_folds": 5,
        "inner_folds": 4,
        "outer_seed": SEED,
        "inner_seed": INNER_SEED,
        "fixed_xgboost_parameters": PARAMS,
        "gate": {
            "minimum_mean_macro_f1_delta": MIN_MEAN_DELTA,
            "minimum_partition_wins": MIN_PARTITION_WINS,
            "minimum_mean_recall_class_0_delta": MIN_RECALL_0_DELTA,
        },
        "distributor_status": "blocked_no_approved_source",
        "accepted_single_groups": list(accepted_groups),
        "locked_groups": list(locked_groups),
        "outer_evaluation_run": False,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }

    if not locked_groups:
        metadata["status"] = "all_blocks_rejected_by_inner_gate_outer_not_run"
        metadata["runtime_seconds"] = time.perf_counter() - started
        (TABLES / f"{PREFIX}_run_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("All entity blocks failed the inner gate. Outer validation was not run.")
        print(summary_frame.round(6).to_string(index=False))
        return

    predictions, parameters, outer_metrics, fold_metrics, confusion = run_outer_once(
        raw, target, folds, locked_groups, cache
    )
    predictions.to_csv(TABLES / f"{PREFIX}_oof_predictions.csv", index=False)
    parameters.to_csv(TABLES / f"{PREFIX}_parameters.csv", index=False)
    outer_metrics.to_csv(TABLES / f"{PREFIX}_metrics.csv", index=False)
    fold_metrics.to_csv(TABLES / f"{PREFIX}_fold_metrics.csv", index=False)
    confusion.to_csv(TABLES / f"{PREFIX}_confusion_matrix.csv")
    metadata["outer_evaluation_run"] = True
    metadata["status"] = "locked_combination_outer_evaluated_once"
    metadata["runtime_seconds"] = time.perf_counter() - started
    (TABLES / f"{PREFIX}_run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Locked groups: {locked_groups}")
    print(outer_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
