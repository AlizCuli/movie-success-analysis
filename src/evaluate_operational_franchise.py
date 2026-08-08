"""Evaluate franchise history against the fixed A+B operational baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold

from src.pre_release_features import load_pre_release_modeling_data
from src.operational_franchise_features import OperationalFranchiseBuilder
from src.reproduce_operational_ab_baseline import (
    PARAMS,
    choose_threshold,
    model,
    outer_assignments,
    preprocessor,
    weights,
)


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
INNER_SEED = 43


def inner_oof(raw: pd.DataFrame, target: pd.Series, outer_fold: int) -> tuple[np.ndarray, list[int]]:
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=INNER_SEED)
    probability = np.full(len(raw), np.nan)
    iterations: list[int] = []
    for inner_fold, (train_index, valid_index) in enumerate(splitter.split(raw, target), start=1):
        train_raw, valid_raw = raw.iloc[train_index], raw.iloc[valid_index]
        train_y, valid_y = target.iloc[train_index], target.iloc[valid_index]
        builder = OperationalFranchiseBuilder(smoothing=10.0)
        train_features = builder.fit_transform(train_raw.reset_index(drop=True), train_y.reset_index(drop=True))
        valid_features = builder.transform(valid_raw.reset_index(drop=True))
        transformer = preprocessor(builder)
        train_matrix = transformer.fit_transform(train_features)
        valid_matrix = transformer.transform(valid_features)
        fitted = model(PARAMS["n_estimators"], inner_fold, early_stopping=True)
        fitted.fit(train_matrix, train_y, sample_weight=weights(train_y), eval_set=[(valid_matrix, valid_y)], verbose=False)
        probability[valid_index] = fitted.predict_proba(valid_matrix)[:, 1]
        iterations.append(int(getattr(fitted, "best_iteration", PARAMS["n_estimators"] - 1)) + 1)
    if not np.isfinite(probability).all():
        raise ValueError("Franchise inner OOF contains NaN/Inf.")
    return probability, iterations


def main() -> None:
    raw, target = load_pre_release_modeling_data()
    folds = outer_assignments(raw, target)
    prediction_rows = []
    parameter_rows = []
    for outer_fold in range(1, 6):
        valid_index = np.flatnonzero(folds.to_numpy() == outer_fold)
        train_index = np.flatnonzero(folds.to_numpy() != outer_fold)
        train_raw, valid_raw = raw.iloc[train_index], raw.iloc[valid_index]
        train_y, valid_y = target.iloc[train_index], target.iloc[valid_index]
        inner_probability, iterations = inner_oof(train_raw.reset_index(drop=True), train_y.reset_index(drop=True), outer_fold)
        threshold, inner_macro_f1 = choose_threshold(train_y.reset_index(drop=True), inner_probability)
        selected_iterations = int(np.median(iterations))
        builder = OperationalFranchiseBuilder(smoothing=10.0)
        train_features = builder.fit_transform(train_raw.reset_index(drop=True), train_y.reset_index(drop=True))
        valid_features = builder.transform(valid_raw.reset_index(drop=True))
        transformer = preprocessor(builder)
        train_matrix = transformer.fit_transform(train_features)
        valid_matrix = transformer.transform(valid_features)
        fitted = model(selected_iterations, outer_fold, early_stopping=False)
        fitted.fit(train_matrix, train_y, sample_weight=weights(train_y), verbose=False)
        probability = fitted.predict_proba(valid_matrix)[:, 1]
        prediction = (probability >= threshold).astype(int)
        parameter_rows.append({"outer_fold": outer_fold, "selected_iterations": selected_iterations, "inner_oof_threshold": threshold, "inner_oof_macro_f1": inner_macro_f1, "raw_feature_count": len(builder.feature_columns_), "transformed_feature_count": len(transformer.get_feature_names_out()), **PARAMS})
        prediction_rows.extend({"tmdb_id": int(raw.iloc[original_index]["tmdb_id"]), "success": int(target.iloc[original_index]), "outer_fold": outer_fold, "probability_success": float(probability[position]), "threshold": threshold, "prediction": int(prediction[position])} for position, original_index in enumerate(valid_index))

    predictions = pd.DataFrame(prediction_rows).sort_values("tmdb_id")
    if len(predictions) != 1646 or not predictions.tmdb_id.is_unique or not np.isfinite(predictions.probability_success).all():
        raise ValueError("Franchise outer OOF validation failed.")
    y = predictions.success.to_numpy(dtype=int)
    pred = predictions.prediction.to_numpy(dtype=int)
    summary = pd.DataFrame([{
        "model": "xgboost_operational_a_b_franchise_history", "scope": "pre_release_operational",
        "rows": len(predictions), "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "f1_class_0": f1_score(y, pred, pos_label=0, zero_division=0),
        "recall_class_0": recall_score(y, pred, pos_label=0, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, pred), "accuracy": accuracy_score(y, pred),
    }])
    baseline = pd.read_csv(TABLES / "operational_ab_fixed_metrics.csv")
    comparison = pd.concat([baseline.assign(reference="a_b_fixed"), summary.assign(reference="franchise_history")], ignore_index=True)
    comparison["macro_f1_delta_vs_a_b_fixed"] = comparison["macro_f1"] - float(baseline.loc[0, "macro_f1"])
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"tmdb_id": raw.tmdb_id, "success": target, "outer_fold": folds}).to_csv(TABLES / "operational_franchise_outer_fold_assignments.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(TABLES / "operational_franchise_parameters.csv", index=False)
    predictions.to_csv(TABLES / "operational_franchise_oof_predictions.csv", index=False)
    summary.to_csv(TABLES / "operational_franchise_metrics.csv", index=False)
    comparison.to_csv(TABLES / "operational_franchise_comparison.csv", index=False)
    print("Operational franchise history: PASS")
    print(summary.to_string(index=False))
    print(comparison[["model", "macro_f1", "f1_class_0", "recall_class_0", "balanced_accuracy", "macro_f1_delta_vs_a_b_fixed"]].to_string(index=False))


if __name__ == "__main__":
    main()
