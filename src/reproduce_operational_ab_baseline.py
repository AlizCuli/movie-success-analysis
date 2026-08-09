"""Reproduce a fixed A+B operational baseline with the historical XGBoost protocol."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src.pre_release_features import PreReleaseFeatureBuilder, load_pre_release_modeling_data


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
SEED = 42
INNER_SEED = 43
THRESHOLDS = np.round(np.arange(0.20, 0.801, 0.01), 2)
PARAMS = {
    "n_estimators": 800,
    "learning_rate": 0.064,
    "max_depth": 4,
    "min_child_weight": 12,
    "subsample": 0.68,
    "colsample_bytree": 0.88,
    "gamma": 1.6,
    "reg_alpha": 0.14,
    "reg_lambda": 0.93,
    "class_weight_0": 1.5,
    "min_frequency": 10,
}


def outer_assignments(data: pd.DataFrame, target: pd.Series) -> pd.Series:
    """Tái tạo cố định outer split 5-fold bằng seed 42."""
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = np.zeros(len(data), dtype=int)
    for fold, (_, valid_index) in enumerate(splitter.split(data, target), start=1):
        folds[valid_index] = fold
    return pd.Series(folds, index=data.index, dtype=int).reset_index(drop=True)


def preprocessor(builder: PreReleaseFeatureBuilder) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), builder.numeric_columns_),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=PARAMS["min_frequency"],
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                builder.categorical_columns_,
            ),
        ],
        sparse_threshold=0.0,
    )


def model(iterations: int, seed_offset: int, early_stopping: bool) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=int(iterations),
        learning_rate=PARAMS["learning_rate"],
        max_depth=PARAMS["max_depth"],
        min_child_weight=PARAMS["min_child_weight"],
        subsample=PARAMS["subsample"],
        colsample_bytree=PARAMS["colsample_bytree"],
        gamma=PARAMS["gamma"],
        reg_alpha=PARAMS["reg_alpha"],
        reg_lambda=PARAMS["reg_lambda"],
        random_state=SEED + seed_offset,
        n_jobs=4,
        verbosity=0,
        early_stopping_rounds=50 if early_stopping else None,
    )


def weights(target: pd.Series) -> np.ndarray:
    return np.where(target.to_numpy(dtype=int) == 0, PARAMS["class_weight_0"], 1.0)


def choose_threshold(target: pd.Series, probability: np.ndarray) -> tuple[float, float]:
    scores = [
        (float(f1_score(target, probability >= threshold, average="macro", zero_division=0)), float(threshold))
        for threshold in THRESHOLDS
    ]
    return max(scores, key=lambda item: (item[0], -abs(item[1] - 0.5)))[1], max(scores, key=lambda item: (item[0], -abs(item[1] - 0.5)))[0]


def inner_oof(raw: pd.DataFrame, target: pd.Series, outer_fold: int) -> tuple[np.ndarray, list[int]]:
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=INNER_SEED)
    probability = np.full(len(raw), np.nan)
    iterations: list[int] = []
    for inner_fold, (train_index, valid_index) in enumerate(splitter.split(raw, target), start=1):
        train_raw, valid_raw = raw.iloc[train_index], raw.iloc[valid_index]
        train_y, valid_y = target.iloc[train_index], target.iloc[valid_index]
        builder = PreReleaseFeatureBuilder()
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
        raise ValueError("Inner OOF contains NaN/Inf.")
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
        builder = PreReleaseFeatureBuilder()
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
    if len(predictions) != len(raw) or not predictions.tmdb_id.is_unique or not np.isfinite(predictions.probability_success).all():
        raise ValueError("Outer OOF validation failed.")
    y = predictions.success.to_numpy(dtype=int)
    pred = predictions.prediction.to_numpy(dtype=int)
    summary = pd.DataFrame([{
        "model": "xgboost_operational_a_b_fixed", "scope": "pre_release_operational",
        "rows": len(predictions), "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "f1_class_0": f1_score(y, pred, pos_label=0, zero_division=0),
        "recall_class_0": recall_score(y, pred, pos_label=0, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, pred), "accuracy": accuracy_score(y, pred),
    }])
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"tmdb_id": raw.tmdb_id, "success": target, "outer_fold": folds}).to_csv(TABLES / "operational_ab_fixed_outer_fold_assignments.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(TABLES / "operational_ab_fixed_parameters.csv", index=False)
    predictions.to_csv(TABLES / "operational_ab_fixed_oof_predictions.csv", index=False)
    summary.to_csv(TABLES / "operational_ab_fixed_metrics.csv", index=False)
    print("Operational A+B fixed baseline: PASS")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
