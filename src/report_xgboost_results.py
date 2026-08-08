"""Tạo bảng và biểu đồ báo cáo cho benchmark XGBoost outer-OOF."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pre_release_features import load_pre_release_modeling_data


TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
PREDICTIONS = TABLES / "operational_franchise_oof_predictions.csv"
BUNDLE = ROOT / "models" / "xgboost_pre_release_operational_bundle.joblib"


def metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(actual, prediction, average="macro", zero_division=0)),
        "f1_class_0": float(f1_score(actual, prediction, pos_label=0, zero_division=0)),
        "f1_class_1": float(f1_score(actual, prediction, pos_label=1, zero_division=0)),
        "recall_class_0": float(recall_score(actual, prediction, pos_label=0, zero_division=0)),
        "recall_class_1": float(recall_score(actual, prediction, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, prediction)),
    }


def save_figure(figure: plt.Figure, name: str) -> None:
    figure.savefig(FIGURES / name, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    predictions = pd.read_csv(PREDICTIONS)
    if len(predictions) != 1646 or predictions["tmdb_id"].duplicated().any():
        raise ValueError("OOF predictions phải có đúng 1.646 tmdb_id duy nhất.")
    if not np.isfinite(predictions["probability_success"]).all():
        raise ValueError("OOF probability chứa NaN/Inf.")

    fold_rows = []
    for fold, group in predictions.groupby("outer_fold", sort=True):
        fold_rows.append(
            {
                "outer_fold": int(fold),
                "rows": len(group),
                "threshold": float(group["threshold"].iloc[0]),
                **metrics(group["success"].to_numpy(int), group["prediction"].to_numpy(int)),
            }
        )
    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(TABLES / "xgboost_fold_metrics.csv", index=False)

    actual = predictions["success"].to_numpy(int)
    predicted = predictions["prediction"].to_numpy(int)
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    pd.DataFrame(
        matrix,
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    ).to_csv(TABLES / "xgboost_confusion_matrix.csv", index=True)

    raw, _ = load_pre_release_modeling_data()
    errors = predictions.merge(
        raw[["tmdb_id", "title", "release_year", "budget", "revenue", "revenue_to_budget"]],
        on="tmdb_id",
        how="left",
        validate="1:1",
    )
    errors["error_type"] = np.select(
        [
            errors["success"].eq(0) & errors["prediction"].eq(1),
            errors["success"].eq(1) & errors["prediction"].eq(0),
        ],
        ["false_positive", "false_negative"],
        default="correct",
    )
    errors["distance_to_target_boundary"] = (errors["revenue_to_budget"] - 2).abs()
    errors.to_csv(TABLES / "xgboost_error_analysis.csv", index=False)

    bundle = joblib.load(BUNDLE)
    importance = pd.DataFrame(
        {
            "feature": bundle["feature_names"],
            "importance": bundle["model"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(TABLES / "xgboost_feature_importance.csv", index=False)

    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
    axis.set_title("XGBoost outer-OOF confusion matrix")
    axis.set_xlabel("Dự đoán")
    axis.set_ylabel("Thực tế")
    axis.set_xticklabels(["Không thành công", "Thành công"])
    axis.set_yticklabels(["Không thành công", "Thành công"], rotation=0)
    save_figure(figure, "09_xgboost_confusion_matrix.png")

    figure, axis = plt.subplots(figsize=(8, 5))
    sns.barplot(data=fold_metrics, x="outer_fold", y="macro_f1", color="#2A6F97", ax=axis)
    axis.axhline(predictions.pipe(lambda frame: f1_score(frame.success, frame.prediction, average="macro")), color="#E76F51", linestyle="--", label="Pooled 0,719483")
    axis.set_ylim(0, 1)
    axis.set_title("Macro-F1 theo outer fold")
    axis.set_xlabel("Outer fold")
    axis.set_ylabel("Macro-F1")
    axis.legend()
    save_figure(figure, "10_xgboost_fold_macro_f1.png")

    top = importance.head(20).sort_values("importance")
    figure, axis = plt.subplots(figsize=(10, 7))
    axis.barh(top["feature"], top["importance"], color="#2A9D8F")
    axis.set_title("Top 20 feature importance của model fit cuối")
    axis.set_xlabel("XGBoost feature importance")
    axis.set_ylabel("Feature đã biến đổi")
    save_figure(figure, "11_xgboost_feature_importance.png")

    figure, axis = plt.subplots(figsize=(9, 5))
    sns.histplot(
        data=predictions,
        x="probability_success",
        hue="success",
        bins=30,
        stat="density",
        common_norm=False,
        element="step",
        ax=axis,
    )
    axis.set_title("Phân bố xác suất outer-OOF theo lớp thực tế")
    axis.set_xlabel("Xác suất thành công")
    save_figure(figure, "12_xgboost_probability_distribution.png")

    pooled = metrics(actual, predicted)
    print("XGBoost report artifacts: PASS")
    print(f"Macro-F1: {pooled['macro_f1']:.6f}")
    print(f"Errors: {(errors['error_type'] != 'correct').sum()}")
    print(f"Created 4 tables and 4 figures in {ROOT / 'reports'}")


if __name__ == "__main__":
    main()
