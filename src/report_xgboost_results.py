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
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
PREDICTIONS = TABLES / "operational_franchise_oof_predictions.csv"
BUNDLE = ROOT / "models" / "xgboost_pre_release_operational_bundle.joblib"
BENCHMARK = TABLES / "operational_franchise_metrics.csv"
FOLD_METRICS = TABLES / "xgboost_fold_metrics.csv"
CONFUSION_MATRIX = TABLES / "xgboost_confusion_matrix.csv"
FEATURE_IMPORTANCE = TABLES / "xgboost_feature_importance.csv"


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


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


def arrays_from_confusion_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Khôi phục nhãn tổng hợp để tính metric mà không cần dữ liệu từng phim."""
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    actual = np.concatenate(
        [
            np.zeros(true_negative + false_positive, dtype=int),
            np.ones(false_negative + true_positive, dtype=int),
        ]
    )
    predicted = np.concatenate(
        [
            np.zeros(true_negative, dtype=int),
            np.ones(false_positive, dtype=int),
            np.zeros(false_negative, dtype=int),
            np.ones(true_positive, dtype=int),
        ]
    )
    return actual, predicted


def load_aggregate_artifacts() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    required = [FOLD_METRICS, CONFUSION_MATRIX, FEATURE_IMPORTANCE, BENCHMARK]
    missing = [path for path in required if not path.exists()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Thiếu artifact tổng hợp. Hãy chạy stage evaluate trước: " + names
        )
    fold_metrics = pd.read_csv(FOLD_METRICS)
    matrix = pd.read_csv(CONFUSION_MATRIX, index_col=0).to_numpy(dtype=int)
    importance = pd.read_csv(FEATURE_IMPORTANCE)
    if fold_metrics["outer_fold"].nunique() != 5 or matrix.shape != (2, 2):
        raise ValueError("Artifact benchmark tổng hợp không đúng schema 5-fold/2 lớp.")
    return fold_metrics, matrix, importance


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    if PREDICTIONS.exists():
        predictions = pd.read_csv(PREDICTIONS)
        if predictions.empty or predictions["tmdb_id"].duplicated().any():
            raise ValueError("OOF predictions phải có tmdb_id duy nhất và không được trống.")
        if not np.isfinite(predictions["probability_success"]).all():
            raise ValueError("OOF probability chứa NaN/Inf.")

        fold_rows = []
        for fold, group in predictions.groupby("outer_fold", sort=True):
            fold_rows.append(
                {
                    "outer_fold": int(fold),
                    "rows": len(group),
                    "threshold": float(group["threshold"].iloc[0]),
                    **metrics(
                        group["success"].to_numpy(int),
                        group["prediction"].to_numpy(int),
                    ),
                }
            )
        fold_metrics = pd.DataFrame(fold_rows)
        fold_metrics.to_csv(FOLD_METRICS, index=False)

        actual = predictions["success"].to_numpy(int)
        predicted = predictions["prediction"].to_numpy(int)
        matrix = confusion_matrix(actual, predicted, labels=[0, 1])
        pd.DataFrame(
            matrix,
            index=["actual_0", "actual_1"],
            columns=["predicted_0", "predicted_1"],
        ).to_csv(CONFUSION_MATRIX, index=True)

        bundle = joblib.load(BUNDLE)
        importance = pd.DataFrame(
            {
                "feature": bundle["feature_names"],
                "importance": bundle["model"].feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        importance.to_csv(FEATURE_IMPORTANCE, index=False)
    else:
        fold_metrics, matrix, importance = load_aggregate_artifacts()
        actual, predicted = arrays_from_confusion_matrix(matrix)

    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
    axis.set_title("Ma trận nhầm lẫn outer-OOF của XGBoost")
    axis.set_xlabel("Dự đoán")
    axis.set_ylabel("Thực tế")
    axis.set_xticklabels(["Không thành công", "Thành công"])
    axis.set_yticklabels(["Không thành công", "Thành công"], rotation=0)
    save_figure(figure, "xgboost_confusion_matrix.png")

    figure, axis = plt.subplots(figsize=(8, 5))
    sns.barplot(data=fold_metrics, x="outer_fold", y="macro_f1", color="#2A6F97", ax=axis)
    pooled_macro_f1 = f1_score(actual, predicted, average="macro")
    axis.axhline(
        pooled_macro_f1,
        color="#E76F51",
        linestyle="--",
        label=f"Pooled {pooled_macro_f1:.6f}",
    )
    axis.set_ylim(0, 1)
    axis.set_title("Macro-F1 theo fold kiểm định ngoài")
    axis.set_xlabel("Fold kiểm định ngoài")
    axis.set_ylabel("Macro-F1")
    axis.legend()
    save_figure(figure, "xgboost_fold_macro_f1.png")

    top = importance.head(20).sort_values("importance").copy()
    top["display_feature"] = (
        top["feature"]
        .str.replace(r"^(numeric|categorical)__", "", regex=True)
        .str.replace("_infrequent_sklearn", " = nhóm ít gặp", regex=False)
    )
    figure, axis = plt.subplots(figsize=(10, 7))
    axis.barh(top["display_feature"], top["importance"], color="#2A9D8F")
    axis.set_title("20 đặc trưng có importance cao nhất của model cuối")
    axis.set_xlabel("Mức độ quan trọng của XGBoost")
    axis.set_ylabel("Đặc trưng sau tiền xử lý")
    save_figure(figure, "xgboost_feature_importance.png")

    pooled = metrics(actual, predicted)
    pd.DataFrame([{"rows": len(actual), **pooled}]).to_csv(
        TABLES / "xgboost_pooled_metrics.csv", index=False
    )
    print("XGBoost report artifacts: PASS")
    print(f"Macro-F1: {pooled['macro_f1']:.6f}")
    source = "OOF local" if PREDICTIONS.exists() else "artifact tổng hợp công khai"
    print(f"Nguồn tái tạo báo cáo: {source}")
    print(f"Created 4 aggregate tables and 3 figures in {ROOT / 'reports'}")


if __name__ == "__main__":
    main()
