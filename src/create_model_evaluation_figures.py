"""Create separate, report-ready XGBoost evaluation figures."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
METRICS_PATH = TABLES / "operational_franchise_metrics.csv"
MATRIX_PATH = TABLES / "xgboost_confusion_matrix.csv"
SUMMARY_OUTPUT = FIGURES / "xgboost_performance_summary.png"
MATRIX_OUTPUT = FIGURES / "xgboost_confusion_matrix_outer_oof.png"


def restore_predictions(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def create_summary(metrics: pd.Series, matrix: np.ndarray) -> None:
    actual, predicted = restore_predictions(matrix)
    values = [
        ("Macro-F1", float(metrics["macro_f1"])),
        ("Accuracy", float(metrics["accuracy"])),
        ("Balanced accuracy", float(metrics["balanced_accuracy"])),
        ("F1 lớp không thành công", float(metrics["f1_class_0"])),
        ("F1 lớp thành công", float(f1_score(actual, predicted, pos_label=1))),
    ]
    navy = "#193B5A"
    text = "#202B36"
    grid = "#D8E1E8"

    figure, axis = plt.subplots(figsize=(6.8, 4.35))
    figure.patch.set_facecolor("white")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    row_top = 0.84
    row_height = 0.14
    for index, (label, value) in enumerate(values):
        y = row_top - index * row_height
        axis.text(0.06, y, label, ha="left", va="center", fontsize=12.0, color=text)
        axis.text(0.94, y, f"{value:.6f}", ha="right", va="center", fontsize=12.5, fontweight="bold", color=navy)
        axis.plot([0.06, 0.94], [y - 0.070, y - 0.070], color=grid, linewidth=0.9)

    axis.text(0.06, 0.09, f"Số phim được đánh giá: {int(metrics['rows']):,}", ha="left", va="center", fontsize=11.5, fontweight="bold", color=text)
    figure.savefig(SUMMARY_OUTPUT, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def create_confusion_matrix(matrix: np.ndarray) -> None:
    navy = "#153E75"
    pale = "#F4F8FC"
    figure, axis = plt.subplots(figsize=(6.2, 5.7), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=int(matrix.max()))
    del image
    labels = ["Không thành công", "Thành công"]
    axis.set_xticks([0, 1], labels=labels)
    axis.set_yticks([0, 1], labels=labels)
    axis.set_xlabel("Dự đoán", fontsize=12)
    axis.set_ylabel("Thực tế", fontsize=12)
    axis.set_title("Ma trận nhầm lẫn XGBoost (outer-OOF)", fontsize=15, fontweight="bold", pad=12)
    axis.tick_params(axis="both", labelsize=10.5)
    threshold = matrix.max() / 2
    for row in range(2):
        for column in range(2):
            color = "white" if matrix[row, column] > threshold else "#202B36"
            axis.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center", fontsize=17, color=color)
    axis.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=2)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.set_facecolor(pale)
    figure.savefig(MATRIX_OUTPUT, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    metrics_frame = pd.read_csv(METRICS_PATH)
    if len(metrics_frame) != 1:
        raise ValueError("Bảng benchmark phải chứa đúng một dòng.")
    metrics = metrics_frame.iloc[0]
    matrix = pd.read_csv(MATRIX_PATH, index_col=0).to_numpy(dtype=int)
    if matrix.shape != (2, 2) or matrix.sum() != int(metrics["rows"]):
        raise ValueError("Ma trận nhầm lẫn không khớp số phim benchmark.")
    create_summary(metrics, matrix)
    create_confusion_matrix(matrix)
    print(f"summary={SUMMARY_OUTPUT}")
    print(f"matrix={MATRIX_OUTPUT}")


if __name__ == "__main__":
    main()
