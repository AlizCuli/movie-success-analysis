"""Recreate the nine figures versioned in ``reports/figures``.

The public repository keeps the rendered PNG files and the generators, while
the row-level TMDb data remain local.  A clone can therefore view the figures
immediately and regenerate them after recreating the private data artifacts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.create_model_evaluation_figures import create_summary
from src.create_raw_tmdb_feature_map import main as create_raw_feature_map
from src.create_theatrical_release_figures import main as create_theatrical_figures
from src.eda_movies import run_eda
from src.report_xgboost_results import main as create_xgboost_figures


FIGURES_DIR = ROOT / "reports" / "figures"
MODEL_METRICS = ROOT / "reports" / "tables" / "operational_franchise_metrics.csv"
MODEL_MATRIX = ROOT / "reports" / "tables" / "xgboost_confusion_matrix.csv"

REQUIRED_LOCAL_DATA = [
    ROOT / "data" / "processed" / "movies_cleaned.csv",
    ROOT / "data" / "processed" / "movies_modeling.csv",
    ROOT / "data" / "raw" / "tmdb_movie_enrichment.jsonl",
]

EXPECTED_FIGURES = [
    "dataset_overview.png",
    "pre_release_spearman_heatmap.png",
    "success_by_genre_collection.png",
    "tmdb_raw_feature_map.png",
    "success_rate_by_theatrical_release_breadth.png",
    "success_rate_by_release_breadth_and_collection.png",
    "xgboost_performance_summary.png",
    "xgboost_confusion_matrix.png",
    "xgboost_fold_macro_f1.png",
]


def check_inputs() -> None:
    missing = [path for path in REQUIRED_LOCAL_DATA if not path.exists()]
    missing_tables = [path for path in (MODEL_METRICS, MODEL_MATRIX) if not path.exists()]
    if missing or missing_tables:
        details = [str(path.relative_to(ROOT)) for path in [*missing, *missing_tables]]
        raise FileNotFoundError(
            "Thiếu đầu vào để tái tạo hình: "
            + ", ".join(details)
            + ". Dữ liệu cấp phim không được phân phối trong repository công khai."
        )


def create_model_summary() -> None:
    metrics_frame = pd.read_csv(MODEL_METRICS)
    matrix = pd.read_csv(MODEL_MATRIX, index_col=0).to_numpy(dtype=int)
    if len(metrics_frame) != 1 or matrix.shape != (2, 2):
        raise ValueError("Artifact XGBoost không đúng schema tổng hợp.")
    if matrix.sum() != int(metrics_frame.iloc[0]["rows"]):
        raise ValueError("Ma trận nhầm lẫn không khớp số phim benchmark.")
    create_summary(metrics_frame.iloc[0], matrix)


def validate_outputs() -> None:
    missing = [name for name in EXPECTED_FIGURES if not (FIGURES_DIR / name).exists()]
    if missing:
        raise RuntimeError("Chưa tạo đủ 9 hình: " + ", ".join(missing))
    empty = [name for name in EXPECTED_FIGURES if (FIGURES_DIR / name).stat().st_size == 0]
    if empty:
        raise RuntimeError("Hình có kích thước bằng 0: " + ", ".join(empty))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    check_inputs()

    # 3 hình EDA: overview, tương quan và genre/collection.
    run_eda()
    # 1 hình inventory metadata TMDb đầu vào.
    create_raw_feature_map()
    # 2 hình phân tích độ rộng phát hành rạp.
    create_theatrical_figures()
    # 1 hình tóm tắt benchmark + 2 hình đánh giá outer-OOF.
    create_model_summary()
    create_xgboost_figures(include_feature_importance=False)

    validate_outputs()
    print(f"Đã tái tạo {len(EXPECTED_FIGURES)} hình trong {FIGURES_DIR}.")
    for name in EXPECTED_FIGURES:
        print(f"- {name}")


if __name__ == "__main__":
    main()
