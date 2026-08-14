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

EXPECTED_TABLES = [
    "success_by_theatrical_release_breadth.csv",
    "success_by_release_breadth_and_collection.csv",
]


def check_inputs() -> None:
    missing = [path for path in REQUIRED_LOCAL_DATA if not path.exists()]
    missing_tables = [path for path in (MODEL_METRICS, MODEL_MATRIX) if not path.exists()]
    if missing or missing_tables:
        details = [str(path.relative_to(ROOT)) for path in [*missing, *missing_tables]]
        raise FileNotFoundError(
            "Missing inputs required to recreate figures: "
            + ", ".join(details)
            + ". Row-level data are not distributed in the public repository."
        )


def create_model_summary() -> None:
    metrics_frame = pd.read_csv(MODEL_METRICS)
    matrix = pd.read_csv(MODEL_MATRIX, index_col=0).to_numpy(dtype=int)
    if len(metrics_frame) != 1 or matrix.shape != (2, 2):
        raise ValueError("XGBoost aggregate artifacts do not match the expected schema.")
    if matrix.sum() != int(metrics_frame.iloc[0]["rows"]):
        raise ValueError("The confusion matrix does not match the benchmark row count.")
    create_summary(metrics_frame.iloc[0], matrix)


def validate_outputs() -> None:
    missing = [name for name in EXPECTED_FIGURES if not (FIGURES_DIR / name).exists()]
    if missing:
        raise RuntimeError("The figure set is incomplete: " + ", ".join(missing))
    empty = [name for name in EXPECTED_FIGURES if (FIGURES_DIR / name).stat().st_size == 0]
    if empty:
        raise RuntimeError("The following figures are empty: " + ", ".join(empty))
    missing_tables = [
        name for name in EXPECTED_TABLES
        if not (ROOT / "reports" / "tables" / name).exists()
    ]
    if missing_tables:
        raise RuntimeError("Missing public aggregate tables: " + ", ".join(missing_tables))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    check_inputs()

    # Three EDA figures: overview, correlations, and genre/collection.
    run_eda()
    # One inventory figure for the original TMDb metadata.
    create_raw_feature_map()
    # Two figures for theatrical-release breadth.
    create_theatrical_figures()
    # One benchmark summary and two outer-OOF evaluation figures.
    create_model_summary()
    create_xgboost_figures(include_feature_importance=False)

    validate_outputs()
    print(f"Recreated {len(EXPECTED_FIGURES)} figures in {FIGURES_DIR}.")
    for name in EXPECTED_FIGURES:
        print(f"- {name}")


if __name__ == "__main__":
    main()
