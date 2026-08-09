"""Huấn luyện và đóng gói XGBoost cuối bằng cấu hình benchmark đã khóa."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate_operational_franchise import inner_oof
from src.operational_franchise_features import OperationalFranchiseBuilder
from src.pre_release_features import ENRICHMENT_PATH, FORBIDDEN_PREDICTORS, MODELING_PATH
from src.pre_release_features import load_pre_release_modeling_data
from src.reproduce_operational_ab_baseline import (
    PARAMS,
    choose_threshold,
    model,
    preprocessor,
    weights,
)


MODELS_DIR = ROOT / "models"
TABLES_DIR = ROOT / "reports" / "tables"
BUNDLE_PATH = MODELS_DIR / "xgboost_pre_release_operational_bundle.joblib"
MODEL_PATH = MODELS_DIR / "xgboost_pre_release_operational_model.json"
MANIFEST_PATH = MODELS_DIR / "xgboost_pre_release_operational_manifest.json"
SUMMARY_PATH = TABLES_DIR / "xgboost_final_training_summary.csv"
BENCHMARK_PATH = TABLES_DIR / "operational_franchise_metrics.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    raw, target = load_pre_release_modeling_data()

    # Đây chỉ là chọn số vòng và threshold cho model fit cuối; không tuning lại
    # hyperparameter và không thay thế kết quả nested outer-OOF 0,719483.
    inner_probability, iterations = inner_oof(
        raw.reset_index(drop=True),
        target.reset_index(drop=True),
        outer_fold=0,
    )
    threshold, inner_macro_f1 = choose_threshold(target, inner_probability)
    selected_iterations = int(np.median(iterations))

    builder = OperationalFranchiseBuilder(smoothing=10.0)
    features = builder.fit_transform(raw.reset_index(drop=True), target.reset_index(drop=True))
    forbidden = sorted(set(features.columns) & FORBIDDEN_PREDICTORS)
    if forbidden:
        raise ValueError(f"Predictor bị cấm trong model cuối: {forbidden}")

    transformer = preprocessor(builder)
    matrix = transformer.fit_transform(features)
    fitted_model = model(selected_iterations, seed_offset=900_000, early_stopping=False)
    fitted_model.fit(matrix, target, sample_weight=weights(target), verbose=False)

    feature_names = transformer.get_feature_names_out().tolist()
    benchmark = pd.read_csv(BENCHMARK_PATH).iloc[0].to_dict()
    created_at = datetime.now(timezone.utc).isoformat()
    bundle = {
        "artifact": "xgboost_pre_release_operational",
        "feature_builder": builder,
        "preprocessor": transformer,
        "model": fitted_model,
        "threshold": float(threshold),
        "feature_names": feature_names,
        "raw_feature_columns": builder.feature_columns_,
        "benchmark_outer_oof": benchmark,
        "created_at_utc": created_at,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, BUNDLE_PATH, compress=3)
    fitted_model.save_model(MODEL_PATH)

    manifest = {
        "package_format_version": 1,
        "artifact": bundle["artifact"],
        "artifact_files": {
            "bundle": BUNDLE_PATH.name,
            "native_xgboost_model": MODEL_PATH.name,
        },
        "artifact_sha256": {
            BUNDLE_PATH.name: sha256(BUNDLE_PATH),
            MODEL_PATH.name: sha256(MODEL_PATH),
        },
        "created_at_utc": created_at,
        "training_rows": int(len(raw)),
        "class_counts": {
            str(label): int(count) for label, count in target.value_counts().sort_index().items()
        },
        "target_definition": "is_successful = 1 if revenue >= 2 * budget else 0",
        "scope": "pre_release_operational",
        "benchmark_outer_oof": benchmark,
        "final_inner_oof_macro_f1": float(inner_macro_f1),
        "final_threshold": float(threshold),
        "selected_n_estimators": selected_iterations,
        "fixed_parameters": PARAMS,
        "raw_feature_count": int(len(builder.feature_columns_)),
        "transformed_feature_count": int(len(feature_names)),
        "raw_feature_columns": builder.feature_columns_,
        "forbidden_predictors_found": forbidden,
        "data_sha256": {
            str(MODELING_PATH.relative_to(ROOT)): sha256(MODELING_PATH),
            str(ENRICHMENT_PATH.relative_to(ROOT)): sha256(ENRICHMENT_PATH),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
        "interpretation_note": (
            f"{float(benchmark['macro_f1']):.6f} is nested outer-OOF performance. "
            f"The packaged model is fit on all {len(raw):,} rows and has no "
            "separate performance estimate."
        ),
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {
                "artifact": bundle["artifact"],
                "training_rows": len(raw),
                "selected_n_estimators": selected_iterations,
                "threshold": threshold,
                "inner_oof_macro_f1": inner_macro_f1,
                "benchmark_outer_oof_macro_f1": float(benchmark["macro_f1"]),
                "raw_feature_count": len(builder.feature_columns_),
                "transformed_feature_count": len(feature_names),
            }
        ]
    ).to_csv(SUMMARY_PATH, index=False)

    print("Final XGBoost package: PASS")
    print(f"Training rows: {len(raw)}")
    print(f"Selected iterations: {selected_iterations}")
    print(f"Threshold: {threshold:.2f}")
    print(f"Benchmark outer-OOF Macro-F1: {float(benchmark['macro_f1']):.6f}")
    print(f"Bundle: {BUNDLE_PATH}")


if __name__ == "__main__":
    main()
