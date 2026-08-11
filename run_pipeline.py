"""Điểm vào duy nhất để chạy pipeline Movie Success Analysis.

Ví dụ:
    python run_pipeline.py all
    python run_pipeline.py preprocess
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STAGES: dict[str, list[Path]] = {
    "collect": [ROOT / "src" / "collect_tmdb_full.py"],
    "preprocess": [ROOT / "src" / "preprocess_movies.py"],
    "enrich": [ROOT / "src" / "collect_tmdb_enrichment.py"],
    "eda": [ROOT / "src" / "eda_movies.py"],
    "evaluate": [
        ROOT / "src" / "reproduce_operational_ab_baseline.py",
        ROOT / "src" / "evaluate_operational_franchise.py",
    ],
    "train": [ROOT / "src" / "train_final_xgboost.py"],
    "report": [ROOT / "src" / "report_xgboost_results.py"],
    "figures": [ROOT / "src" / "generate_report_figures.py"],
}

PIPELINE_ORDER = ["collect", "preprocess", "enrich", "eda", "evaluate", "train", "report", "figures"]


def run_script(script: Path) -> None:
    command = [sys.executable, str(script)]
    print(f"\n[RUN] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[*STAGES, "all"],
        help="Công đoạn cần chạy. Dùng 'all' để chạy toàn bộ theo đúng thứ tự.",
    )
    arguments = parser.parse_args()

    selected = PIPELINE_ORDER if arguments.stage == "all" else [arguments.stage]
    for stage in selected:
        print(f"\n=== {stage.upper()} ===", flush=True)
        for script in STAGES[stage]:
            run_script(script)

    print("\nPipeline hoàn tất.")


if __name__ == "__main__":
    main()
