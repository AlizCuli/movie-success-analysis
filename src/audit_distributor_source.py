"""Audit whether current project sources identify real movie distributors."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from src.entity_history_schema import ENRICHMENT_PATH, ROOT, load_base_movies


OUTPUT = ROOT / "reports" / "tables" / "distributor_source_audit.csv"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def first_enrichment_record() -> dict:
    with ENRICHMENT_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
    raise ValueError("TMDb enrichment cache is empty.")


def main() -> None:
    base_movies, _ = load_base_movies()
    base_fields = set(base_movies[0])
    enrichment = first_enrichment_record()
    release_results = (enrichment.get("release_dates") or {}).get("results") or []
    release_fields: set[str] = set()
    for country in release_results:
        for item in country.get("release_dates") or []:
            release_fields.update(item)

    rows = [
        {
            "source": "TMDb movie details production_companies",
            "stable_movie_join": True,
            "stable_distributor_id": False,
            "contains_actual_distributor": False,
            "approved": False,
            "evidence": f"production_companies_present={'production_companies' in base_fields}; field describes producers, not distributors",
        },
        {
            "source": "TMDb movie release_dates",
            "stable_movie_join": True,
            "stable_distributor_id": False,
            "contains_actual_distributor": False,
            "approved": False,
            "evidence": "release fields=" + "|".join(sorted(release_fields)),
        },
        {
            "source": "IMDb Non-Commercial Datasets",
            "stable_movie_join": True,
            "stable_distributor_id": False,
            "contains_actual_distributor": False,
            "approved": False,
            "evidence": "official non-commercial files do not include a company/distributor table",
        },
        {
            "source": "Wikidata P750 joined by P4947/P345",
            "stable_movie_join": "candidate",
            "stable_distributor_id": True,
            "contains_actual_distributor": "not_yet_verified",
            "approved": False,
            "evidence": "requires a separate coverage, reference, territory and theatrical-semantics pilot",
        },
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)
    print("Distributor source audit: BLOCKED")
    print(f"Report: {OUTPUT}")
    print("production_companies was not substituted for distributor.")


if __name__ == "__main__":
    main()
