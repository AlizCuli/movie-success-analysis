"""Audit entity-history coverage, missingness, duplicates, and time safety."""

from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.entity_history_schema import (
    CAST_PATH,
    COMPANIES_PATH,
    DIRECTORS_PATH,
    MANIFEST_PATH,
    MOVIES_PATH,
    REQUEST_RUNS_PATH,
    ROOT,
    load_config,
    load_modeling,
    read_jsonl,
    target_entity_relations,
)
from src.reproduce_operational_ab_baseline import outer_assignments


TABLES = ROOT / "reports" / "tables"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def as_frame(path: Path) -> pd.DataFrame:
    records = read_jsonl(path)
    return pd.DataFrame(records)


def target_relation_frame(group: str, relations: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    frame = pd.DataFrame(relations[group])
    if frame.empty:
        return frame
    frame["target_release_date"] = pd.to_datetime(frame["target_release_date"], errors="coerce")
    return frame


def history_frame(path: Path) -> pd.DataFrame:
    frame = as_frame(path)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "movie_tmdb_id", "entity_id", "release_date", "budget", "revenue",
                "budget_valid", "revenue_valid", "financial_pair_valid",
            ]
        )
    frame["release_date"] = pd.to_datetime(frame["release_date"], errors="coerce")
    for column in ("movie_tmdb_id", "entity_id"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    return frame


def build_training_edges(
    group_relations: pd.DataFrame,
    modeling: pd.DataFrame,
    allowed_movie_ids: set[int],
) -> pd.DataFrame:
    if group_relations.empty:
        return pd.DataFrame()
    source = modeling[
        ["tmdb_id", "release_date", "budget", "revenue"]
    ].copy()
    source["release_date"] = pd.to_datetime(source["release_date"], errors="coerce")
    source["budget"] = pd.to_numeric(source["budget"], errors="coerce")
    source["revenue"] = pd.to_numeric(source["revenue"], errors="coerce")
    source = source[source["tmdb_id"].isin(allowed_movie_ids)]
    edges = group_relations.rename(columns={"target_tmdb_id": "movie_tmdb_id"})[
        ["movie_tmdb_id", "entity_id"]
    ].merge(source, left_on="movie_tmdb_id", right_on="tmdb_id", how="inner", validate="m:1")
    edges["budget_valid"] = edges["budget"].gt(0).astype(int)
    edges["revenue_valid"] = edges["revenue"].gt(0).astype(int)
    edges["financial_pair_valid"] = (edges["budget"].gt(0) & edges["revenue"].gt(0)).astype(int)
    return edges.drop(columns="tmdb_id")


def counts_for_movie(
    movie_id: int,
    query_date: pd.Timestamp,
    entity_ids: set[int],
    history: pd.DataFrame,
    maturity_days: int,
) -> tuple[int, int, int]:
    if not entity_ids or pd.isna(query_date) or history.empty:
        return 0, 0, 0
    eligible = history[
        history["entity_id"].isin(entity_ids)
        & history["release_date"].notna()
        & history["release_date"].lt(query_date)
        & history["movie_tmdb_id"].ne(movie_id)
    ]
    prior_count = int(eligible["movie_tmdb_id"].nunique())
    mature_cutoff = query_date - timedelta(days=maturity_days)
    mature = eligible[
        eligible["release_date"].le(mature_cutoff)
        & eligible["financial_pair_valid"].eq(1)
    ]
    mature_count = int(mature["movie_tmdb_id"].nunique())
    entity_with_history = int(eligible["entity_id"].nunique())
    return prior_count, mature_count, entity_with_history


def fold_coverage(
    group: str,
    relation: pd.DataFrame,
    external: pd.DataFrame,
    modeling: pd.DataFrame,
    folds: pd.Series,
    maturity_days: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    relation_by_movie = {
        int(movie_id): set(values["entity_id"].astype(int))
        for movie_id, values in relation.groupby("target_tmdb_id")
    } if not relation.empty else {}
    modeling_by_id = modeling.set_index("tmdb_id")

    for outer_fold in range(1, 6):
        validation_ids = set(modeling.loc[folds.eq(outer_fold), "tmdb_id"].astype(int))
        training_ids = set(modeling.loc[folds.ne(outer_fold), "tmdb_id"].astype(int))
        training_edges = build_training_edges(relation, modeling, training_ids)
        history = pd.concat([external, training_edges], ignore_index=True)
        if not history.empty:
            history = history.drop_duplicates(["movie_tmdb_id", "entity_id"])

        movie_counts = []
        mature_counts = []
        seen_entities = []
        for movie_id in sorted(validation_ids):
            query_date = pd.to_datetime(modeling_by_id.loc[movie_id, "release_date"], errors="coerce")
            counts = counts_for_movie(
                movie_id,
                query_date,
                relation_by_movie.get(movie_id, set()),
                history,
                maturity_days,
            )
            movie_counts.append(counts[0])
            mature_counts.append(counts[1])
            seen_entities.append(counts[2])
        values = np.asarray(movie_counts, dtype=int)
        rows.append({
            "entity_group": group,
            "outer_fold": outer_fold,
            "validation_movies": len(validation_ids),
            "entity_match_rate": float(np.mean([movie_id in relation_by_movie for movie_id in validation_ids])),
            "unseen_movie_rate": float(np.mean(values == 0)),
            "at_least_1_history_rate": float(np.mean(values >= 1)),
            "at_least_2_history_rate": float(np.mean(values >= 2)),
            "at_least_3_history_rate": float(np.mean(values >= 3)),
            "at_least_5_history_rate": float(np.mean(values >= 5)),
            "mature_financial_history_rate": float(np.mean(np.asarray(mature_counts) >= 1)),
            "mean_entities_with_history": float(np.mean(seen_entities)),
            "median_prior_movie_count": float(np.median(values)),
        })
    return pd.DataFrame(rows)


def duplicate_audit(group: str, frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"entity_group": group, "rows": 0, "duplicate_rows": 0, "duplicate_rate": 0.0}
    duplicate = frame.duplicated(["movie_tmdb_id", "entity_id", "role"], keep=False)
    return {
        "entity_group": group,
        "rows": len(frame),
        "duplicate_rows": int(duplicate.sum()),
        "duplicate_rate": float(duplicate.mean()),
    }


def source_audit() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "entity_group": "director",
            "candidate_source": "TMDb /person/{person_id}/movie_credits",
            "stable_entity_id": "TMDb person ID",
            "approved": True,
            "status": "approved",
            "reason": "Movie credits expose stable person and movie IDs plus crew job.",
        },
        {
            "entity_group": "production_company",
            "candidate_source": "TMDb /discover/movie?with_companies={company_id}",
            "stable_entity_id": "TMDb company ID",
            "approved": True,
            "status": "approved",
            "reason": "Company relation is verified using stable IDs; it is not called a distributor.",
        },
        {
            "entity_group": "cast",
            "candidate_source": "TMDb /person/{person_id}/movie_credits",
            "stable_entity_id": "TMDb person ID",
            "approved": True,
            "status": "approved",
            "reason": "Movie credits expose stable IDs and billing order.",
        },
        {
            "entity_group": "distributor",
            "candidate_source": "Wikidata P750 joined by P4947/P345",
            "stable_entity_id": "Wikidata QID",
            "approved": False,
            "status": "blocked_pending_provenance_and_coverage_pilot",
            "reason": "TMDb production companies are not distributors; Wikidata coverage and theatrical semantics are not yet verified.",
        },
    ])


def main() -> None:
    config = load_config()
    modeling = load_modeling()
    target = modeling["is_successful"].astype(int)
    folds = outer_assignments(modeling, target)
    relations = target_entity_relations(int(config["top_cast_k"]))
    paths = {
        "director": DIRECTORS_PATH,
        "production_company": COMPANIES_PATH,
        "cast": CAST_PATH,
    }
    TABLES.mkdir(parents=True, exist_ok=True)

    coverage_frames = []
    missing_rows = []
    duplicate_rows = []
    leakage_rows = []
    target_id_set = set(modeling["tmdb_id"].astype(int))

    for group, path in paths.items():
        relation = target_relation_frame(group, relations)
        history = history_frame(path)
        coverage_frames.append(
            fold_coverage(
                group,
                relation,
                history,
                modeling,
                folds,
                int(config["maturity_lag_days"]),
            )
        )
        duplicate_rows.append(duplicate_audit(group, history))
        missing_rows.append({
            "entity_group": group,
            "history_rows": len(history),
            "release_date_coverage": float(history["release_date"].notna().mean()) if len(history) else 0.0,
            "budget_coverage": float(history["budget_valid"].eq(1).mean()) if len(history) else 0.0,
            "revenue_coverage": float(history["revenue_valid"].eq(1).mean()) if len(history) else 0.0,
            "financial_pair_coverage": float(history["financial_pair_valid"].eq(1).mean()) if len(history) else 0.0,
            "unique_entities": int(history["entity_id"].nunique()) if len(history) else 0,
            "unique_history_movies": int(history["movie_tmdb_id"].nunique()) if len(history) else 0,
        })
        leakage_rows.append({
            "entity_group": group,
            "target_ids_in_external_history": int(history["movie_tmdb_id"].isin(target_id_set).sum()) if len(history) else 0,
            "missing_release_dates": int(history["release_date"].isna().sum()) if len(history) else 0,
            "release_after_snapshot_limit": int((history["release_date"] > pd.Timestamp(config["release_date_max"])).sum()) if len(history) else 0,
        })

    coverage = pd.concat(coverage_frames, ignore_index=True)
    coverage.to_csv(TABLES / "entity_history_coverage_by_fold.csv", index=False)
    coverage.groupby("entity_group", as_index=False).mean(numeric_only=True).to_csv(
        TABLES / "entity_history_coverage_by_group.csv", index=False
    )
    pd.DataFrame(missing_rows).to_csv(TABLES / "entity_history_missingness.csv", index=False)
    pd.DataFrame(duplicate_rows).to_csv(TABLES / "entity_history_duplicates.csv", index=False)
    leakage = pd.DataFrame(leakage_rows)
    leakage.to_csv(TABLES / "entity_history_leakage_audit.csv", index=False)
    source_audit().to_csv(TABLES / "entity_history_source_audit.csv", index=False)
    request_runs = as_frame(REQUEST_RUNS_PATH)
    if not request_runs.empty:
        request_runs.to_csv(TABLES / "entity_history_request_summary.csv", index=False)

    movie_history = as_frame(MOVIES_PATH)
    summary = {
        "manifest_exists": MANIFEST_PATH.exists(),
        "history_movies": len(movie_history),
        "coverage_rows": len(coverage),
        "target_id_leakage_rows": int(leakage["target_ids_in_external_history"].sum()),
        "distributor_status": "blocked_pending_provenance_and_coverage_pilot",
    }
    (TABLES / "entity_history_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if summary["target_id_leakage_rows"]:
        raise ValueError("Target movie IDs leaked into the external history warehouse.")
    print("Entity history audit: PASS")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(coverage.groupby("entity_group", as_index=False).mean(numeric_only=True).to_string(index=False))


if __name__ == "__main__":
    main()
