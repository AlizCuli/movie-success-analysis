"""Shared schema and local-cache helpers for entity history enrichment."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterable, Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "entity_history_v1.json"
BASE_RAW_PATH = ROOT / "data" / "raw" / "tmdb_movies_2000_2025.json"
ENRICHMENT_PATH = ROOT / "data" / "raw" / "tmdb_movie_enrichment.jsonl"
MODELING_PATH = ROOT / "data" / "processed" / "movies_modeling.csv"
HISTORY_ROOT = ROOT / "data" / "external" / "entity_history" / "v1"
CHECKPOINT_ROOT = HISTORY_ROOT / "checkpoints"

MOVIES_PATH = HISTORY_ROOT / "movies.jsonl.gz"
DIRECTORS_PATH = HISTORY_ROOT / "directors.jsonl.gz"
COMPANIES_PATH = HISTORY_ROOT / "production_companies.jsonl.gz"
CAST_PATH = HISTORY_ROOT / "top_cast.jsonl.gz"
DISTRIBUTORS_PATH = HISTORY_ROOT / "distributors.jsonl.gz"
ENTITIES_PATH = HISTORY_ROOT / "entities.jsonl.gz"
MANIFEST_PATH = HISTORY_ROOT / "manifest.json"
ERROR_PATH = HISTORY_ROOT / "request_failures.jsonl"
REQUEST_RUNS_PATH = HISTORY_ROOT / "request_runs.jsonl"

PERSON_INDEX_PATH = CHECKPOINT_ROOT / "person_movie_credits.jsonl"
COMPANY_INDEX_PATH = CHECKPOINT_ROOT / "company_movies.jsonl"
MOVIE_DETAILS_PATH = CHECKPOINT_ROOT / "movie_details.jsonl"

ENTITY_TYPES = ("director", "production_company", "cast", "distributor")


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    handle_context = (
        gzip.open(path, mode="rt", encoding="utf-8")
        if path.suffix == ".gz"
        else path.open(mode="r", encoding="utf-8")
    )
    with handle_context as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
    return records


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    for record in read_jsonl(path):
        yield record


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def write_jsonl_gz_atomic(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    count = 0
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    temporary.replace(path)
    return count


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_base_movies() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(BASE_RAW_PATH.read_text(encoding="utf-8"))
    movies = payload.get("movie_details") or []
    if len(movies) != 2597:
        raise ValueError(f"Expected 2,597 base movies, found {len(movies)}.")
    ids = [int(movie["id"]) for movie in movies]
    if len(ids) != len(set(ids)):
        raise ValueError("Base TMDb cache contains duplicate movie IDs.")
    return movies, payload.get("metadata") or {}


def load_enrichment() -> dict[int, dict[str, Any]]:
    records = read_jsonl(ENRICHMENT_PATH)
    by_id = {int(record["tmdb_id"]): record for record in records}
    if len(records) != len(by_id):
        raise ValueError("TMDb enrichment contains duplicate movie IDs.")
    return by_id


def load_modeling() -> pd.DataFrame:
    frame = pd.read_csv(MODELING_PATH)
    if len(frame) != 1646 or frame["tmdb_id"].duplicated().any():
        raise ValueError("The fixed modeling set must contain 1,646 unique tmdb_id values.")
    frame["tmdb_id"] = frame["tmdb_id"].astype(int)
    return frame


def target_ids() -> set[int]:
    return set(load_modeling()["tmdb_id"].astype(int))


def positive_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def normalized_movie_record(
    movie: dict[str, Any],
    source_endpoint: str,
    retrieved_at_utc: str,
    snapshot_version: str,
) -> dict[str, Any]:
    movie_id = int(movie.get("id") or movie.get("tmdb_id"))
    budget = positive_number(movie.get("budget"))
    revenue = positive_number(movie.get("revenue"))
    release_date = str(movie.get("release_date") or "").strip() or None
    return {
        "movie_tmdb_id": movie_id,
        "imdb_id": movie.get("imdb_id"),
        "release_date": release_date,
        "budget": budget,
        "revenue": revenue,
        "budget_valid": int(budget is not None),
        "revenue_valid": int(revenue is not None),
        "financial_pair_valid": int(budget is not None and revenue is not None),
        "adult": bool(movie.get("adult", False)),
        "status": movie.get("status"),
        "source": "TMDb Official API",
        "source_endpoint": source_endpoint,
        "retrieved_at_utc": retrieved_at_utc,
        "snapshot_version": snapshot_version,
        "schema_version": "1.0.0",
    }


@lru_cache(maxsize=4)
def target_entity_relations(top_cast_k: int = 5) -> dict[str, list[dict[str, Any]]]:
    """Extract stable target entity IDs without using names as join keys."""
    modeling = load_modeling().set_index("tmdb_id", drop=False)
    enrichment = load_enrichment()
    base_movies, _ = load_base_movies()
    base_by_id = {int(movie["id"]): movie for movie in base_movies}
    output = {"director": [], "production_company": [], "cast": []}

    for movie_id, target in modeling.iterrows():
        record = enrichment[int(movie_id)]
        release_date = target.get("release_date")
        credits = record.get("credits") or {}
        directors = [
            item for item in credits.get("crew") or []
            if item.get("job") == "Director" and item.get("id") is not None
        ]
        seen_directors: set[int] = set()
        for item in directors:
            entity_id = int(item["id"])
            if entity_id in seen_directors:
                continue
            seen_directors.add(entity_id)
            output["director"].append({
                "target_tmdb_id": int(movie_id),
                "target_release_date": release_date,
                "entity_id": entity_id,
                "entity_name": item.get("name"),
                "role": "Director",
                "credit_id": item.get("credit_id"),
                "billing_order": None,
            })

        cast = sorted(
            [item for item in credits.get("cast") or [] if item.get("id") is not None],
            key=lambda item: int(item.get("order", 10**9)),
        )[:top_cast_k]
        for item in cast:
            output["cast"].append({
                "target_tmdb_id": int(movie_id),
                "target_release_date": release_date,
                "entity_id": int(item["id"]),
                "entity_name": item.get("name"),
                "role": "Cast",
                "credit_id": item.get("credit_id"),
                "billing_order": int(item.get("order", 0)),
            })

        companies = base_by_id[int(movie_id)].get("production_companies") or []
        seen_companies: set[int] = set()
        for order, item in enumerate(companies):
            if item.get("id") is None:
                continue
            entity_id = int(item["id"])
            if entity_id in seen_companies:
                continue
            seen_companies.add(entity_id)
            output["production_company"].append({
                "target_tmdb_id": int(movie_id),
                "target_release_date": release_date,
                "entity_id": entity_id,
                "entity_name": item.get("name"),
                "role": "Production Company",
                "credit_id": None,
                "billing_order": order,
            })
    return output
