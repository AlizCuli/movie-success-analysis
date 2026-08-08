"""Collect a resumable TMDb history warehouse for target-linked entities.

The 1,646 target movies are never added as new labeled observations. They are
also excluded from the external history warehouse during finalization.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import msvcrt
import os
import shutil
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from src.entity_history_schema import (
    CAST_PATH,
    COMPANIES_PATH,
    COMPANY_INDEX_PATH,
    DIRECTORS_PATH,
    ENTITIES_PATH,
    ERROR_PATH,
    HISTORY_ROOT,
    MANIFEST_PATH,
    MOVIE_DETAILS_PATH,
    MOVIES_PATH,
    PERSON_INDEX_PATH,
    REQUEST_RUNS_PATH,
    append_jsonl,
    load_base_movies,
    load_config,
    normalized_movie_record,
    read_jsonl,
    sha256_file,
    target_entity_relations,
    target_ids,
    write_json_atomic,
    write_jsonl_gz_atomic,
)


API_ROOT = "https://api.themoviedb.org/3"
LOCK_PATH = HISTORY_ROOT / "collector.lock"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RequestBudgetReached(RuntimeError):
    """Stop cleanly after the requested number of API calls."""


class CollectorLock:
    """Prevent two collector instances from appending to one checkpoint."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Any = None

    def __enter__(self) -> "CollectorLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        self.handle.seek(0)
        if self.handle.read(1) == b"":
            self.handle.seek(0)
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            self.handle.close()
            raise RuntimeError("Another entity-history collector is already running.") from error
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        if self.handle is not None:
            self.handle.seek(0)
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            self.handle.close()


class TMDbClient:
    def __init__(self, token: str, config: dict[str, Any], max_requests: int = 0) -> None:
        self.token = token
        self.local = threading.local()
        self.sessions: list[requests.Session] = []
        self.session_lock = threading.Lock()
        self.rate_lock = threading.Lock()
        self.timeout = int(config["timeout_seconds"])
        self.max_retries = int(config["max_retries"])
        self.interval = 1.0 / float(config["requests_per_second"])
        self.max_requests = int(max_requests)
        self.request_count = 0
        self.last_request_at = 0.0

    def close(self) -> None:
        for session in self.sessions:
            session.close()

    def _session(self) -> requests.Session:
        session = getattr(self.local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update({
                "Authorization": f"Bearer {self.token}",
                "accept": "application/json",
            })
            self.local.session = session
            with self.session_lock:
                self.sessions.append(session)
        return session

    def _reserve_request_slot(self) -> None:
        with self.rate_lock:
            if self.max_requests and self.request_count >= self.max_requests:
                raise RequestBudgetReached(f"Request budget reached: {self.max_requests}")
            delay = self.interval - (time.monotonic() - self.last_request_at)
            if delay > 0:
                time.sleep(delay)
            self.request_count += 1
            self.last_request_at = time.monotonic()

    def get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                self._reserve_request_slot()
                response = self._session().get(
                    f"{API_ROOT}{endpoint}",
                    params=params or {},
                    timeout=self.timeout,
                )
                if response.status_code == 200:
                    return response.json()
                if response.status_code == 429 or 500 <= response.status_code < 600:
                    retry_after = response.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else 2**attempt
                    time.sleep(min(wait, 60.0))
                    continue
                raise RuntimeError(f"HTTP {response.status_code} for {endpoint}")
            except (requests.RequestException, ValueError) as error:
                last_error = error
                if attempt + 1 < self.max_retries:
                    time.sleep(min(2**attempt, 60.0))
        raise RuntimeError(f"Unrecoverable request failure for {endpoint}") from last_error


def log_failure(stage: str, entity_id: int, error: Exception) -> None:
    append_jsonl(
        ERROR_PATH,
        {
            "stage": stage,
            "entity_id": int(entity_id),
            "error_type": type(error).__name__,
            "error": str(error),
            "recorded_at_utc": utc_now(),
        },
    )


def valid_credit_date(value: Any, maximum: str) -> bool:
    date = str(value or "").strip()
    return bool(date) and date <= maximum


def bootstrap_movie_checkpoint(config: dict[str, Any]) -> int:
    existing = {int(record["movie_tmdb_id"]) for record in read_jsonl(MOVIE_DETAILS_PATH)}
    movies, metadata = load_base_movies()
    retrieved = str(metadata.get("last_updated_at") or metadata.get("created_at") or "")
    added = 0
    for movie in movies:
        movie_id = int(movie["id"])
        if movie_id in existing:
            continue
        append_jsonl(
            MOVIE_DETAILS_PATH,
            normalized_movie_record(
                movie,
                source_endpoint="local-cache:tmdb_movies_2000_2025.json",
                retrieved_at_utc=retrieved,
                snapshot_version=config["snapshot_version"],
            ),
        )
        existing.add(movie_id)
        added += 1
    return added


def repair_jsonl_checkpoint(path: Path, key_fields: tuple[str, ...]) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "rows": 0, "valid": 0, "invalid": 0, "duplicates": 0}
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_name(f"{path.name}.{timestamp}.backup")
    rejected = path.with_name(f"{path.name}.{timestamp}.rejected")
    shutil.copy2(path, backup)
    valid_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    rejected_lines: list[str] = []
    total = 0
    valid = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            total += 1
            try:
                record = json.loads(line)
                key = tuple(record[field] for field in key_fields)
                valid_by_key[key] = record
                valid += 1
            except (json.JSONDecodeError, KeyError, TypeError):
                rejected_lines.append(line.rstrip("\n"))
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8") as handle:
        for key in sorted(valid_by_key):
            handle.write(json.dumps(valid_by_key[key], ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)
    if rejected_lines:
        rejected.write_text("\n".join(rejected_lines) + "\n", encoding="utf-8")
    return {
        "path": str(path),
        "rows": total,
        "valid": valid,
        "invalid": len(rejected_lines),
        "duplicates": valid - len(valid_by_key),
        "recovered_unique": len(valid_by_key),
        "backup": str(backup),
        "rejected": str(rejected) if rejected_lines else None,
    }


def repair_checkpoints() -> None:
    results = [
        repair_jsonl_checkpoint(PERSON_INDEX_PATH, ("person_id",)),
        repair_jsonl_checkpoint(COMPANY_INDEX_PATH, ("company_id", "page")),
        repair_jsonl_checkpoint(MOVIE_DETAILS_PATH, ("movie_tmdb_id",)),
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))


def checkpoint_stats(config: dict[str, Any]) -> None:
    people = read_jsonl(PERSON_INDEX_PATH)
    company_pages = read_jsonl(COMPANY_INDEX_PATH)
    details = read_jsonl(MOVIE_DETAILS_PATH)
    person_ids = [int(row["person_id"]) for row in people]
    page_keys = [(int(row["company_id"]), int(row["page"])) for row in company_pages]
    detail_ids = [int(row["movie_tmdb_id"]) for row in details]
    failures = read_jsonl(ERROR_PATH)
    terminal_movie_ids = {
        int(row["entity_id"])
        for row in failures
        if row.get("stage") == "movie_details" and str(row.get("error", "")).startswith("HTTP 404")
    }
    discovered_ids = discovered_movie_ids()
    expected_people, names = seed_maps(config)
    expected_companies = {
        entity_id for entity_type, entity_id in names if entity_type == "production_company"
    }
    payload = {
        "expected_people": len(expected_people),
        "person_rows": len(person_ids),
        "unique_people": len(set(person_ids)),
        "missing_people": len(set(expected_people) - set(person_ids)),
        "expected_companies": len(expected_companies),
        "company_page_rows": len(page_keys),
        "unique_company_pages": len(set(page_keys)),
        "companies_with_at_least_one_page": len({company_id for company_id, _ in page_keys}),
        "movie_detail_rows": len(detail_ids),
        "unique_movie_details": len(set(detail_ids)),
        "discovered_unique_movie_ids": len(discovered_ids),
        "terminal_movie_404_ids": len(terminal_movie_ids),
        "pending_discovered_movie_ids": len(discovered_ids - set(detail_ids) - terminal_movie_ids),
        "request_failure_rows": len(failures),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def seed_maps(config: dict[str, Any]) -> tuple[dict[int, set[str]], dict[tuple[str, int], str | None]]:
    relations = target_entity_relations(int(config["top_cast_k"]))
    person_roles: dict[int, set[str]] = {}
    names: dict[tuple[str, int], str | None] = {}
    for group in ("director", "cast"):
        for row in relations[group]:
            entity_id = int(row["entity_id"])
            person_roles.setdefault(entity_id, set()).add(group)
            names[(group, entity_id)] = row.get("entity_name")
    for row in relations["production_company"]:
        names[("production_company", int(row["entity_id"]))] = row.get("entity_name")
    return person_roles, names


def collect_people(client: TMDbClient, config: dict[str, Any]) -> None:
    person_roles, _ = seed_maps(config)
    completed = {int(record["person_id"]) for record in read_jsonl(PERSON_INDEX_PATH)}
    pending = sorted(set(person_roles) - completed)
    print(f"People checkpoint: {len(completed)}; pending: {len(pending)}")
    maximum = str(config["release_date_max"])

    for position, person_id in enumerate(pending, start=1):
        try:
            payload = client.get(f"/person/{person_id}/movie_credits", {"language": "en-US"})
            director_movies = []
            if "director" in person_roles[person_id]:
                director_movies = [
                    {
                        "movie_tmdb_id": int(item["id"]),
                        "release_date": item.get("release_date"),
                        "credit_id": item.get("credit_id"),
                        "role": "Director",
                    }
                    for item in payload.get("crew") or []
                    if item.get("id") is not None
                    and item.get("job") == "Director"
                    and valid_credit_date(item.get("release_date"), maximum)
                    and not bool(item.get("adult", False))
                ]
            cast_movies = []
            if "cast" in person_roles[person_id]:
                cast_movies = [
                    {
                        "movie_tmdb_id": int(item["id"]),
                        "release_date": item.get("release_date"),
                        "credit_id": item.get("credit_id"),
                        "role": "Cast",
                        "billing_order": int(item["order"]) if item.get("order") is not None else None,
                        "character": item.get("character"),
                    }
                    for item in payload.get("cast") or []
                    if item.get("id") is not None
                    and valid_credit_date(item.get("release_date"), maximum)
                    and not bool(item.get("adult", False))
                ]
            append_jsonl(
                PERSON_INDEX_PATH,
                {
                    "person_id": person_id,
                    "seed_roles": sorted(person_roles[person_id]),
                    "director_movies": director_movies,
                    "cast_movies": cast_movies,
                    "source": "TMDb Official API",
                    "source_endpoint": f"/person/{person_id}/movie_credits",
                    "retrieved_at_utc": utc_now(),
                    "snapshot_version": config["snapshot_version"],
                },
            )
        except RequestBudgetReached:
            raise
        except Exception as error:  # checkpoint must survive individual failures
            log_failure("person_movie_credits", person_id, error)
        if position % 100 == 0 or position == len(pending):
            print(f"People progress: {position}/{len(pending)}; requests={client.request_count}")


def collect_companies(client: TMDbClient, config: dict[str, Any]) -> None:
    _, names = seed_maps(config)
    company_ids = sorted(entity_id for entity_type, entity_id in names if entity_type == "production_company")
    existing_rows = read_jsonl(COMPANY_INDEX_PATH)
    completed_pages = {(int(row["company_id"]), int(row["page"])) for row in existing_rows}
    known_total_pages: dict[int, int] = {}
    for row in existing_rows:
        known_total_pages[int(row["company_id"])] = int(row["total_pages"])

    for company_position, company_id in enumerate(company_ids, start=1):
        page = 1
        total_pages = known_total_pages.get(company_id, 1)
        while page <= total_pages:
            if (company_id, page) in completed_pages:
                page += 1
                continue
            try:
                payload = client.get(
                    "/discover/movie",
                    {
                        "with_companies": company_id,
                        "include_adult": "false",
                        "include_video": "false",
                        "primary_release_date.lte": config["release_date_max"],
                        "sort_by": "primary_release_date.asc",
                        "page": page,
                    },
                )
                total_pages = min(int(payload.get("total_pages") or 1), 500)
                known_total_pages[company_id] = total_pages
                movie_ids = [
                    int(item["id"])
                    for item in payload.get("results") or []
                    if item.get("id") is not None
                    and valid_credit_date(item.get("release_date"), str(config["release_date_max"]))
                    and not bool(item.get("adult", False))
                ]
                append_jsonl(
                    COMPANY_INDEX_PATH,
                    {
                        "company_id": company_id,
                        "page": page,
                        "total_pages": total_pages,
                        "movie_ids": movie_ids,
                        "source": "TMDb Official API",
                        "source_endpoint": "/discover/movie",
                        "retrieved_at_utc": utc_now(),
                        "snapshot_version": config["snapshot_version"],
                    },
                )
                completed_pages.add((company_id, page))
                page += 1
            except RequestBudgetReached:
                raise
            except Exception as error:
                log_failure("company_discover", company_id, error)
                break
        if company_position % 50 == 0 or company_position == len(company_ids):
            print(
                f"Company progress: {company_position}/{len(company_ids)}; "
                f"completed_pages={len(completed_pages)}; requests={client.request_count}"
            )


def discovered_movie_ids() -> set[int]:
    ids: set[int] = set()
    for record in read_jsonl(PERSON_INDEX_PATH):
        for key in ("director_movies", "cast_movies"):
            ids.update(int(item["movie_tmdb_id"]) for item in record.get(key) or [])
    for record in read_jsonl(COMPANY_INDEX_PATH):
        ids.update(int(movie_id) for movie_id in record.get("movie_ids") or [])
    return ids


def collect_movie_details(client: TMDbClient, config: dict[str, Any]) -> None:
    existing = {int(record["movie_tmdb_id"]) for record in read_jsonl(MOVIE_DETAILS_PATH)}
    terminal_movie_ids = {
        int(row["entity_id"])
        for row in read_jsonl(ERROR_PATH)
        if row.get("stage") == "movie_details" and str(row.get("error", "")).startswith("HTTP 404")
    }
    pending = sorted(discovered_movie_ids() - existing - terminal_movie_ids)
    print(f"Movie details checkpoint: {len(existing)}; pending discovered movies: {len(pending)}")
    workers = int(config.get("movie_detail_workers", 1))

    def fetch(movie_id: int) -> tuple[int, dict[str, Any] | None, Exception | None]:
        try:
            payload = client.get(f"/movie/{movie_id}", {"language": "en-US"})
            record = normalized_movie_record(
                payload,
                source_endpoint=f"/movie/{movie_id}",
                retrieved_at_utc=utc_now(),
                snapshot_version=config["snapshot_version"],
            )
            if record["release_date"] and record["release_date"] <= config["release_date_max"] and not record["adult"]:
                return movie_id, record, None
            return movie_id, None, None
        except RequestBudgetReached as error:
            return movie_id, None, error
        except Exception as error:
            return movie_id, None, error

    stopped_by_budget = False
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for chunk_start in range(0, len(pending), 500):
            chunk = pending[chunk_start:chunk_start + 500]
            for offset, (movie_id, record, error) in enumerate(executor.map(fetch, chunk), start=1):
                position = chunk_start + offset
                if isinstance(error, RequestBudgetReached):
                    stopped_by_budget = True
                elif error is not None:
                    log_failure("movie_details", movie_id, error)
                elif record is not None:
                    append_jsonl(MOVIE_DETAILS_PATH, record)
                if position % 500 == 0 or position == len(pending):
                    print(
                        f"Movie detail progress: {position}/{len(pending)}; "
                        f"requests={client.request_count}; workers={workers}"
                    )
            if stopped_by_budget:
                raise RequestBudgetReached(f"Request budget reached: {client.max_requests}")


def edge_with_finance(
    movie_id: int,
    entity_id: int,
    entity_type: str,
    role: str,
    details: dict[int, dict[str, Any]],
    source_endpoint: str,
    retrieved_at_utc: str,
    snapshot_version: str,
    credit_id: str | None = None,
    billing_order: int | None = None,
) -> dict[str, Any] | None:
    movie = details.get(int(movie_id))
    if movie is None:
        return None
    return {
        "movie_tmdb_id": int(movie_id),
        "entity_id": int(entity_id),
        "entity_type": entity_type,
        "role": role,
        "billing_order": billing_order,
        "credit_id": credit_id,
        "release_date": movie.get("release_date"),
        "budget": movie.get("budget"),
        "revenue": movie.get("revenue"),
        "budget_valid": int(bool(movie.get("budget_valid"))),
        "revenue_valid": int(bool(movie.get("revenue_valid"))),
        "financial_pair_valid": int(bool(movie.get("financial_pair_valid"))),
        "source": "TMDb Official API",
        "source_endpoint": source_endpoint,
        "retrieved_at_utc": retrieved_at_utc,
        "snapshot_version": snapshot_version,
        "schema_version": "1.0.0",
    }


def finalize(config: dict[str, Any]) -> dict[str, Any]:
    excluded_ids = target_ids()
    all_detail_records = read_jsonl(MOVIE_DETAILS_PATH)
    details = {
        int(record["movie_tmdb_id"]): record
        for record in all_detail_records
        if int(record["movie_tmdb_id"]) not in excluded_ids
        and record.get("release_date")
        and record["release_date"] <= config["release_date_max"]
        and not bool(record.get("adult", False))
    }
    people = read_jsonl(PERSON_INDEX_PATH)
    company_pages = read_jsonl(COMPANY_INDEX_PATH)

    director_edges: dict[tuple[int, int, str], dict[str, Any]] = {}
    cast_edges: dict[tuple[int, int], dict[str, Any]] = {}
    for person in people:
        person_id = int(person["person_id"])
        for item in person.get("director_movies") or []:
            edge = edge_with_finance(
                int(item["movie_tmdb_id"]), person_id, "director", "Director", details,
                person["source_endpoint"], person["retrieved_at_utc"], config["snapshot_version"],
                credit_id=item.get("credit_id"),
            )
            if edge:
                director_edges[(edge["movie_tmdb_id"], person_id, "Director")] = edge
        for item in person.get("cast_movies") or []:
            edge = edge_with_finance(
                int(item["movie_tmdb_id"]), person_id, "cast", "Cast", details,
                person["source_endpoint"], person["retrieved_at_utc"], config["snapshot_version"],
                credit_id=item.get("credit_id"), billing_order=item.get("billing_order"),
            )
            if edge:
                key = (edge["movie_tmdb_id"], person_id)
                previous = cast_edges.get(key)
                current_order = edge.get("billing_order")
                previous_order = previous.get("billing_order") if previous else None
                if previous is None or (
                    current_order is not None
                    and (previous_order is None or int(current_order) < int(previous_order))
                ):
                    cast_edges[key] = edge

    company_movie_pairs: dict[tuple[int, int], dict[str, Any]] = {}
    for page in company_pages:
        company_id = int(page["company_id"])
        for movie_id_value in page.get("movie_ids") or []:
            movie_id = int(movie_id_value)
            movie = details.get(movie_id)
            if movie is None:
                continue
            edge = edge_with_finance(
                movie_id, company_id, "production_company", "Production Company", details,
                page["source_endpoint"], page["retrieved_at_utc"], config["snapshot_version"],
            )
            if edge:
                company_movie_pairs[(movie_id, company_id)] = edge

    used_movie_ids = {
        edge["movie_tmdb_id"]
        for edge in [*director_edges.values(), *cast_edges.values(), *company_movie_pairs.values()]
    }
    movie_records = [details[movie_id] for movie_id in sorted(used_movie_ids)]
    _, names = seed_maps(config)
    entity_records = [
        {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "canonical_name": name,
            "name_is_join_key": False,
            "source": "TMDb Official API",
            "snapshot_version": config["snapshot_version"],
            "schema_version": "1.0.0",
        }
        for (entity_type, entity_id), name in sorted(names.items())
    ]

    counts = {
        "movies": write_jsonl_gz_atomic(MOVIES_PATH, movie_records),
        "director_edges": write_jsonl_gz_atomic(
            DIRECTORS_PATH, (director_edges[key] for key in sorted(director_edges))
        ),
        "production_company_edges": write_jsonl_gz_atomic(
            COMPANIES_PATH, (company_movie_pairs[key] for key in sorted(company_movie_pairs))
        ),
        "cast_edges": write_jsonl_gz_atomic(
            CAST_PATH, (cast_edges[key] for key in sorted(cast_edges))
        ),
        "entities": write_jsonl_gz_atomic(ENTITIES_PATH, entity_records),
    }
    files = [MOVIES_PATH, DIRECTORS_PATH, COMPANIES_PATH, CAST_PATH, ENTITIES_PATH]
    manifest = {
        "schema_version": config["schema_version"],
        "snapshot_version": config["snapshot_version"],
        "created_at_utc": utc_now(),
        "source": "TMDb Official API",
        "target_movie_count": len(excluded_ids),
        "target_movies_excluded_from_external_history": True,
        "distributor_status": "blocked_no_approved_source",
        "counts": counts,
        "files": {
            str(path.relative_to(HISTORY_ROOT)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        },
        "rules": {
            "release_date_max": config["release_date_max"],
            "maturity_lag_days": config["maturity_lag_days"],
            "top_cast_k": config["top_cast_k"],
            "movie_deduplication_key": "movie_tmdb_id",
            "entity_join_rule": "stable source ID only; never name",
        },
    }
    write_json_atomic(MANIFEST_PATH, manifest)
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    return manifest


def run(phase: str, max_requests: int) -> None:
    if phase == "repair":
        with CollectorLock(LOCK_PATH):
            repair_checkpoints()
        return
    if phase == "stats":
        checkpoint_stats(load_config())
        return
    with CollectorLock(LOCK_PATH):
        _run_locked(phase, max_requests)


def _run_locked(phase: str, max_requests: int) -> None:
    run_started = utc_now()
    config = load_config()
    added = bootstrap_movie_checkpoint(config)
    print(f"Bootstrapped movie detail records: {added}")
    if phase == "finalize":
        finalize(config)
        return

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    token = os.getenv("TMDB_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TMDB_API_TOKEN is empty in .env; token was not displayed.")
    client = TMDbClient(token, config, max_requests=max_requests)
    status = "completed"
    try:
        if phase in {"people", "all"}:
            collect_people(client, config)
        if phase in {"companies", "all"}:
            collect_companies(client, config)
        if phase in {"movies", "all"}:
            collect_movie_details(client, config)
        if phase == "all":
            finalize(config)
    except RequestBudgetReached as error:
        status = "request_budget_reached_checkpoint_saved"
        print(f"Checkpoint saved. {error}")
    except Exception:
        status = "failed_checkpoint_preserved"
        raise
    finally:
        client.close()
        append_jsonl(
            REQUEST_RUNS_PATH,
            {
                "phase": phase,
                "started_at_utc": run_started,
                "finished_at_utc": utc_now(),
                "request_count": client.request_count,
                "status": status,
                "snapshot_version": config["snapshot_version"],
            },
        )
        print(f"API requests in this run: {client.request_count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("people", "companies", "movies", "finalize", "repair", "stats", "all"),
        default="all",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=0,
        help="Stop after this many API calls; 0 means no explicit call budget.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.phase, arguments.max_requests)
