"""Collect missing TMDb enrichment fields with checkpoint and resume support."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "tmdb_movies_2000_2025.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "tmdb_movie_enrichment.jsonl"
ERROR_PATH = PROJECT_ROOT / "data" / "raw" / "tmdb_movie_enrichment_errors.jsonl"
AUDIT_PATH = PROJECT_ROOT / "reports" / "tables" / "tmdb_enrichment_audit.csv"
API_URL = "https://api.themoviedb.org/3/movie/{movie_id}"
TIMEOUT_SECONDS = 30
MAX_RETRIES = 5
APPENDED_FIELDS = "credits,keywords,release_dates"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_base_movies() -> tuple[list[dict[str, Any]], str]:
    payload = json.loads(BASE_RAW_PATH.read_text(encoding="utf-8"))
    movies = payload["movie_details"]
    if len(movies) != 2597:
        raise ValueError(f"Dữ liệu gốc phải có 2.597 phim, hiện có {len(movies)}.")
    ids = [int(movie["id"]) for movie in movies]
    if len(ids) != len(set(ids)):
        raise ValueError("Dữ liệu gốc có tmdb_id trùng.")
    retrieved_at = payload.get("metadata", {}).get("last_updated_at", "")
    return movies, retrieved_at


def read_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                records[int(record["tmdb_id"])] = record
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"JSONL lỗi tại dòng {line_number}: {error}") from error
    return records


def normalize_checkpoint(records: dict[int, dict[str, Any]]) -> None:
    if not OUTPUT_PATH.exists():
        return
    line_count = sum(1 for line in OUTPUT_PATH.open("r", encoding="utf-8") if line.strip())
    if line_count == len(records):
        return
    temporary = OUTPUT_PATH.with_suffix(".jsonl.part")
    with temporary.open("w", encoding="utf-8") as handle:
        for movie_id in sorted(records):
            handle.write(json.dumps(records[movie_id], ensure_ascii=False) + "\n")
    temporary.replace(OUTPUT_PATH)


def present(value: Any) -> bool:
    return value not in (None, "", [], {})


def make_audit(
    base_movies: list[dict[str, Any]],
    enrichment: dict[int, dict[str, Any]],
    base_retrieved_at: str,
) -> pd.DataFrame:
    base_fields = [
        "belongs_to_collection",
        "production_companies",
        "production_countries",
        "spoken_languages",
        "overview",
        "tagline",
    ]
    appended_fields = ["credits", "keywords", "release_dates"]
    rows = []
    total = len(base_movies)
    for field in base_fields:
        count = sum(present(movie.get(field)) for movie in base_movies)
        rows.append(
            {
                "field": field,
                "source_cache": "tmdb_movies_2000_2025.json",
                "total_movies": total,
                "present_before_api": count,
                "missing_before_api": total - count,
                "coverage_before_api": count / total,
                "present_after_enrichment": count,
                "coverage_after_enrichment": count / total,
                "retrieved_at": base_retrieved_at,
            }
        )
    for field in appended_fields:
        count = sum(present(record.get(field)) for record in enrichment.values())
        rows.append(
            {
                "field": field,
                "source_cache": "tmdb_movie_enrichment.jsonl",
                "total_movies": total,
                "present_before_api": 0,
                "missing_before_api": total,
                "coverage_before_api": 0.0,
                "present_after_enrichment": count,
                "coverage_after_enrichment": count / total,
                "retrieved_at": max(
                    (str(record.get("retrieved_at", "")) for record in enrichment.values()),
                    default="",
                ),
            }
        )
    rows.append(
        {
            "field": "enrichment_record",
            "source_cache": "tmdb_movie_enrichment.jsonl",
            "total_movies": total,
            "present_before_api": 0,
            "missing_before_api": total,
            "coverage_before_api": 0.0,
            "present_after_enrichment": len(enrichment),
            "coverage_after_enrichment": len(enrichment) / total,
            "retrieved_at": max(
                (str(record.get("retrieved_at", "")) for record in enrichment.values()),
                default="",
            ),
        }
    )
    frame = pd.DataFrame(rows)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(AUDIT_PATH, index=False)
    return frame


def request_movie(session: requests.Session, token: str, movie_id: int) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}", "accept": "application/json"}
    params = {"append_to_response": APPENDED_FIELDS, "language": "en-US"}
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(
                API_URL.format(movie_id=movie_id),
                headers=headers,
                params=params,
                timeout=TIMEOUT_SECONDS,
            )
            if response.status_code == 200:
                return response.json()
            if response.status_code == 429 or 500 <= response.status_code < 600:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else 2**attempt
                time.sleep(min(delay, 30.0))
                continue
            if 400 <= response.status_code < 500:
                raise RuntimeError(f"HTTP {response.status_code} cho movie_id={movie_id}")
            response.raise_for_status()
        except (requests.RequestException, ValueError) as error:
            last_error = error
            if attempt == MAX_RETRIES - 1:
                break
            time.sleep(min(2**attempt, 30.0))
    raise RuntimeError(f"Không lấy được movie_id={movie_id} sau {MAX_RETRIES} lần") from last_error


def first_id(people: list[dict[str, Any]]) -> int | None:
    return int(people[0]["id"]) if people and people[0].get("id") is not None else None


def parse_release_dates(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results") or []
    events = []
    for country in results:
        country_code = country.get("iso_3166_1")
        for item in country.get("release_dates") or []:
            events.append({**item, "country": country_code})
    theatrical = [event for event in events if event.get("type") in (2, 3)]
    premiere = [event for event in events if event.get("type") == 1]

    def earliest(items: list[dict[str, Any]]) -> str | None:
        dates = sorted(str(item.get("release_date")) for item in items if item.get("release_date"))
        return dates[0] if dates else None

    certification_candidates = [
        event for event in theatrical if event.get("country") == "US" and str(event.get("certification", "")).strip()
    ]
    if not certification_candidates:
        certification_candidates = [
            event for event in theatrical if str(event.get("certification", "")).strip()
        ]
    certification = (
        str(certification_candidates[0]["certification"]).strip()
        if certification_candidates
        else None
    )
    return {
        "certification": certification,
        "theatrical_country_count": len({event.get("country") for event in theatrical if event.get("country")}),
        "release_event_count": len(events),
        "earliest_premiere_date": earliest(premiere),
        "earliest_theatrical_date": earliest(theatrical),
    }


def build_record(base: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    credits = response.get("credits") or {}
    cast = credits.get("cast") or []
    crew = credits.get("crew") or []
    directors = [person for person in crew if person.get("job") == "Director"]
    writers = [
        person
        for person in crew
        if person.get("department") == "Writing"
        or person.get("job") in {"Writer", "Screenplay", "Story"}
    ]
    producers = [
        person for person in crew if person.get("job") in {"Producer", "Executive Producer"}
    ]
    composers = [person for person in crew if person.get("job") == "Original Music Composer"]
    keywords_payload = response.get("keywords") or {}
    keywords = keywords_payload.get("keywords") or keywords_payload.get("results") or []
    companies = base.get("production_companies") or response.get("production_companies") or []
    countries = base.get("production_countries") or response.get("production_countries") or []
    languages = base.get("spoken_languages") or response.get("spoken_languages") or []
    collection = base.get("belongs_to_collection") or response.get("belongs_to_collection")
    top_cast_ids = [int(person["id"]) for person in cast[:5] if person.get("id") is not None]
    writer_ids = list(dict.fromkeys(int(person["id"]) for person in writers if person.get("id") is not None))
    producer_ids = list(dict.fromkeys(int(person["id"]) for person in producers if person.get("id") is not None))
    overview = base.get("overview") or response.get("overview") or ""
    tagline = base.get("tagline") or response.get("tagline") or ""
    record = {
        "tmdb_id": int(base["id"]),
        "release_date": base.get("release_date"),
        "collection_id": int(collection["id"]) if collection and collection.get("id") is not None else None,
        "collection_name": collection.get("name") if collection else None,
        "is_collection": int(collection is not None),
        "cast_count": len(cast),
        "crew_count": len(crew),
        "top_cast_ids": top_cast_ids,
        "director_id": first_id(directors),
        "writer_id": writer_ids[0] if writer_ids else None,
        "writer_ids": writer_ids,
        "producer_id": producer_ids[0] if producer_ids else None,
        "producer_ids": producer_ids,
        "composer_id": first_id(composers),
        "keyword_ids": [int(item["id"]) for item in keywords if item.get("id") is not None],
        "keyword_names": [str(item["name"]).strip() for item in keywords if item.get("name")],
        "overview": overview,
        "tagline": tagline,
        "overview_word_count": len(str(overview).split()),
        "has_tagline": int(bool(str(tagline).strip())),
        "keyword_count": len(keywords),
        "company_ids": [int(item["id"]) for item in companies if item.get("id") is not None],
        "primary_company_id": int(companies[0]["id"]) if companies and companies[0].get("id") is not None else None,
        "company_count": len(companies),
        "production_country_count": len(countries),
        "spoken_language_count": len(languages),
        "credits": {"cast": cast, "crew": crew},
        "keywords": keywords,
        "release_dates": response.get("release_dates") or {},
        "retrieved_at": utc_now(),
    }
    record.update(parse_release_dates(record["release_dates"]))
    return record


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def collect(force: bool = False, audit_only: bool = False) -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    base_movies, base_retrieved_at = load_base_movies()
    records = {} if force else read_jsonl(OUTPUT_PATH)
    if force and OUTPUT_PATH.exists():
        backup = OUTPUT_PATH.with_suffix(".jsonl.previous")
        OUTPUT_PATH.replace(backup)
    normalize_checkpoint(records)
    audit = make_audit(base_movies, records, base_retrieved_at)
    print(audit.to_string(index=False))
    if audit_only:
        return
    token = os.getenv("TMDB_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TMDB_API_TOKEN chưa có giá trị trong .env.")
    pending = [movie for movie in base_movies if int(movie["id"]) not in records]
    print(f"Đã có checkpoint: {len(records)}; còn thiếu: {len(pending)}")
    errors = 0
    with requests.Session() as session:
        for position, movie in enumerate(pending, start=1):
            movie_id = int(movie["id"])
            try:
                response = request_movie(session, token, movie_id)
                record = build_record(movie, response)
                append_jsonl(OUTPUT_PATH, record)
                records[movie_id] = record
            except Exception as error:  # keep collecting and preserve checkpoint
                errors += 1
                append_jsonl(
                    ERROR_PATH,
                    {"tmdb_id": movie_id, "retrieved_at": utc_now(), "error": str(error)},
                )
                print(f"Lỗi movie_id={movie_id}: {error}", file=sys.stderr)
            if position % 50 == 0 or position == len(pending):
                print(f"Tiến độ {position}/{len(pending)}; thành công tổng={len(records)}; lỗi lượt này={errors}")
    normalize_checkpoint(records)
    make_audit(base_movies, records, base_retrieved_at)
    print(f"Hoàn tất: {len(records)}/2597 phim; lỗi chưa phục hồi trong lượt này: {errors}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Tải lại toàn bộ enrichment.")
    parser.add_argument("--audit-only", action="store_true", help="Chỉ ghi bảng kiểm kê, không gọi API.")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    collect(force=arguments.force, audit_only=arguments.audit_only)
