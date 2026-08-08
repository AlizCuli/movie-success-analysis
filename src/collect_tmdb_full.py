import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


API_BASE_URL = "https://api.themoviedb.org/3"
FIRST_YEAR = 2000
LAST_YEAR = 2025
MAX_PAGES_PER_YEAR = 5
MAX_RETRIES = 4
REQUEST_TIMEOUT_SECONDS = 30
REQUEST_DELAY_SECONDS = 0.1
CHECKPOINT_EVERY_DETAILS = 10

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
RAW_OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "tmdb_movies_2000_2025.json"
CSV_OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "tmdb_movies_2000_2025.csv"
ERROR_LOG_PATH = PROJECT_ROOT / "data" / "raw" / "tmdb_movies_2000_2025_errors.log"

COLUMNS = [
    "tmdb_id",
    "imdb_id",
    "title",
    "original_title",
    "release_date",
    "original_language",
    "budget",
    "revenue",
    "runtime",
    "vote_average",
    "vote_count",
    "popularity",
    "genres",
    "production_countries",
    "production_companies",
    "adult",
    "status",
]

REQUIRED_DETAIL_FIELDS = {
    "id",
    "imdb_id",
    "title",
    "original_title",
    "release_date",
    "original_language",
    "budget",
    "revenue",
    "runtime",
    "vote_average",
    "vote_count",
    "popularity",
    "genres",
    "production_countries",
    "production_companies",
    "adult",
    "status",
}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class RequestFailed(RuntimeError):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def setup_logger():
    logger = logging.getLogger("tmdb_full_collection")
    logger.setLevel(logging.ERROR)
    logger.handlers.clear()

    handler = logging.FileHandler(ERROR_LOG_PATH, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def new_state():
    return {
        "metadata": {
            "source": "TMDb API",
            "started_at": utc_now(),
            "last_updated_at": utc_now(),
            "elapsed_seconds": 0.0,
            "collection_range": [FIRST_YEAR, LAST_YEAR],
            "max_pages_per_year": MAX_PAGES_PER_YEAR,
        },
        "completed_years": [],
        "checked_tmdb_ids": [],
        "movie_details": [],
        "unrecoverable_errors": {},
    }


def load_state():
    if not RAW_OUTPUT_PATH.exists():
        return new_state()

    with RAW_OUTPUT_PATH.open(encoding="utf-8") as file:
        state = json.load(file)

    if not isinstance(state, dict):
        raise ValueError("Checkpoint JSON không có cấu trúc object hợp lệ.")

    state.setdefault("metadata", {})
    state.setdefault("completed_years", [])
    state.setdefault("checked_tmdb_ids", [])
    state.setdefault("movie_details", [])
    state.setdefault("unrecoverable_errors", {})

    if not isinstance(state["movie_details"], list):
        raise ValueError("Checkpoint JSON thiếu danh sách movie_details hợp lệ.")

    state["completed_years"] = sorted({int(year) for year in state["completed_years"]})
    state["checked_tmdb_ids"] = list({int(movie_id) for movie_id in state["checked_tmdb_ids"]})
    return state


def save_state(state):
    state["metadata"]["last_updated_at"] = utc_now()
    temporary_path = RAW_OUTPUT_PATH.with_suffix(".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False, indent=2)

    temporary_path.replace(RAW_OUTPUT_PATH)


def retry_delay(response, attempt):
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            return max(float(retry_after), 2**attempt)

    return float(2**attempt)


def get_json(session, url, params=None):
    last_message = "Không rõ lỗi request."
    last_status_code = None

    for attempt in range(MAX_RETRIES + 1):
        response = None
        try:
            response = session.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            status_code = response.status_code

            if status_code == 429 or 500 <= status_code <= 599:
                last_message = f"Lỗi HTTP {status_code}."
                last_status_code = status_code
            elif status_code >= 400:
                raise RequestFailed(f"Lỗi HTTP {status_code}.", status_code)
            else:
                try:
                    payload = response.json()
                except ValueError as error:
                    raise RequestFailed("Phản hồi không phải JSON hợp lệ.", status_code) from error

                time.sleep(REQUEST_DELAY_SECONDS)
                return payload, status_code
        except requests.Timeout:
            last_message = "Hết thời gian chờ request."
            last_status_code = None
        except requests.RequestException as error:
            last_message = f"Lỗi kết nối: {error.__class__.__name__}."
            last_status_code = None

        if attempt < MAX_RETRIES:
            delay = retry_delay(response, attempt)
            print(f"{last_message} Thử lại sau {delay:.0f} giây.")
            time.sleep(delay)
        else:
            raise RequestFailed(last_message, last_status_code)


def discover_params(year, page):
    return {
        "primary_release_date.gte": f"{year}-01-01",
        "primary_release_date.lte": f"{year}-12-31",
        "include_adult": "false",
        "include_video": "false",
        "sort_by": "popularity.desc",
        "page": page,
    }


def release_date_is_in_year(release_date, year):
    if not isinstance(release_date, str) or len(release_date) != 10:
        return False

    return f"{year}-01-01" <= release_date <= f"{year}-12-31"


def record_error(state, logger, key, year, message, status_code=None):
    state["unrecoverable_errors"][key] = {
        "year": year,
        "status_code": status_code,
        "message": message,
        "last_seen_at": utc_now(),
    }
    logger.error("%s | year=%s | status=%s | %s", key, year, status_code, message)


def clear_error(state, key):
    state["unrecoverable_errors"].pop(key, None)


def collect_year(session, state, year, logger):
    checked_ids = {int(movie_id) for movie_id in state["checked_tmdb_ids"]}
    collected_ids = {int(movie["id"]) for movie in state["movie_details"] if movie.get("id") is not None}
    new_details = 0
    year_has_error = False
    total_pages = MAX_PAGES_PER_YEAR

    print(f"Thu thập năm {year}...")
    for page in range(1, MAX_PAGES_PER_YEAR + 1):
        discover_key = f"discover:{year}:{page}"
        try:
            discover_response, _ = get_json(
                session,
                f"{API_BASE_URL}/discover/movie",
                params=discover_params(year, page),
            )
            clear_error(state, discover_key)
        except RequestFailed as error:
            record_error(state, logger, discover_key, year, str(error), error.status_code)
            year_has_error = True
            print(f"Không lấy được trang {page} của năm {year}: {error}")
            break

        total_pages = min(int(discover_response.get("total_pages", 0)), MAX_PAGES_PER_YEAR)
        summaries = discover_response.get("results", [])
        print(f"  Trang {page}/{max(total_pages, 1)}: {len(summaries)} phim.")

        for summary in summaries:
            movie_id = summary.get("id")
            if movie_id is None:
                continue

            movie_id = int(movie_id)
            if movie_id in checked_ids:
                continue

            detail_key = f"detail:{movie_id}"
            try:
                detail, _ = get_json(session, f"{API_BASE_URL}/movie/{movie_id}")
                clear_error(state, detail_key)
            except RequestFailed as error:
                record_error(state, logger, detail_key, year, str(error), error.status_code)
                year_has_error = True
                print(f"  Không lấy được chi tiết movie_id={movie_id}: {error}")
                continue

            checked_ids.add(movie_id)
            state["checked_tmdb_ids"].append(movie_id)

            if detail.get("adult", False):
                continue

            if not release_date_is_in_year(detail.get("release_date"), year):
                continue

            detail_id = detail.get("id")
            if detail_id is None or int(detail_id) in collected_ids:
                continue

            state["movie_details"].append(detail)
            collected_ids.add(int(detail_id))
            new_details += 1

            if new_details % CHECKPOINT_EVERY_DETAILS == 0:
                save_state(state)

        if page >= total_pages:
            break

    if not year_has_error and year not in state["completed_years"]:
        state["completed_years"].append(year)
        state["completed_years"].sort()

    save_state(state)
    print(f"Năm {year}: thêm {new_details} phim hợp lệ.")
    return not year_has_error


def join_names(items):
    if items is None:
        return None

    return "|".join(item["name"] for item in items if item.get("name"))


def movie_to_row(movie):
    return {
        "tmdb_id": movie.get("id"),
        "imdb_id": movie.get("imdb_id"),
        "title": movie.get("title"),
        "original_title": movie.get("original_title"),
        "release_date": movie.get("release_date"),
        "original_language": movie.get("original_language"),
        "budget": movie.get("budget"),
        "revenue": movie.get("revenue"),
        "runtime": movie.get("runtime"),
        "vote_average": movie.get("vote_average"),
        "vote_count": movie.get("vote_count"),
        "popularity": movie.get("popularity"),
        "genres": join_names(movie.get("genres")),
        "production_countries": join_names(movie.get("production_countries")),
        "production_companies": join_names(movie.get("production_companies")),
        "adult": movie.get("adult"),
        "status": movie.get("status"),
    }


def build_dataframe(movie_details):
    return pd.DataFrame([movie_to_row(movie) for movie in movie_details], columns=COLUMNS)


def validate_test_year(state, year):
    movie_details = state["movie_details"]
    movie_ids = [movie.get("id") for movie in movie_details]
    invalid_dates = [
        movie.get("release_date")
        for movie in movie_details
        if not release_date_is_in_year(movie.get("release_date"), year)
    ]
    missing_fields = [
        movie.get("id")
        for movie in movie_details
        if not REQUIRED_DETAIL_FIELDS.issubset(movie)
    ]

    if year not in state["completed_years"]:
        raise RuntimeError(f"Năm {year} chưa hoàn thành do còn request lỗi.")
    if not movie_details:
        raise RuntimeError(f"Không thu thập được phim hợp lệ nào cho năm {year}.")
    if invalid_dates:
        raise RuntimeError(f"Có {len(invalid_dates)} phim ngoài phạm vi năm {year}.")
    if len(movie_ids) != len(set(movie_ids)):
        raise RuntimeError("Có tmdb_id trùng trong dữ liệu thử.")
    if missing_fields:
        raise RuntimeError("Có phản hồi chi tiết thiếu trường bắt buộc.")

    print(f"Kiểm tra năm {year} thành công: {len(movie_details)} phim, không trùng và đúng cấu trúc.")


def write_csv(dataframe):
    dataframe.to_csv(CSV_OUTPUT_PATH, index=False, encoding="utf-8")


def validate_full_dataset(state):
    dataframe = build_dataframe(state["movie_details"])
    release_dates = pd.to_datetime(dataframe["release_date"], errors="coerce")
    outside_range = (
        (release_dates < pd.Timestamp(f"{FIRST_YEAR}-01-01"))
        | (release_dates > pd.Timestamp(f"{LAST_YEAR}-12-31"))
        | release_dates.isna()
    ).sum()
    duplicate_ids = dataframe.duplicated(subset="tmdb_id").sum()
    adult_movies = dataframe["adult"].eq(True).sum()
    missing_fields = [
        movie.get("id")
        for movie in state["movie_details"]
        if not REQUIRED_DETAIL_FIELDS.issubset(movie)
    ]

    if outside_range or duplicate_ids or adult_movies or missing_fields:
        raise RuntimeError(
            "Kiểm tra toàn bộ dữ liệu thất bại: "
            f"ngoài phạm vi={outside_range}, trùng={duplicate_ids}, "
            f"adult={adult_movies}, thiếu trường={len(missing_fields)}."
        )

    return dataframe


def print_report(state, dataframe, elapsed_seconds):
    release_dates = pd.to_datetime(dataframe["release_date"], errors="coerce")
    year_counts = release_dates.dt.year.value_counts().sort_index()
    outside_range = (
        (release_dates < pd.Timestamp(f"{FIRST_YEAR}-01-01"))
        | (release_dates > pd.Timestamp(f"{LAST_YEAR}-12-31"))
        | release_dates.isna()
    ).sum()
    missing_imdb = (dataframe["imdb_id"].isna() | dataframe["imdb_id"].eq("")).sum()

    print("\nBÁO CÁO THU THẬP")
    print(f"Tổng số phim: {len(dataframe)}")
    print("Số phim theo từng năm:")
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        print(f"  {year}: {year_counts.get(year, 0)}")
    print(f"Ngày phát hành nhỏ nhất: {dataframe['release_date'].min()}")
    print(f"Ngày phát hành lớn nhất: {dataframe['release_date'].max()}")
    print(f"Số tmdb_id trùng: {dataframe.duplicated(subset='tmdb_id').sum()}")
    print(f"Số phim ngoài phạm vi {FIRST_YEAR}-{LAST_YEAR}: {outside_range}")
    print(f"Số phim thiếu imdb_id: {missing_imdb}")
    print(f"Số phim có budget bằng 0: {dataframe['budget'].eq(0).sum()}")
    print(f"Số phim có revenue bằng 0: {dataframe['revenue'].eq(0).sum()}")
    print(f"Số request lỗi chưa khắc phục: {len(state['unrecoverable_errors'])}")
    print(f"JSON: {RAW_OUTPUT_PATH}")
    print(f"CSV: {CSV_OUTPUT_PATH}")
    print(f"Thời gian thu thập tích lũy: {elapsed_seconds:.1f} giây")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Thu thập phim TMDb theo từng năm.")
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        help="Các năm cần chạy. Không truyền để chạy 2000-2025.",
    )
    parser.add_argument(
        "--skip-csv",
        action="store_true",
        help="Chỉ dùng cho lần kiểm tra nhỏ, chưa tạo CSV toàn bộ.",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    years = sorted(set(arguments.years or range(FIRST_YEAR, LAST_YEAR + 1)))

    if any(year < FIRST_YEAR or year > LAST_YEAR for year in years):
        raise ValueError(f"Năm phải nằm trong khoảng {FIRST_YEAR}-{LAST_YEAR}.")

    load_dotenv(ENV_PATH)
    api_token = os.getenv("TMDB_API_TOKEN", "").strip()
    if not api_token:
        raise RuntimeError("TMDB_API_TOKEN chưa có giá trị trong .env.")

    logger = setup_logger()
    state = load_state()
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {api_token}",
            "accept": "application/json",
        }
    )

    started = time.perf_counter()
    for year in years:
        if year in state["completed_years"]:
            print(f"Năm {year} đã có checkpoint hoàn chỉnh, bỏ qua.")
            continue
        collect_year(session, state, year, logger)

    run_seconds = time.perf_counter() - started
    state["metadata"]["elapsed_seconds"] = state["metadata"].get("elapsed_seconds", 0.0) + run_seconds
    save_state(state)

    if arguments.years and len(years) == 1:
        validate_test_year(state, years[0])

    all_years_complete = all(year in state["completed_years"] for year in range(FIRST_YEAR, LAST_YEAR + 1))
    if arguments.skip_csv:
        return

    if not all_years_complete or state["unrecoverable_errors"]:
        print("Chưa tạo CSV toàn bộ vì vẫn còn năm hoặc request chưa hoàn thành.")
        return

    dataframe = validate_full_dataset(state)
    write_csv(dataframe)
    print_report(state, dataframe, state["metadata"]["elapsed_seconds"])


if __name__ == "__main__":
    main()
