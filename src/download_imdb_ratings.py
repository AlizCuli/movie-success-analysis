"""Tải file IMDb Ratings chính thức mà không nạp toàn bộ file vào RAM."""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


SOURCE_URL = "https://datasets.imdbws.com/title.ratings.tsv.gz"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "external" / "title.ratings.tsv.gz"
PART_PATH = OUTPUT_PATH.with_suffix(OUTPUT_PATH.suffix + ".part")
METADATA_PATH = PROJECT_ROOT / "data" / "external" / "title.ratings.metadata.json"
TIMEOUT_SECONDS = 60
MAX_RETRIES = 4
CHUNK_SIZE_BYTES = 1024 * 1024


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class DownloadError(RuntimeError):
    """Biểu thị lỗi khi tải file IMDb."""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def retry_delay(response, attempt):
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            return max(float(retry_after), float(2**attempt))

    return float(2**attempt)


def write_metadata(response, file_size_bytes):
    metadata = {
        "source_url": SOURCE_URL,
        "downloaded_at": utc_now(),
        "file_name": OUTPUT_PATH.name,
        "file_size_bytes": file_size_bytes,
        "content_length_bytes": response.headers.get("Content-Length"),
    }
    temporary_metadata_path = METADATA_PATH.with_suffix(".tmp")

    with temporary_metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    temporary_metadata_path.replace(METADATA_PATH)


def download_file(force):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists() and not force:
        print(f"File hoàn chỉnh đã tồn tại: {OUTPUT_PATH}")
        print("Không tải lại. Dùng --force nếu muốn tải mới.")
        return

    session = requests.Session()
    last_error = "Không rõ lỗi tải file."

    for attempt in range(MAX_RETRIES + 1):
        response = None
        try:
            with session.get(SOURCE_URL, stream=True, timeout=TIMEOUT_SECONDS) as response:
                if response.status_code == 429 or 500 <= response.status_code <= 599:
                    raise DownloadError(f"Lỗi HTTP {response.status_code} có thể thử lại.")

                response.raise_for_status()

                file_size_bytes = 0
                with PART_PATH.open("wb") as file:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE_BYTES):
                        if chunk:
                            file.write(chunk)
                            file_size_bytes += len(chunk)

                if file_size_bytes == 0:
                    raise DownloadError("File tải về rỗng.")

                PART_PATH.replace(OUTPUT_PATH)
                write_metadata(response, file_size_bytes)
                print(f"Tải thành công: {OUTPUT_PATH}")
                print(f"Kích thước: {file_size_bytes:,} bytes")
                return
        except (requests.RequestException, OSError, DownloadError) as error:
            last_error = f"{error.__class__.__name__}: {error}"

            if attempt == MAX_RETRIES:
                break

            delay = retry_delay(response, attempt)
            print(f"{last_error} Thử lại sau {delay:.0f} giây.")
            time.sleep(delay)

    raise DownloadError(
        f"Không thể tải IMDb Ratings sau {MAX_RETRIES + 1} lần thử. {last_error}"
    )


def main():
    parser = argparse.ArgumentParser(description="Tải file IMDb title.ratings.tsv.gz.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Tải lại dù file hoàn chỉnh đã tồn tại.",
    )
    args = parser.parse_args()

    download_file(force=args.force)


if __name__ == "__main__":
    main()
