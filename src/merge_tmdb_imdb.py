"""Ghép IMDb Ratings vào dữ liệu TMDb bằng khóa imdb_id/tconst."""

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMDB_INPUT_PATH = PROJECT_ROOT / "data" / "interim" / "tmdb_movies_2000_2025.csv"
IMDB_RATINGS_PATH = PROJECT_ROOT / "data" / "external" / "title.ratings.tsv.gz"
OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "tmdb_imdb_merged.csv"
EXPECTED_TMDB_ROWS = 2597


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def read_tmdb_movies():
    if not TMDB_INPUT_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu TMDb: {TMDB_INPUT_PATH}")

    movies = pd.read_csv(TMDB_INPUT_PATH, dtype={"imdb_id": "string"})
    if len(movies) != EXPECTED_TMDB_ROWS:
        raise ValueError(
            f"Dữ liệu TMDb phải có {EXPECTED_TMDB_ROWS} dòng, nhưng hiện có {len(movies)}."
        )

    return movies


def read_imdb_ratings():
    if not IMDB_RATINGS_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file IMDb Ratings: {IMDB_RATINGS_PATH}. "
            "Hãy chạy src/download_imdb_ratings.py trước."
        )

    ratings = pd.read_csv(
        IMDB_RATINGS_PATH,
        sep="\t",
        compression="gzip",
        usecols=["tconst", "averageRating", "numVotes"],
        dtype={
            "tconst": "string",
            "averageRating": "Float64",
            "numVotes": "Int64",
        },
        na_values="\\N",
    )
    ratings = ratings.rename(
        columns={
            "averageRating": "imdb_rating",
            "numVotes": "imdb_vote_count",
        }
    )
    return ratings


def merge_movies_and_ratings(movies, ratings):
    duplicate_tconst_count = int(ratings["tconst"].duplicated().sum())
    if duplicate_tconst_count:
        raise ValueError(
            "IMDb Ratings có tconst trùng nên không thể ghép an toàn với validate='m:1'."
        )

    merged = movies.merge(
        ratings,
        how="left",
        left_on="imdb_id",
        right_on="tconst",
        validate="m:1",
    )

    if len(merged) != len(movies) or len(merged) != EXPECTED_TMDB_ROWS:
        raise ValueError(
            "Số dòng sau khi ghép không còn đúng 2.597; dừng để tránh dữ liệu bị nhân bản."
        )

    duplicate_tmdb_id_count = int(merged["tmdb_id"].duplicated().sum())
    if duplicate_tmdb_id_count:
        raise ValueError("Xuất hiện tmdb_id trùng sau khi ghép; dừng để kiểm tra.")

    return merged, duplicate_tconst_count, duplicate_tmdb_id_count


def report_and_save(movies, merged, duplicate_tconst_count, duplicate_tmdb_id_count):
    total_movies = len(movies)
    missing_imdb_id = int(movies["imdb_id"].isna().sum())
    has_imdb_id = total_movies - missing_imdb_id
    matched_mask = merged["tconst"].notna()
    matched_count = int(matched_mask.sum())
    missing_rating_count = int(((merged["imdb_id"].notna()) & (~matched_mask)).sum())

    result = merged.drop(columns=["tconst"])
    temporary_output_path = OUTPUT_PATH.with_suffix(".tmp")
    result.to_csv(temporary_output_path, index=False)
    temporary_output_path.replace(OUTPUT_PATH)

    print(f"Tổng số phim TMDb: {total_movies}")
    print(f"Số phim thiếu imdb_id: {missing_imdb_id}")
    print(f"Số phim có imdb_id: {has_imdb_id}")
    print(f"Số phim ghép được IMDb Ratings: {matched_count}")
    print(f"Có imdb_id nhưng không tìm thấy rating: {missing_rating_count}")
    print(f"Tỷ lệ ghép trên toàn bộ dữ liệu: {matched_count / total_movies:.2%}")
    print(f"Tỷ lệ ghép trong nhóm có imdb_id: {matched_count / has_imdb_id:.2%}")
    print(f"Số dòng trước/sau ghép: {len(movies)}/{len(result)}")
    print(f"Số tmdb_id trùng sau ghép: {duplicate_tmdb_id_count}")
    print(f"Số tconst trùng trong IMDb Ratings: {duplicate_tconst_count}")
    print(f"Kiểu imdb_rating: {result['imdb_rating'].dtype}")
    print(f"Kiểu imdb_vote_count: {result['imdb_vote_count'].dtype}")
    print("Ba dòng mẫu có kết quả IMDb:")
    sample_columns = ["tmdb_id", "title", "imdb_id", "imdb_rating", "imdb_vote_count"]
    print(result.loc[result["imdb_rating"].notna(), sample_columns].head(3).to_string(index=False))
    print(f"Đã lưu: {OUTPUT_PATH}")


def main():
    movies = read_tmdb_movies()
    ratings = read_imdb_ratings()
    merged, duplicate_tconst_count, duplicate_tmdb_id_count = merge_movies_and_ratings(
        movies, ratings
    )
    report_and_save(movies, merged, duplicate_tconst_count, duplicate_tmdb_id_count)


if __name__ == "__main__":
    main()
