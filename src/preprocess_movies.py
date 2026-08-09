"""Tiền xử lý dữ liệu TMDb cho EDA và pipeline XGBoost."""

import math
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "interim" / "tmdb_movies_2000_2025.csv"
CLEANED_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "movies_cleaned.csv"
MODELING_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "movies_modeling.csv"

NUMERIC_COLUMNS = [
    "budget",
    "revenue",
    "runtime",
    "vote_average",
    "vote_count",
    "popularity",
]
LIST_COLUMNS = ["genres", "production_countries", "production_companies"]
IMPORTANT_COLUMNS = [
    "release_date",
    "budget",
    "revenue",
    "runtime",
    "genres",
    "is_successful",
]


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def normalize_column_names(dataframe):
    """Đưa tên cột về dạng chữ thường, không có khoảng trắng thừa."""
    dataframe = dataframe.copy()
    dataframe.columns = [
        column.strip().lower().replace(" ", "_") for column in dataframe.columns
    ]
    return dataframe


def normalize_pipe_text(value):
    """Loại khoảng trắng thừa quanh từng giá trị, vẫn giữ mọi phần tử có nghĩa."""
    if pd.isna(value):
        return pd.NA

    parts = [part.strip() for part in str(value).split("|") if part.strip()]
    return "|".join(parts) if parts else pd.NA


def split_pipe_text(value):
    if pd.isna(value):
        return []
    return [part for part in str(value).split("|") if part]


def item_count(value):
    if pd.isna(value):
        return pd.NA
    return len(split_pipe_text(value))


def first_item(value):
    items = split_pipe_text(value)
    return items[0] if items else pd.NA


def log1p_series(series):
    """Tính log1p cho giá trị không âm, giữ thiếu cho giá trị không hợp lệ."""
    transformed = pd.Series(pd.NA, index=series.index, dtype="Float64")
    valid_values = series.notna() & (series >= 0)
    transformed.loc[valid_values] = series.loc[valid_values].map(math.log1p)
    return transformed


def mark_iqr_outliers(series):
    """Đánh dấu ngoại lai theo IQR; không sửa hay loại bỏ giá trị gốc."""
    valid_values = series.dropna()
    flags = pd.Series(pd.NA, index=series.index, dtype="Int64")
    flags.loc[series.notna()] = 0

    if valid_values.empty:
        return flags, None, None

    first_quartile = valid_values.quantile(0.25)
    third_quartile = valid_values.quantile(0.75)
    iqr = third_quartile - first_quartile
    lower_bound = first_quartile - 1.5 * iqr
    upper_bound = third_quartile + 1.5 * iqr

    outlier_mask = (series < lower_bound) | (series > upper_bound)
    flags.loc[outlier_mask] = 1
    return flags, lower_bound, upper_bound


def read_and_clean_input():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file đầu vào: {INPUT_PATH}")

    movies = pd.read_csv(INPUT_PATH, dtype={"imdb_id": "string"})
    movies = normalize_column_names(movies)
    movies = movies.replace(r"^\s*$|^\\N$", pd.NA, regex=True)

    if movies.empty:
        raise ValueError("Dữ liệu TMDb đầu vào đang trống.")
    if movies["tmdb_id"].duplicated().any():
        raise ValueError("Đầu vào có tmdb_id trùng; dừng để kiểm tra dữ liệu nguồn.")

    movies["tmdb_id"] = pd.to_numeric(movies["tmdb_id"], errors="coerce")
    if movies["tmdb_id"].isna().any():
        raise ValueError("Có tmdb_id không hợp lệ trong dữ liệu đầu vào.")
    movies["tmdb_id"] = movies["tmdb_id"].astype("Int64")

    for column in NUMERIC_COLUMNS:
        movies[column] = pd.to_numeric(movies[column], errors="coerce")

    movies["release_date"] = pd.to_datetime(
        movies["release_date"], format="%Y-%m-%d", errors="coerce"
    )

    for column in LIST_COLUMNS:
        movies[column] = movies[column].map(normalize_pipe_text).astype("string")

    for column in ["budget", "revenue", "runtime"]:
        movies.loc[movies[column] <= 0, column] = pd.NA

    invalid_rating = movies["vote_average"].notna() & ~movies["vote_average"].between(0, 10)
    movies.loc[invalid_rating, "vote_average"] = pd.NA

    return movies


def create_features(movies):
    movies = movies.copy()

    movies["budget_available"] = movies["budget"].notna().astype("Int64")
    movies["revenue_available"] = movies["revenue"].notna().astype("Int64")

    movies["release_year"] = movies["release_date"].dt.year.astype("Int64")
    movies["release_month"] = movies["release_date"].dt.month.astype("Int64")
    movies["release_quarter"] = movies["release_date"].dt.quarter.astype("Int64")

    movies["profit"] = movies["revenue"] - movies["budget"]
    movies["revenue_to_budget"] = movies["revenue"] / movies["budget"]
    movies["roi"] = (movies["revenue"] - movies["budget"]) / movies["budget"]
    movies["log_budget"] = log1p_series(movies["budget"])
    movies["log_revenue"] = log1p_series(movies["revenue"])

    movies["genre_count"] = movies["genres"].map(item_count).astype("Int64")
    movies["primary_genre"] = movies["genres"].map(first_item).astype("string")
    movies["production_country_count"] = (
        movies["production_countries"].map(item_count).astype("Int64")
    )
    movies["primary_country"] = (
        movies["production_countries"].map(first_item).astype("string")
    )
    movies["production_company_count"] = (
        movies["production_companies"].map(item_count).astype("Int64")
    )

    movies["budget_outlier"], _, _ = mark_iqr_outliers(movies["budget"])
    movies["revenue_outlier"], _, _ = mark_iqr_outliers(movies["revenue"])

    movies["is_successful"] = pd.Series(pd.NA, index=movies.index, dtype="Int64")
    financial_data_available = movies["budget"].notna() & movies["revenue"].notna()
    movies.loc[financial_data_available, "is_successful"] = (
        movies.loc[financial_data_available, "revenue_to_budget"] >= 2
    ).astype("Int64")

    return movies


def build_modeling_dataset(cleaned_movies):
    modeling_mask = (
        cleaned_movies["budget"].notna()
        & cleaned_movies["revenue"].notna()
        & cleaned_movies["runtime"].notna()
        & cleaned_movies["release_date"].notna()
        & cleaned_movies["is_successful"].notna()
    )
    return cleaned_movies.loc[modeling_mask].copy(), modeling_mask


def save_csv(dataframe, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp")
    dataframe.to_csv(temporary_path, index=False, date_format="%Y-%m-%d")
    temporary_path.replace(output_path)


def print_quality_report(source_movies, cleaned_movies, modeling_movies, modeling_mask):
    budget_stats = cleaned_movies["budget"].dropna().agg(["min", "median", "mean", "max"])
    revenue_stats = cleaned_movies["revenue"].dropna().agg(["min", "median", "mean", "max"])
    missing_counts = cleaned_movies[IMPORTANT_COLUMNS].isna().sum()
    class_counts = modeling_movies["is_successful"].value_counts().sort_index()

    print(f"Số dòng trước/sau tiền xử lý: {len(source_movies)}/{len(cleaned_movies)}")
    print(f"Số dòng movies_cleaned.csv: {len(cleaned_movies)}")
    print(f"Số dòng movies_modeling.csv: {len(modeling_movies)}")
    print(f"Số phim bị loại khỏi modeling: {int((~modeling_mask).sum())}")
    print("Lý do không đủ điều kiện modeling (có thể chồng lấp):")
    print(f"- budget không hợp lệ: {int(cleaned_movies['budget'].isna().sum())}")
    print(f"- revenue không hợp lệ: {int(cleaned_movies['revenue'].isna().sum())}")
    print(f"- runtime không hợp lệ: {int(cleaned_movies['runtime'].isna().sum())}")
    print(f"- release_date không hợp lệ: {int(cleaned_movies['release_date'].isna().sum())}")
    print(f"- không có is_successful: {int(cleaned_movies['is_successful'].isna().sum())}")
    print("Số giá trị thiếu ở cột quan trọng:")
    for column, count in missing_counts.items():
        print(f"- {column}: {int(count)}")
    print(f"Số ngoại lai budget: {int(cleaned_movies['budget_outlier'].sum())}")
    print(f"Số ngoại lai revenue: {int(cleaned_movies['revenue_outlier'].sum())}")
    print(
        "Budget hợp lệ (min, median, mean, max): "
        + ", ".join(f"{name}={value:.2f}" for name, value in budget_stats.items())
    )
    print(
        "Revenue hợp lệ (min, median, mean, max): "
        + ", ".join(f"{name}={value:.2f}" for name, value in revenue_stats.items())
    )
    print("Phân bố is_successful trong tập modeling:")
    for class_value in [0, 1]:
        class_count = int(class_counts.get(class_value, 0))
        class_ratio = class_count / len(modeling_movies) if len(modeling_movies) else 0
        print(f"- Lớp {class_value}: {class_count} ({class_ratio:.2%})")
    print(
        "tmdb_id trùng (cleaned/modeling): "
        f"{int(cleaned_movies['tmdb_id'].duplicated().sum())}/"
        f"{int(modeling_movies['tmdb_id'].duplicated().sum())}"
    )
    print(
        "budget/revenue âm (cleaned): "
        f"{int((cleaned_movies['budget'] < 0).sum())}/"
        f"{int((cleaned_movies['revenue'] < 0).sum())}"
    )
    release_years = cleaned_movies["release_year"].dropna()
    outside_year_range = int((~release_years.between(2000, 2025)).sum())
    print(f"release_year ngoài 2000-2025: {outside_year_range}")


def quality_problems(cleaned_movies, modeling_movies):
    problems = []
    if len(modeling_movies) < 500:
        problems.append("Tập modeling có ít hơn 500 phim.")

    class_ratios = modeling_movies["is_successful"].value_counts(normalize=True)
    for class_value in [0, 1]:
        if class_ratios.get(class_value, 0) < 0.20:
            problems.append(f"Lớp is_successful={class_value} chiếm dưới 20%.")

    if cleaned_movies["tmdb_id"].duplicated().any() or modeling_movies["tmdb_id"].duplicated().any():
        problems.append("Phát hiện tmdb_id trùng trong đầu ra.")

    release_years = cleaned_movies["release_year"].dropna()
    if (~release_years.between(2000, 2025)).any():
        problems.append("Có release_year ngoài phạm vi 2000-2025.")

    return problems


def main():
    source_movies = read_and_clean_input()
    cleaned_movies = create_features(source_movies)
    modeling_movies, modeling_mask = build_modeling_dataset(cleaned_movies)

    save_csv(cleaned_movies, CLEANED_OUTPUT_PATH)
    save_csv(modeling_movies, MODELING_OUTPUT_PATH)
    print_quality_report(source_movies, cleaned_movies, modeling_movies, modeling_mask)
    print(f"Đã lưu: {CLEANED_OUTPUT_PATH}")
    print(f"Đã lưu: {MODELING_OUTPUT_PATH}")

    problems = quality_problems(cleaned_movies, modeling_movies)
    if problems:
        raise RuntimeError("Kiểm tra chất lượng không đạt:\n- " + "\n- ".join(problems))


if __name__ == "__main__":
    main()
