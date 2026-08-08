"""Tạo bảng EDA và biểu đồ từ dữ liệu phim đã tiền xử lý.

Script chỉ đọc các CSV trong data/processed và lưu kết quả báo cáo vào reports/.
Không thực hiện chia train/test hoặc huấn luyện mô hình.
"""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEANED_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "movies_cleaned.csv"
MODELING_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "movies_modeling.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"

DESCRIPTIVE_COLUMNS = [
    "budget",
    "revenue",
    "runtime",
    "popularity",
    "vote_average",
    "vote_count",
    "imdb_rating",
    "imdb_vote_count",
    "revenue_to_budget",
]
CORRELATION_COLUMNS = [
    "budget",
    "revenue",
    "runtime",
    "popularity",
    "vote_average",
    "vote_count",
    "imdb_rating",
    "imdb_vote_count",
]

PRIMARY_COLOR = "#2A6F97"
SECONDARY_COLOR = "#E76F51"
SUCCESS_COLOR = "#2A9D8F"
FAILURE_COLOR = "#E76F51"


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def prepare_output_directories():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_datasets():
    """Đọc hai tập dữ liệu dẫn xuất, không thay đổi file nguồn."""
    for input_path in [CLEANED_INPUT_PATH, MODELING_INPUT_PATH]:
        if not input_path.exists():
            raise FileNotFoundError(f"Không tìm thấy dữ liệu đầu vào: {input_path}")

    cleaned = pd.read_csv(CLEANED_INPUT_PATH, parse_dates=["release_date"])
    modeling = pd.read_csv(MODELING_INPUT_PATH, parse_dates=["release_date"])
    return cleaned, modeling


def save_table(dataframe, file_name, index=False):
    output_path = TABLES_DIR / file_name
    dataframe.to_csv(output_path, index=index)
    return output_path


def save_figure(figure, file_name):
    output_path = FIGURES_DIR / file_name
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path


def descriptive_statistics(cleaned, modeling):
    """Thống kê các biến, dùng modeling cho revenue_to_budget hợp lệ."""
    rows = []
    for column in DESCRIPTIVE_COLUMNS:
        source = modeling[column] if column == "revenue_to_budget" else cleaned[column]
        values = source.dropna()
        rows.append(
            {
                "variable": column,
                "dataset": "movies_modeling" if column == "revenue_to_budget" else "movies_cleaned",
                "count": int(values.count()),
                "mean": values.mean(),
                "median": values.median(),
                "std": values.std(),
                "min": values.min(),
                "q1": values.quantile(0.25),
                "q3": values.quantile(0.75),
                "max": values.max(),
                "skewness": values.skew(),
            }
        )
    return pd.DataFrame(rows)


def missing_values_table(cleaned, modeling):
    rows = []
    for dataset_name, dataframe in [
        ("movies_cleaned", cleaned),
        ("movies_modeling", modeling),
    ]:
        for column, missing_count in dataframe.isna().sum().items():
            rows.append(
                {
                    "dataset": dataset_name,
                    "column": column,
                    "missing_count": int(missing_count),
                    "missing_percentage": missing_count / len(dataframe) * 100,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["dataset", "missing_count", "column"], ascending=[True, False, True]
    )


def genre_summary(modeling):
    """Phân tích đa thể loại bằng explode, chỉ giữ thể loại có ít nhất 30 phim."""
    genre_data = modeling[["tmdb_id", "genres", "revenue", "is_successful"]].copy()
    genre_data = genre_data.dropna(subset=["genres"])
    genre_data["genre"] = genre_data["genres"].str.split("|")
    genre_data = genre_data.explode("genre")
    genre_data["genre"] = genre_data["genre"].str.strip()
    genre_data = genre_data[genre_data["genre"].ne("")]

    summary = (
        genre_data.groupby("genre", as_index=False)
        .agg(
            movie_count=("tmdb_id", "nunique"),
            median_revenue=("revenue", "median"),
            success_rate=("is_successful", "mean"),
        )
        .query("movie_count >= 30")
        .sort_values("movie_count", ascending=False)
    )
    return summary


def yearly_summary(modeling):
    return (
        modeling.groupby("release_year", as_index=False)
        .agg(
            movie_count=("tmdb_id", "nunique"),
            median_budget=("budget", "median"),
            median_revenue=("revenue", "median"),
            success_rate=("is_successful", "mean"),
        )
        .sort_values("release_year")
    )


def monthly_success_summary(modeling):
    return (
        modeling.groupby("release_month", as_index=False)
        .agg(movie_count=("tmdb_id", "nunique"), success_rate=("is_successful", "mean"))
        .sort_values("release_month")
    )


def chi_square_summary(modeling):
    """Kiểm định Chi-square cho primary_genre và is_successful."""
    data = modeling[["primary_genre", "is_successful"]].dropna().copy()
    genre_counts = data["primary_genre"].value_counts()
    data["genre_group"] = data["primary_genre"].where(
        data["primary_genre"].map(genre_counts) >= 30, "Other"
    )
    contingency_table = pd.crosstab(data["genre_group"], data["is_successful"])

    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        raise ValueError("Bảng chéo Chi-square không đủ ít nhất 2 hàng và 2 cột.")

    chi_square, p_value, degrees_of_freedom, expected = chi2_contingency(
        contingency_table, correction=False
    )
    expected_values = [float(value) for row in expected for value in row]
    expected_below_five = sum(value < 5 for value in expected_values)
    expected_below_five_percentage = expected_below_five / len(expected_values) * 100
    assumptions_met = min(expected_values) >= 1 and expected_below_five_percentage <= 20

    total_observations = int(contingency_table.to_numpy().sum())
    smaller_dimension = min(contingency_table.shape) - 1
    cramers_v = math.sqrt((chi_square / total_observations) / smaller_dimension)

    result = pd.DataFrame(
        [
            {
                "chi_square_statistic": chi_square,
                "degrees_of_freedom": degrees_of_freedom,
                "p_value": p_value,
                "cramers_v": cramers_v,
                "minimum_expected_frequency": min(expected_values),
                "expected_frequency_below_5_count": expected_below_five,
                "expected_frequency_below_5_percentage": expected_below_five_percentage,
                "assumptions_met": assumptions_met,
                "genre_groups": contingency_table.shape[0],
                "observations": total_observations,
            }
        ]
    )
    return result, contingency_table


def plot_missing_values(missing_values):
    plot_data = (
        missing_values.query("dataset == 'movies_cleaned'")
        .sort_values("missing_percentage", ascending=True)
        .tail(12)
    )
    figure, axis = plt.subplots(figsize=(9, 6))
    sns.barplot(data=plot_data, x="missing_percentage", y="column", color=PRIMARY_COLOR, ax=axis)
    axis.set_title("Tỷ lệ giá trị thiếu trong movies_cleaned")
    axis.set_xlabel("Tỷ lệ thiếu (%)")
    axis.set_ylabel("Cột")
    return save_figure(figure, "01_missing_values.png")


def plot_distributions(modeling):
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    plots = [
        ("budget", "Budget (USD)", PRIMARY_COLOR),
        ("log_budget", "log1p(Budget)", PRIMARY_COLOR),
        ("revenue", "Revenue (USD)", SECONDARY_COLOR),
        ("log_revenue", "log1p(Revenue)", SECONDARY_COLOR),
    ]
    for axis, (column, label, color) in zip(axes.flat, plots):
        sns.histplot(modeling[column].dropna(), bins=35, color=color, ax=axis)
        axis.set_title(label)
        axis.set_xlabel(label)
        axis.set_ylabel("Số phim")
    figure.suptitle("Phân phối budget và revenue trên thang gốc và log", y=1.02)
    figure.tight_layout()
    return save_figure(figure, "02_budget_revenue_distribution.png")


def plot_budget_revenue(modeling):
    figure, axis = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=modeling,
        x="log_budget",
        y="log_revenue",
        hue="is_successful",
        palette={0: FAILURE_COLOR, 1: SUCCESS_COLOR},
        alpha=0.65,
        s=35,
        ax=axis,
    )
    axis.set_title("Mối liên hệ giữa budget và revenue trên thang log")
    axis.set_xlabel("log1p(Budget)")
    axis.set_ylabel("log1p(Revenue)")
    axis.legend(title="is_successful", labels=["0", "1"])
    return save_figure(figure, "03_budget_vs_revenue.png")


def plot_correlation_heatmap(pearson_correlation):
    figure, axis = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        pearson_correlation,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        square=True,
        linewidths=0.4,
        cbar_kws={"label": "Pearson correlation"},
        ax=axis,
    )
    axis.set_title("Ma trận tương quan Pearson")
    figure.tight_layout()
    return save_figure(figure, "04_correlation_heatmap.png")


def plot_genre_analysis(summary):
    display_data = summary.sort_values("movie_count", ascending=True)
    figure, axes = plt.subplots(1, 2, figsize=(14, 7))
    sns.barplot(data=display_data, x="movie_count", y="genre", color=PRIMARY_COLOR, ax=axes[0])
    axes[0].set_title("Số phim theo thể loại (ít nhất 30 phim)")
    axes[0].set_xlabel("Số phim")
    axes[0].set_ylabel("Thể loại")
    sns.barplot(data=display_data, x="success_rate", y="genre", color=SUCCESS_COLOR, ax=axes[1])
    axes[1].set_title("Tỷ lệ thành công theo thể loại")
    axes[1].set_xlabel("Tỷ lệ thành công")
    axes[1].set_ylabel("")
    axes[1].set_xlim(0, 1)
    figure.tight_layout()
    return save_figure(figure, "05_genre_analysis.png")


def plot_yearly_trends(summary, monthly):
    figure, axes = plt.subplots(4, 1, figsize=(12, 14))
    axes[0].plot(summary["release_year"], summary["median_budget"] / 1_000_000, color=PRIMARY_COLOR)
    axes[0].set_title("Median budget theo năm phát hành")
    axes[0].set_ylabel("Triệu USD")
    axes[1].plot(summary["release_year"], summary["median_revenue"] / 1_000_000, color=SECONDARY_COLOR)
    axes[1].set_title("Median revenue theo năm phát hành")
    axes[1].set_ylabel("Triệu USD")
    axes[2].plot(summary["release_year"], summary["success_rate"], color=SUCCESS_COLOR)
    axes[2].set_title("Tỷ lệ thành công theo năm phát hành")
    axes[2].set_ylabel("Tỷ lệ thành công")
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("Năm phát hành")
    axes[3].bar(monthly["release_month"], monthly["success_rate"], color=SUCCESS_COLOR)
    axes[3].set_title("Tỷ lệ thành công theo tháng phát hành")
    axes[3].set_xlabel("Tháng phát hành")
    axes[3].set_ylabel("Tỷ lệ thành công")
    axes[3].set_xticks(range(1, 13))
    axes[3].set_ylim(0, 1)
    figure.tight_layout()
    return save_figure(figure, "06_yearly_trends.png")


def plot_success_class_balance(modeling):
    class_counts = modeling["is_successful"].value_counts().sort_index().rename_axis("is_successful")
    plot_data = class_counts.reset_index(name="movie_count")
    figure, axis = plt.subplots(figsize=(7, 5))
    sns.barplot(
        data=plot_data,
        x="is_successful",
        y="movie_count",
        hue="is_successful",
        palette={0: FAILURE_COLOR, 1: SUCCESS_COLOR},
        legend=False,
        ax=axis,
    )
    for index, row in plot_data.iterrows():
        percentage = row["movie_count"] / len(modeling) * 100
        axis.text(index, row["movie_count"] + 25, f"{int(row['movie_count'])}\n({percentage:.1f}%)", ha="center")
    axis.set_title("Phân bố nhãn is_successful")
    axis.set_xlabel("is_successful")
    axis.set_ylabel("Số phim")
    return save_figure(figure, "07_success_class_balance.png")


def plot_rating_revenue_relationship(modeling):
    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    plots = [
        ("popularity", "Popularity", PRIMARY_COLOR),
        ("imdb_rating", "IMDb rating", SUCCESS_COLOR),
        ("log_vote_count", "log1p(TMDb vote count)", SECONDARY_COLOR),
        ("log_imdb_vote_count", "log1p(IMDb vote count)", "#6A4C93"),
    ]
    for axis, (column, label, color) in zip(axes.flat, plots):
        sns.regplot(
            data=modeling,
            x=column,
            y="log_revenue",
            scatter_kws={"alpha": 0.35, "s": 18, "color": color},
            line_kws={"color": "#1D3557"},
            ax=axis,
        )
        axis.set_title(f"{label} và log1p(Revenue)")
        axis.set_xlabel(label)
        axis.set_ylabel("log1p(Revenue)")
    figure.suptitle(
        "Tín hiệu hậu phát hành chỉ dùng mô tả, không dùng làm predictor",
        y=1.02,
    )
    figure.tight_layout()
    return save_figure(figure, "08_rating_revenue_relationship.png")


def build_figures(cleaned, modeling, missing_values, pearson, genre, yearly, monthly):
    return [
        plot_missing_values(missing_values),
        plot_distributions(modeling),
        plot_budget_revenue(modeling),
        plot_correlation_heatmap(pearson),
        plot_genre_analysis(genre),
        plot_yearly_trends(yearly, monthly),
        plot_success_class_balance(modeling),
        plot_rating_revenue_relationship(modeling),
    ]


def validate_data(cleaned, modeling):
    checks = {
        "cleaned_tmdb_id_duplicates": int(cleaned["tmdb_id"].duplicated().sum()),
        "modeling_tmdb_id_duplicates": int(modeling["tmdb_id"].duplicated().sum()),
        "cleaned_invalid_release_year": int(
            (~cleaned["release_year"].dropna().between(2000, 2025)).sum()
        ),
        "modeling_invalid_release_year": int(
            (~modeling["release_year"].dropna().between(2000, 2025)).sum()
        ),
    }
    if any(checks.values()):
        raise ValueError(f"Kiểm tra dữ liệu EDA không đạt: {checks}")
    return checks


def run_eda():
    """Chạy toàn bộ EDA và trả lại kết quả để notebook có thể sử dụng."""
    prepare_output_directories()
    sns.set_theme(style="whitegrid", context="notebook")
    cleaned, modeling = load_datasets()
    validation = validate_data(cleaned, modeling)

    missing_values = missing_values_table(cleaned, modeling)
    descriptive = descriptive_statistics(cleaned, modeling)
    pearson = modeling[CORRELATION_COLUMNS].corr(method="pearson")
    spearman = modeling[CORRELATION_COLUMNS].corr(method="spearman")
    genre = genre_summary(modeling)
    yearly = yearly_summary(modeling)
    monthly = monthly_success_summary(modeling)
    chi_square, contingency = chi_square_summary(modeling)

    table_paths = [
        save_table(descriptive, "descriptive_statistics.csv"),
        save_table(missing_values, "missing_values.csv"),
        save_table(pearson, "pearson_correlation.csv", index=True),
        save_table(spearman, "spearman_correlation.csv", index=True),
        save_table(genre, "genre_summary.csv"),
        save_table(yearly, "yearly_summary.csv"),
        save_table(monthly, "monthly_success_summary.csv"),
        save_table(chi_square, "chi_square_result.csv"),
    ]
    figure_paths = build_figures(
        cleaned, modeling, missing_values, pearson, genre, yearly, monthly
    )

    print(f"movies_cleaned: {cleaned.shape[0]} dòng, {cleaned.shape[1]} cột")
    print(f"movies_modeling: {modeling.shape[0]} dòng, {modeling.shape[1]} cột")
    print(f"IMDb rating thiếu trong movies_cleaned: {int(cleaned['imdb_rating'].isna().sum())}")
    print(f"Chi-square p-value: {chi_square.loc[0, 'p_value']:.6g}")
    print(f"Cramér's V: {chi_square.loc[0, 'cramers_v']:.4f}")
    print(f"Đã tạo {len(table_paths)} bảng và {len(figure_paths)} biểu đồ.")

    return {
        "cleaned": cleaned,
        "modeling": modeling,
        "missing_values": missing_values,
        "descriptive": descriptive,
        "pearson": pearson,
        "spearman": spearman,
        "genre": genre,
        "yearly": yearly,
        "monthly": monthly,
        "chi_square": chi_square,
        "contingency": contingency,
        "validation": validation,
    }


if __name__ == "__main__":
    run_eda()
