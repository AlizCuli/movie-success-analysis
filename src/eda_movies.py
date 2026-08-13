"""Tạo các bảng và hình EDA TMDb phục vụ báo cáo cuối.

EDA chỉ mô tả dữ liệu và nhãn. Các predictor hậu phát hành như popularity,
rating và vote không xuất hiện trong bảng/hình chính của báo cáo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pre_release_features import load_pre_release_modeling_data


CLEANED_PATH = ROOT / "data" / "processed" / "movies_cleaned.csv"
MODELING_PATH = ROOT / "data" / "processed" / "movies_modeling.csv"
FIGURES_DIR = ROOT / "reports" / "figures"
TABLES_DIR = ROOT / "reports" / "tables"

CORE_MISSING_COLUMNS = [
    "budget",
    "revenue",
    "runtime",
    "release_date",
    "genres",
    "production_countries",
    "production_companies",
]

CORRELATION_COLUMNS = [
    "is_successful",
    "log_budget",
    "runtime",
    "release_year",
    "genre_count",
    "production_country_count",
    "production_company_count",
    "is_collection",
    "cast_count",
    "crew_count",
    "theatrical_country_count",
    "release_event_count",
]

PRIMARY_COLOR = "#2A6F97"
SECONDARY_COLOR = "#2A9D8F"
FAILURE_COLOR = "#8DA0CB"


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    for path in (CLEANED_PATH, MODELING_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy {path}. Hãy chạy src/preprocess_movies.py trước."
            )

    cleaned = pd.read_csv(CLEANED_PATH, parse_dates=["release_date"])
    modeling = pd.read_csv(MODELING_PATH, parse_dates=["release_date"])
    raw_features, target = load_pre_release_modeling_data()

    if cleaned.empty or modeling.empty:
        raise ValueError("Dữ liệu cleaned/modeling đang trống.")
    if cleaned["tmdb_id"].duplicated().any() or modeling["tmdb_id"].duplicated().any():
        raise ValueError("Phát hiện tmdb_id trùng trong dữ liệu EDA.")
    if not raw_features["tmdb_id"].reset_index(drop=True).equals(
        modeling["tmdb_id"].reset_index(drop=True)
    ):
        raise ValueError("Thứ tự tmdb_id giữa modeling và enrichment không khớp.")

    return cleaned, modeling, raw_features, target


def dataset_summary(cleaned: pd.DataFrame, modeling: pd.DataFrame) -> pd.DataFrame:
    class_counts = modeling["is_successful"].value_counts().sort_index()
    return pd.DataFrame(
        [
            {"metric": "movies_collected", "value": len(cleaned)},
            {"metric": "movies_modeling", "value": len(modeling)},
            {"metric": "class_0_count", "value": int(class_counts.get(0, 0))},
            {"metric": "class_1_count", "value": int(class_counts.get(1, 0))},
            {"metric": "class_0_rate", "value": float(class_counts.get(0, 0) / len(modeling))},
            {"metric": "class_1_rate", "value": float(class_counts.get(1, 0) / len(modeling))},
            {"metric": "release_year_min", "value": int(modeling["release_year"].min())},
            {"metric": "release_year_max", "value": int(modeling["release_year"].max())},
        ]
    )


def missingness_summary(cleaned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in CORE_MISSING_COLUMNS:
        missing = int(cleaned[column].isna().sum())
        rows.append(
            {
                "column": column,
                "missing_count": missing,
                "missing_rate": missing / len(cleaned),
            }
        )
    return pd.DataFrame(rows).sort_values("missing_rate", ascending=False)


def association_tables(
    raw_features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis = raw_features.copy()
    analysis["is_successful"] = target.to_numpy(dtype=int)
    correlation = analysis[CORRELATION_COLUMNS].corr(method="spearman")

    top_genres = analysis["primary_genre"].value_counts().head(8).index
    genre_collection = (
        analysis[analysis["primary_genre"].isin(top_genres)]
        .groupby(["primary_genre", "is_collection"], observed=True)["is_successful"]
        .agg(success_rate="mean", movie_count="size")
        .reset_index()
    )
    return correlation, genre_collection


def yearly_summary(modeling: pd.DataFrame) -> pd.DataFrame:
    return (
        modeling.groupby("release_year", as_index=False)
        .agg(movie_count=("tmdb_id", "nunique"), success_rate=("is_successful", "mean"))
        .sort_values("release_year")
    )


def save_table(frame: pd.DataFrame, name: str, index: bool = False) -> Path:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / name
    frame.to_csv(path, index=index)
    return path


def save_figure(figure: plt.Figure, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def plot_dataset_overview(
    modeling: pd.DataFrame,
    missingness: pd.DataFrame,
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    display_missing = missingness.sort_values("missing_rate")
    axes[0].barh(
        display_missing["column"],
        display_missing["missing_rate"] * 100,
        color=PRIMARY_COLOR,
    )
    axes[0].set_title("A. Missingness in core fields")
    axes[0].set_xlabel("Missing rate (%)")
    axes[0].set_ylabel("")
    for row, value in enumerate(display_missing["missing_rate"] * 100):
        axes[0].text(value + 0.7, row, f"{value:.1f}%", va="center", fontsize=9)

    class_counts = modeling["is_successful"].value_counts().sort_index()
    bars = axes[1].bar(
        ["is_successful = 0", "is_successful = 1"],
        class_counts.reindex([0, 1]).to_numpy(),
        color=[FAILURE_COLOR, SECONDARY_COLOR],
    )
    axes[1].set_title("B. Class balance in modeling cohort")
    axes[1].set_ylabel("Movies")
    for bar, count in zip(bars, class_counts.reindex([0, 1]).to_numpy()):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            count + 20,
            f"{int(count):,}\n({count / len(modeling):.1%})",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[1].set_ylim(0, class_counts.max() * 1.16)

    figure.suptitle("TMDb Data Quality Overview", fontsize=15, fontweight="bold")
    figure.tight_layout()
    return save_figure(figure, "dataset_overview.png")


def plot_feature_associations(
    correlation: pd.DataFrame,
    genre_collection: pd.DataFrame,
    movie_count: int,
) -> Path:
    figure = plt.figure(figsize=(7.2, 4.15), facecolor="white")
    grid = figure.add_gridspec(1, 2, width_ratios=[1.50, 0.82], wspace=0.34)

    axis_correlation = figure.add_subplot(grid[0, 0])
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    image = axis_correlation.imshow(
        correlation.to_numpy(), cmap="RdBu", norm=norm, aspect="equal"
    )
    axis_correlation.set_xticks(
        range(len(correlation.columns)),
        labels=correlation.columns,
        rotation=65,
        ha="right",
        va="top",
    )
    axis_correlation.set_yticks(range(len(correlation.index)), labels=correlation.index)
    axis_correlation.tick_params(axis="x", labelsize=6.0, pad=1)
    axis_correlation.tick_params(axis="y", labelsize=6.2)
    axis_correlation.set_title(
        "A. Spearman associations among variables",
        loc="left",
        pad=7,
        fontsize=8.2,
        fontweight="bold",
    )
    axis_correlation.set_xticks(np.arange(-0.5, len(correlation.columns), 1), minor=True)
    axis_correlation.set_yticks(np.arange(-0.5, len(correlation.index), 1), minor=True)
    axis_correlation.grid(which="minor", color="white", linewidth=1)
    axis_correlation.tick_params(which="minor", bottom=False, left=False)
    for row in range(correlation.shape[0]):
        for column in range(correlation.shape[1]):
            value = correlation.iloc[row, column]
            color = "white" if abs(value) >= 0.55 else "#1F2937"
            axis_correlation.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=5.0,
            )
    colorbar = figure.colorbar(image, ax=axis_correlation, fraction=0.035, pad=0.025)
    colorbar.ax.set_title("ρ", fontsize=6.5, pad=3)
    colorbar.ax.tick_params(labelsize=5.8)

    axis_groups = figure.add_subplot(grid[0, 1])
    overall = (
        genre_collection.groupby("primary_genre", observed=True)
        .apply(
            lambda frame: np.average(frame["success_rate"], weights=frame["movie_count"]),
            include_groups=False,
        )
        .sort_values()
    )
    genre_order = overall.index.tolist()
    positions = np.arange(len(genre_order))
    bar_height = 0.34
    colors = {0: FAILURE_COLOR, 1: SECONDARY_COLOR}
    for flag, offset in ((0, -bar_height / 2), (1, bar_height / 2)):
        subset = genre_collection[genre_collection["is_collection"] == flag].set_index(
            "primary_genre"
        )
        rates = np.array(
            [subset.loc[genre, "success_rate"] if genre in subset.index else np.nan for genre in genre_order],
            dtype=float,
        )
        counts = np.array(
            [subset.loc[genre, "movie_count"] if genre in subset.index else 0 for genre in genre_order],
            dtype=int,
        )
        bars = axis_groups.barh(
            positions + offset,
            rates * 100,
            height=bar_height,
            color=colors[flag],
            label="In a collection" if flag else "Not in a collection",
        )
        for bar, rate, count in zip(bars, rates, counts):
            if np.isnan(rate):
                continue
            axis_groups.text(
                rate * 100 + 0.8,
                bar.get_y() + bar.get_height() / 2,
                f"{rate:.0%} (n={count})",
                va="center",
                fontsize=5.8,
            )
    axis_groups.set_yticks(positions, labels=genre_order)
    axis_groups.set_xlim(0, 112)
    axis_groups.set_ylim(-0.55, len(genre_order) + 0.60)
    axis_groups.set_xlabel("Observed success rate (%)", fontsize=6.8)
    axis_groups.set_title(
        "B. Observed success rate by primary genre\nand collection status",
        loc="left",
        pad=7,
        fontsize=8.2,
        fontweight="bold",
    )
    axis_groups.grid(axis="x", color="#D1D5DB", linewidth=0.8, alpha=0.8)
    axis_groups.set_axisbelow(True)
    axis_groups.spines[["top", "right", "left"]].set_visible(False)
    axis_groups.tick_params(axis="x", labelsize=6.0)
    axis_groups.tick_params(axis="y", length=0, labelsize=6.2)
    axis_groups.legend(
        frameon=False,
        loc="upper left",
        ncol=2,
        fontsize=6.0,
        columnspacing=0.9,
        handlelength=1.4,
    )
    axis_groups.text(
        0.5,
        -0.22,
        "Panel B: eight most common primary genres.",
        transform=axis_groups.transAxes,
        ha="center",
        va="top",
        fontsize=6.2,
        color="#4B5563",
    )

    figure.suptitle(
        "Pre-release variables and financial success",
        x=0.075,
        y=0.965,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
    )
    figure.subplots_adjust(top=0.84, bottom=0.25, left=0.145, right=0.975)
    return save_figure(figure, "pre_release_association_overview.png")


def plot_spearman_heatmap(correlation: pd.DataFrame) -> Path:
    """Tạo heatmap độc lập để đặt theo chiều rộng hai cột của báo cáo."""
    figure, axis = plt.subplots(figsize=(7.0, 4.3), facecolor="white")
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    image = axis.imshow(
        correlation.to_numpy(), cmap="RdBu", norm=norm, aspect="auto"
    )

    axis.set_xticks(
        range(len(correlation.columns)),
        labels=correlation.columns,
        rotation=55,
        ha="right",
        va="top",
    )
    axis.set_yticks(range(len(correlation.index)), labels=correlation.index)
    axis.tick_params(axis="x", labelsize=7.2, pad=2)
    axis.tick_params(axis="y", labelsize=8.0)
    axis.set_title(
        "Spearman associations among descriptive variables",
        loc="left",
        pad=10,
        fontsize=11,
        fontweight="bold",
    )
    axis.set_xticks(np.arange(-0.5, len(correlation.columns), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(correlation.index), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1)
    axis.tick_params(which="minor", bottom=False, left=False)

    for row in range(correlation.shape[0]):
        for column in range(correlation.shape[1]):
            value = correlation.iloc[row, column]
            color = "white" if abs(value) >= 0.55 else "#1F2937"
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=6.5,
            )

    colorbar = figure.colorbar(image, ax=axis, fraction=0.038, pad=0.025)
    colorbar.ax.set_title("ρ", fontsize=8, pad=4)
    colorbar.ax.tick_params(labelsize=7.2)
    figure.subplots_adjust(top=0.89, bottom=0.31, left=0.22, right=0.93)
    return save_figure(figure, "pre_release_spearman_heatmap.png")


def plot_genre_collection_rates(genre_collection: pd.DataFrame) -> Path:
    """Tạo biểu đồ thể loại–franchise độc lập cho một cột báo cáo."""
    figure, axis = plt.subplots(figsize=(4.2, 4.0), facecolor="white")
    overall = (
        genre_collection.groupby("primary_genre", observed=True)
        .apply(
            lambda frame: np.average(frame["success_rate"], weights=frame["movie_count"]),
            include_groups=False,
        )
        .sort_values()
    )
    genre_order = overall.index.tolist()
    positions = np.arange(len(genre_order))
    bar_height = 0.34
    colors = {0: FAILURE_COLOR, 1: SECONDARY_COLOR}

    for flag, offset in ((0, -bar_height / 2), (1, bar_height / 2)):
        subset = genre_collection[genre_collection["is_collection"] == flag].set_index(
            "primary_genre"
        )
        rates = np.array(
            [
                subset.loc[genre, "success_rate"]
                if genre in subset.index
                else np.nan
                for genre in genre_order
            ],
            dtype=float,
        )
        counts = np.array(
            [
                subset.loc[genre, "movie_count"] if genre in subset.index else 0
                for genre in genre_order
            ],
            dtype=int,
        )
        bars = axis.barh(
            positions + offset,
            rates * 100,
            height=bar_height,
            color=colors[flag],
            label="In a collection" if flag else "Not in a collection",
        )
        for bar, rate, count in zip(bars, rates, counts):
            if np.isnan(rate):
                continue
            axis.text(
                rate * 100 + 0.8,
                bar.get_y() + bar.get_height() / 2,
                f"{rate:.0%} (n={count})",
                va="center",
                fontsize=7.2,
            )

    axis.set_yticks(positions, labels=genre_order)
    axis.set_xlim(0, 113)
    axis.set_ylim(-0.55, len(genre_order) + 0.60)
    axis.set_xlabel("Observed success rate (%)", fontsize=8)
    axis.set_title(
        "Observed success rate by primary genre\nand collection status",
        loc="left",
        pad=9,
        fontsize=10,
        fontweight="bold",
    )
    axis.grid(axis="x", color="#D1D5DB", linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="x", labelsize=7.2)
    axis.tick_params(axis="y", length=0, labelsize=8)
    axis.legend(
        frameon=False,
        loc="upper left",
        ncol=2,
        fontsize=7.2,
        columnspacing=0.9,
        handlelength=1.4,
    )
    axis.text(
        0.5,
        -0.20,
        "Eight most common primary genres.",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=7.2,
        color="#4B5563",
    )
    figure.subplots_adjust(top=0.82, bottom=0.21, left=0.28, right=0.94)
    return save_figure(figure, "success_by_genre_collection.png")


def run_eda() -> dict[str, object]:
    sns.set_theme(style="whitegrid", context="notebook")
    cleaned, modeling, raw_features, target = load_datasets()
    summary = dataset_summary(cleaned, modeling)
    missingness = missingness_summary(cleaned)
    correlation, genre_collection = association_tables(raw_features, target)
    yearly = yearly_summary(modeling)

    tables = [
        save_table(summary, "dataset_summary.csv"),
        save_table(missingness, "core_missingness.csv"),
        save_table(correlation, "pre_release_spearman.csv", index=True),
        save_table(genre_collection, "success_by_primary_genre_collection.csv"),
        save_table(yearly, "yearly_success_summary.csv"),
    ]
    # Bộ EDA chính gồm ba hình độc lập được lưu trong reports/figures.
    # Hình ghép cũ vẫn được giữ dưới dạng hàm để tương thích, nhưng không còn
    # được gọi trong pipeline chính vì báo cáo sử dụng hai hình độc lập.
    figures = [
        plot_dataset_overview(modeling, missingness),
        plot_spearman_heatmap(correlation),
        plot_genre_collection_rates(genre_collection),
    ]

    print(f"movies_cleaned: {len(cleaned)}")
    print(f"movies_modeling: {len(modeling)}")
    print(f"Đã tạo {len(tables)} bảng và {len(figures)} hình EDA TMDb-only.")
    return {
        "cleaned": cleaned,
        "modeling": modeling,
        "raw_features": raw_features,
        "target": target,
        "dataset_summary": summary,
        "missingness": missingness,
        "correlation": correlation,
        "genre_collection": genre_collection,
        "yearly": yearly,
        "table_paths": tables,
        "figure_paths": figures,
    }


if __name__ == "__main__":
    run_eda()
