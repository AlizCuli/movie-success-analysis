"""Create validated report figures for theatrical release breadth.

The figures are generated from the local modeling table and TMDb enrichment
snapshot.  The expected aggregate counts are checked before the PNG files are
written so that an outdated or mismatched snapshot cannot be published as the
official result.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "data" / "processed" / "movies_modeling.csv"
ENRICHMENT_PATH = ROOT / "data" / "raw" / "tmdb_movie_enrichment.jsonl"
FIGURES_DIR = ROOT / "reports" / "figures"

SINGLE_OUTPUT = FIGURES_DIR / "Tỷ lệ thành công tài chính theo số quốc gia phát hành rạp..png"
INTERACTION_OUTPUT = FIGURES_DIR / "ti le.png"

GROUPS = ["0–5 quốc gia", "6–15 quốc gia", "16–30 quốc gia", ">30 quốc gia"]
BIN_EDGES = [-1, 5, 15, 30, np.inf]

EXPECTED_SINGLE = {
    "0–5 quốc gia": (42, 28.6),
    "6–15 quốc gia": (210, 49.5),
    "16–30 quốc gia": (446, 70.9),
    ">30 quốc gia": (948, 77.7),
}

EXPECTED_INTERACTION = {
    ("0–5 quốc gia", 0): (28, 21.4),
    ("0–5 quốc gia", 1): (14, 42.9),
    ("6–15 quốc gia", 0): (146, 44.5),
    ("6–15 quốc gia", 1): (64, 60.9),
    ("16–30 quốc gia", 0): (278, 62.2),
    ("16–30 quốc gia", 1): (168, 85.1),
    (">30 quốc gia", 0): (485, 65.4),
    (">30 quốc gia", 1): (463, 90.7),
}


def load_data() -> pd.DataFrame:
    modeling = pd.read_csv(MODEL_PATH, usecols=["tmdb_id", "is_successful"])
    records = []
    with ENRICHMENT_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    enrichment = pd.DataFrame(records)
    required = ["tmdb_id", "theatrical_country_count", "is_collection"]
    missing = sorted(set(required) - set(enrichment.columns))
    if missing:
        raise ValueError(f"Thiếu cột enrichment: {missing}")
    if modeling["tmdb_id"].duplicated().any():
        raise ValueError("tmdb_id bị trùng trong tập modeling.")
    enrichment = enrichment[required].drop_duplicates("tmdb_id")
    merged = modeling.merge(enrichment, on="tmdb_id", how="inner", validate="1:1")
    if len(merged) != len(modeling):
        raise ValueError("Không ghép đủ phim modeling với enrichment.")

    merged["release_group"] = pd.cut(
        merged["theatrical_country_count"],
        bins=BIN_EDGES,
        labels=GROUPS,
        include_lowest=True,
    )
    merged["is_collection"] = merged["is_collection"].astype(int)
    return merged


def aggregate(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    single = (
        data.groupby("release_group", observed=False)["is_successful"]
        .agg(movie_count="size", success_rate="mean")
        .reindex(GROUPS)
    )
    interaction = (
        data.groupby(["release_group", "is_collection"], observed=False)["is_successful"]
        .agg(movie_count="size", success_rate="mean")
        .reindex(pd.MultiIndex.from_product([GROUPS, [0, 1]], names=["release_group", "is_collection"]))
    )

    for group, (count, rate) in EXPECTED_SINGLE.items():
        observed = single.loc[group]
        if int(observed["movie_count"]) != count or not np.isclose(observed["success_rate"] * 100, rate, atol=0.06):
            raise ValueError(f"Số liệu single không khớp ở nhóm {group}: {observed.to_dict()}")
    for key, (count, rate) in EXPECTED_INTERACTION.items():
        observed = interaction.loc[key]
        if int(observed["movie_count"]) != count or not np.isclose(observed["success_rate"] * 100, rate, atol=0.06):
            raise ValueError(f"Số liệu interaction không khớp ở nhóm {key}: {observed.to_dict()}")
    return single, interaction


def pct(value: float) -> str:
    return f"{value:.1f}%".replace(".", ",")


def create_single(single: pd.DataFrame) -> None:
    colors = ["#2F80C0", "#3E9CD0", "#20A6AE", "#159B93"]
    figure, axis = plt.subplots(figsize=(8.2, 4.8), facecolor="white")
    bars = axis.bar(GROUPS, single["success_rate"] * 100, color=colors, width=0.58)
    axis.set_ylim(0, 100)
    axis.set_ylabel("Tỷ lệ phim đạt nhãn thành công (%)", fontsize=11)
    axis.set_xlabel("Nhóm theatrical_country_count", fontsize=11, labelpad=10)
    axis.set_title("Tỷ lệ thành công tài chính theo độ rộng phát hành rạp", fontsize=14, fontweight="bold", pad=14)
    axis.text(
        0.5,
        1.01,
        "theatrical_country_count = số quốc gia có bản ghi phát hành rạp trên TMDb",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#4B5563",
    )
    axis.grid(axis="y", color="#D9E1E8", linestyle="--", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    for bar, (_, row) in zip(bars, single.iterrows()):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2.0,
            f"{pct(row['success_rate'] * 100)}\n(n={int(row['movie_count'])})",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    axis.text(
        0.5,
        -0.22,
        "Tập modeling gồm 1.646 phim; nhãn: revenue ≥ 2 × budget.",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=9.5,
        color="#4B5563",
    )
    figure.subplots_adjust(top=0.82, bottom=0.22, left=0.12, right=0.98)
    figure.savefig(SINGLE_OUTPUT, dpi=300, facecolor="white")
    plt.close(figure)


def create_interaction(interaction: pd.DataFrame) -> None:
    colors = {0: "#7E9ACB", 1: "#159B93"}
    labels = {0: "Không thuộc collection", 1: "Thuộc collection"}
    x = np.arange(len(GROUPS))
    width = 0.34
    figure, axis = plt.subplots(figsize=(8.0, 5.0), facecolor="white")
    for flag, offset in ((0, -width / 2), (1, width / 2)):
        subset = interaction.xs(flag, level="is_collection").reindex(GROUPS)
        bars = axis.bar(
            x + offset,
            subset["success_rate"] * 100,
            width=width,
            color=colors[flag],
            label=labels[flag],
        )
        for bar, (_, row) in zip(bars, subset.iterrows()):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{pct(row['success_rate'] * 100)}\n(n={int(row['movie_count'])})",
                ha="center",
                va="bottom",
                fontsize=8.7,
            )
    axis.set_xticks(x, GROUPS)
    axis.set_ylim(0, 105)
    axis.set_ylabel("Tỷ lệ phim đạt nhãn thành công (%)", fontsize=11)
    axis.set_xlabel("Nhóm theatrical_country_count", fontsize=11, labelpad=10)
    axis.set_title("Tỷ lệ thành công theo độ rộng phát hành và collection", fontsize=16, fontweight="bold", pad=14)
    axis.legend(frameon=False, loc="upper left", ncol=2, fontsize=10)
    axis.grid(axis="y", color="#D9E1E8", linestyle="--", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(
        0.5,
        -0.20,
        "Số liệu tính trên 1.646 phim modeling; collection được xác định từ metadata TMDb.",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=9.5,
        color="#4B5563",
    )
    figure.subplots_adjust(top=0.81, bottom=0.22, left=0.11, right=0.98)
    figure.savefig(INTERACTION_OUTPUT, dpi=300, facecolor="white")
    plt.close(figure)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    single, interaction = aggregate(data)
    create_single(single)
    create_interaction(interaction)
    print(f"Đã tạo: {SINGLE_OUTPUT}")
    print(f"Đã tạo: {INTERACTION_OUTPUT}")


if __name__ == "__main__":
    main()
