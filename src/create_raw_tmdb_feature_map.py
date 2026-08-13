"""Render a minimal feature map of the 15 TMDb input metadata fields."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "figures" / "tmdb_raw_feature_map.png"

FEATURE_GROUPS = [
    ("Financial and\ntime", ["budget", "runtime", "release_date"]),
    ("Movie\ncontent", ["genres", "overview", "tagline", "keywords"]),
    (
        "Language and\nproduction",
        [
            "original_\nlanguage",
            "production_\ncountries",
            "production_\ncompanies",
            "spoken_\nlanguages",
        ],
    ),
    (
        "Franchise and\ncrew",
        ["belongs_to_\ncollection", "credits.cast", "credits.crew"],
    ),
    ("Release\nmetadata", ["release_dates"]),
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    source_count = sum(len(fields) for _, fields in FEATURE_GROUPS)
    if source_count != 15:
        raise ValueError(f"Feature map phải có 15 trường, hiện có {source_count}.")

    navy = "#173F5F"
    teal = "#2A7F8E"
    text = "#202B36"
    grid = "#D7E0E8"

    figure, axis = plt.subplots(figsize=(14.5, 6.1))
    figure.patch.set_facecolor("white")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    axis.text(
        0.05,
        0.925,
        "15 TMDb metadata fields before feature engineering",
        ha="left",
        va="center",
        fontsize=23,
        fontweight="bold",
        color=text,
    )
    axis.add_patch(Rectangle((0.05, 0.852), 0.90, 0.004, facecolor=teal, edgecolor="none"))

    left, right = 0.05, 0.95
    bottom, top = 0.15, 0.790
    gap = 0.012
    column_width = (right - left - gap * 4) / 5

    for index, (group, fields) in enumerate(FEATURE_GROUPS):
        x = left + index * (column_width + gap)
        if index:
            axis.plot([x - gap / 2, x - gap / 2], [bottom, top], color=grid, linewidth=1.0)

        axis.add_patch(Rectangle((x, top - 0.105), column_width, 0.105, facecolor=navy, edgecolor="none"))
        axis.text(
            x + 0.018,
            top - 0.051,
            group,
            ha="left",
            va="center",
            fontsize=11.8,
            fontweight="bold",
            color="white",
            linespacing=1.18,
        )
        content_top = top - 0.165
        content_bottom = bottom + 0.090
        item_positions = (
            [(content_top + content_bottom) / 2]
            if len(fields) == 1
            else np.linspace(content_top, content_bottom, len(fields))
        )
        for item_y, field in zip(item_positions, fields):
            axis.text(x + 0.020, item_y, "•", ha="left", va="center", fontsize=13, color=teal)
            axis.text(
                x + 0.048,
                item_y,
                field,
                ha="left",
                va="center",
                fontsize=10.1,
                color=text,
                fontfamily="DejaVu Sans Mono",
                linespacing=1.25,
            )

    axis.plot([left, right], [bottom, bottom], color=grid, linewidth=1.0)

    figure.savefig(OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"source_metadata_count={source_count}")
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()
