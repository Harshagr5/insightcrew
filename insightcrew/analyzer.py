# SPDX-License-Identifier: Apache-2.0
"""Deterministic analysis: given a (dimension, metric), compute the numbers with pandas
and save a chart. No LLM here, so the figures are exact and reproducible."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# metric -> (column, aggregation, human label)
METRICS: dict[str, tuple[str, str, str]] = {
    "total_revenue": ("revenue", "sum", "Total revenue"),
    "total_units": ("units", "sum", "Total units"),
    "avg_unit_price": ("unit_price", "mean", "Average unit price"),
}


def _fmt(metric: str, value: float) -> str:
    if metric == "total_units":
        return f"{value:,.0f} units"
    if metric == "avg_unit_price":
        return f"${value:,.2f}"
    return f"${value:,.0f}"


def compute_step(
    df: pd.DataFrame,
    dimension: str,
    metric: str,
    charts_dir: str = "charts",
    step_id: int = 0,
) -> tuple[str, str]:
    """Compute one analysis step. Returns (finding_text, chart_path).

    - dimension == "month": trend over time (line chart + change/peak summary).
    - otherwise: ranking across the dimension (bar chart + top/bottom + full ranking).
    """
    column, agg, label = METRICS.get(metric, METRICS["total_revenue"])
    os.makedirs(charts_dir, exist_ok=True)

    if dimension == "month":
        d = df.copy()
        d["month"] = pd.to_datetime(d["date"]).dt.to_period("M").astype(str)
        series = d.groupby("month")[column].agg(agg).sort_index()
        first, last = series.iloc[0], series.iloc[-1]
        pct = (last / first - 1) * 100 if first else 0.0
        peak = series.idxmax()
        finding = (
            f"{label} moved from {_fmt(metric, first)} in {series.index[0]} to "
            f"{_fmt(metric, last)} in {series.index[-1]} ({pct:+.0f}%); "
            f"peak was {series.index[series.argmax()]} at {_fmt(metric, series.max())}."
        )
        ax = series.plot(kind="line", marker="o", figsize=(9, 5))
        title = f"{label} by month"
    else:
        series = df.groupby(dimension)[column].agg(agg).sort_values(ascending=False)
        total = series.sum()
        top, bottom = series.index[0], series.index[-1]
        share_txt = ""
        if metric == "total_revenue" and total:
            share_txt = f" ({series.iloc[0] / total * 100:.0f}% of total)"
        ranking = ", ".join(f"{idx}={_fmt(metric, val)}" for idx, val in series.items())
        finding = (
            f"{top} leads by {label.lower()} at {_fmt(metric, series.iloc[0])}{share_txt}; "
            f"{bottom} is lowest at {_fmt(metric, series.iloc[-1])}. "
            f"Full ranking: {ranking}."
        )
        ax = series.plot(kind="bar", figsize=(9, 5))
        title = f"{label} by {dimension}"

    ax.set_title(title)
    ax.set_ylabel(label)
    plt.tight_layout()
    path = os.path.join(charts_dir, f"{step_id:02d}_{metric}_by_{dimension}.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return finding, path
