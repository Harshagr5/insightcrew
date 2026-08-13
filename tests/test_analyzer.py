# SPDX-License-Identifier: Apache-2.0
"""The deterministic engine must produce correct numbers and a saved chart."""

import os

from insightcrew.datatools import load_dataframe
from insightcrew.analyzer import compute_step

CSV = os.path.join(os.path.dirname(__file__), "..", "data", "sample_sales.csv")


def test_ranking_step_reports_real_top_group(tmp_path):
    df = load_dataframe(CSV)
    finding, chart = compute_step(df, "region", "total_revenue",
                                  charts_dir=str(tmp_path), step_id=1)
    # In the bundled dataset North has the highest revenue by construction.
    expected_top = df.groupby("region")["revenue"].sum().idxmax()
    assert finding.startswith(expected_top)
    assert "Full ranking" in finding
    assert os.path.exists(chart)


def test_month_step_describes_a_trend(tmp_path):
    df = load_dataframe(CSV)
    finding, chart = compute_step(df, "month", "total_revenue",
                                  charts_dir=str(tmp_path), step_id=2)
    assert "moved from" in finding and "%" in finding
    assert os.path.exists(chart)


def test_units_metric_uses_unit_label(tmp_path):
    df = load_dataframe(CSV)
    finding, _ = compute_step(df, "category", "total_units",
                              charts_dir=str(tmp_path), step_id=3)
    assert "units" in finding
