# SPDX-License-Identifier: Apache-2.0
"""The deterministic data helpers should work on the bundled sample dataset."""

import os

from insightcrew.datatools import describe_dataset, load_dataframe

CSV = os.path.join(os.path.dirname(__file__), "..", "data", "sample_sales.csv")


def test_load_dataframe_has_expected_shape():
    df = load_dataframe(CSV)
    assert len(df) > 100
    for col in ["date", "region", "category", "revenue"]:
        assert col in df.columns


def test_describe_dataset_mentions_columns():
    desc = describe_dataset(CSV)
    for col in ["region", "category", "revenue"]:
        assert col in desc
    assert "rows" in desc
