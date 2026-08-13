# SPDX-License-Identifier: Apache-2.0
"""Dataset loading and a short schema description for the planner prompt."""

from __future__ import annotations

import pandas as pd


def load_dataframe(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def describe_dataset(csv_path: str, max_examples: int = 3) -> str:
    """A compact, LLM-friendly summary of the columns, used in the planner prompt."""
    df = load_dataframe(csv_path)
    lines = [f"{len(df)} rows x {len(df.columns)} columns.", "Columns:"]
    for col in df.columns:
        dtype = str(df[col].dtype)
        examples = ", ".join(str(v) for v in df[col].dropna().unique()[:max_examples])
        lines.append(f"  - {col} ({dtype}); examples: {examples}")
    return "\n".join(lines)
