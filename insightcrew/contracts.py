# SPDX-License-Identifier: Apache-2.0
"""Pydantic models passed between the agents and the analysis engine.

Planner steps use Literal dimension/metric so the model's output is always valid and
directly executable by the engine.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# The fixed menus the Planner must choose from. Using "month" as a dimension is how
# the crew analyses change over time.
Dimension = Literal["region", "category", "channel", "segment", "product", "month"]
Metric = Literal["total_revenue", "total_units", "avg_unit_price"]


class AnalysisStep(BaseModel):
    """One concrete, executable analysis: aggregate `metric` grouped by `dimension`."""

    id: int = Field(ge=1, description="1-based step number")
    goal: str = Field(description="Plain-language description of what this step reveals")
    dimension: Dimension = Field(description="Which column to group by")
    metric: Metric = Field(description="Which quantity to aggregate")


class AnalysisPlan(BaseModel):
    """An ordered plan for answering the user's question about a dataset.

    When the question cannot be answered with the available dimensions/metrics, the
    Planner sets answerable=False, explains why in `note`, and leaves `steps` empty —
    so the system declines instead of forcing the question onto the nearest breakdown.
    """

    question: str = Field(description="The user's original question")
    answerable: bool = Field(
        default=True,
        description="False if the question needs data/columns this dataset does not have",
    )
    note: str = Field(default="", description="If not answerable, a one-line reason why")
    steps: list[AnalysisStep] = Field(default_factory=list, max_length=6)


class StepResult(BaseModel):
    """The outcome of executing one analysis step (produced deterministically)."""

    step_id: int
    goal: str
    finding: str = Field(description="The concrete result, computed from the data")
    chart_path: str | None = Field(default=None, description="Path to the saved chart")


class Review(BaseModel):
    """A critic's judgment, plus any concrete breakdown it wants added before approving."""

    approved: bool = Field(description="True if the findings already answer the question")
    issues: list[str] = Field(default_factory=list, description="Short reasons, if rejecting")
    missing_steps: list[AnalysisStep] = Field(
        default_factory=list,
        description="Specific, computable (dimension, metric) steps to add before approving",
    )


class FinalReport(BaseModel):
    """The assembled, human-readable analysis report."""

    title: str
    headline: str = Field(description="One-sentence takeaway a busy reader could quote")
    confidence: Literal["low", "medium", "high"]
    body_markdown: str = Field(
        description="Full report in Markdown, grounded only in the given findings"
    )
