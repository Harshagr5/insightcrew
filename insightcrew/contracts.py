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
    """An ordered plan for answering the user's question about a dataset."""

    question: str = Field(description="The user's original question")
    steps: list[AnalysisStep] = Field(min_length=1, max_length=6)


class StepResult(BaseModel):
    """The outcome of executing one analysis step (produced deterministically)."""

    step_id: int
    goal: str
    finding: str = Field(description="The concrete result, computed from the data")
    chart_path: str | None = Field(default=None, description="Path to the saved chart")


class Review(BaseModel):
    """A critic's judgment of whether the findings answer the question."""

    approved: bool = Field(description="True if the findings fully answer the question")
    issues: list[str] = Field(default_factory=list, description="Problems to fix")
    missing: list[str] = Field(
        default_factory=list, description="Aspects of the question not yet addressed"
    )


class FinalReport(BaseModel):
    """The assembled, human-readable analysis report."""

    title: str
    headline: str = Field(description="One-sentence takeaway a busy reader could quote")
    confidence: Literal["low", "medium", "high"]
    body_markdown: str = Field(
        description="Full report in Markdown, grounded only in the given findings"
    )
