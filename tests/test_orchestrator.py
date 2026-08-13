# SPDX-License-Identifier: Apache-2.0
"""Orchestration logic tested with fake LLM agents + a real deterministic engine.

The Planner/Critic/Writer are faked (no network), but the analysis step runs the real
pandas engine on a tiny in-memory DataFrame — so we verify both the control flow
(including the critic-driven revision loop) and that real findings/charts are produced.
"""

import pandas as pd
import pytest

from insightcrew import InsightCrew
from insightcrew.contracts import AnalysisPlan, AnalysisStep, FinalReport, Review


def tiny_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01"],
            "region": ["North", "South", "North", "East"],
            "category": ["Electronics", "Home", "Electronics", "Home"],
            "channel": ["Online", "Retail", "Online", "Retail"],
            "segment": ["Consumer", "Business", "Consumer", "Business"],
            "product": ["Laptop", "Lamp", "Phone", "Vacuum"],
            "units": [2, 5, 1, 3],
            "unit_price": [1000.0, 40.0, 800.0, 250.0],
            "revenue": [2000.0, 200.0, 800.0, 750.0],
        }
    )


class FakePlanner:
    def __init__(self):
        self.calls = 0

    async def plan(self, question: str) -> AnalysisPlan:
        self.calls += 1
        if self.calls == 1:
            steps = [
                AnalysisStep(id=1, goal="Revenue by region", dimension="region", metric="total_revenue"),
                AnalysisStep(id=2, goal="Revenue over time", dimension="month", metric="total_revenue"),
            ]
        else:
            steps = [AnalysisStep(id=3, goal="Revenue by category", dimension="category", metric="total_revenue")]
        return AnalysisPlan(question=question, steps=steps)


class FakeCritic:
    def __init__(self, approvals: list[bool]):
        self.approvals = list(approvals)

    async def review(self, question: str, findings: str) -> Review:
        approved = self.approvals.pop(0)
        return Review(approved=approved, issues=[] if approved else ["thin"], missing_steps=[])


class FakeWriter:
    def __init__(self):
        self.last: dict | None = None

    async def write(self, question: str, findings: str, chart_paths: list[str]) -> FinalReport:
        self.last = {"findings": findings, "chart_paths": chart_paths}
        return FinalReport(title="R", headline="ok", confidence="high", body_markdown=findings)


@pytest.mark.asyncio
async def test_happy_path_analyses_every_step(tmp_path):
    planner, critic, writer = FakePlanner(), FakeCritic([True]), FakeWriter()
    crew = InsightCrew.from_agents(planner, critic, writer, tiny_df(),
                                   charts_dir=str(tmp_path), max_revisions=1)

    report = await crew.run("what drives revenue?")

    assert isinstance(report, FinalReport)
    assert planner.calls == 1                              # critic approved, no revision
    assert len(writer.last["chart_paths"]) == 2           # both steps produced charts
    assert "North" in writer.last["findings"]             # real computed number present
    assert all((tmp_path / p.split("/")[-1]).exists() for p in writer.last["chart_paths"])


class GapCritic:
    """Rejects once with a specific missing step, then approves — mirrors the real critic."""

    def __init__(self):
        self.calls = 0

    async def review(self, question: str, findings: str) -> Review:
        self.calls += 1
        if self.calls == 1:
            return Review(
                approved=False,
                issues=["also break down by category"],
                missing_steps=[AnalysisStep(id=1, goal="by category",
                                            dimension="category", metric="total_revenue")],
            )
        return Review(approved=True)


@pytest.mark.asyncio
async def test_revision_applies_critic_named_step(tmp_path):
    planner, critic, writer = FakePlanner(), GapCritic(), FakeWriter()
    crew = InsightCrew.from_agents(planner, critic, writer, tiny_df(),
                                   charts_dir=str(tmp_path), max_revisions=1)

    await crew.run("q")

    assert planner.calls == 1                             # revision no longer re-plans
    assert critic.calls == 2                             # review -> add step -> re-review
    assert len(writer.last["chart_paths"]) == 3          # 2 planned + 1 critic-added


@pytest.mark.asyncio
async def test_max_revisions_zero_never_revises(tmp_path):
    planner, critic, writer = FakePlanner(), FakeCritic([False]), FakeWriter()
    crew = InsightCrew.from_agents(planner, critic, writer, tiny_df(),
                                   charts_dir=str(tmp_path), max_revisions=0)

    report = await crew.run("q")

    assert planner.calls == 1
    assert isinstance(report, FinalReport)


class RefusingPlanner:
    async def plan(self, question: str) -> AnalysisPlan:
        return AnalysisPlan(question=question, answerable=False, note="needs profit data", steps=[])


@pytest.mark.asyncio
async def test_refuses_unanswerable_without_touching_engine_or_writer(tmp_path):
    critic, writer = FakeCritic([True]), FakeWriter()
    crew = InsightCrew.from_agents(RefusingPlanner(), critic, writer, tiny_df(),
                                   charts_dir=str(tmp_path), max_revisions=1)

    report = await crew.run("what is our profit by region?")

    assert "Cannot answer" in report.title       # deterministic refusal
    assert report.confidence == "low"
    assert writer.last is None                    # Writer never called -> nothing to hallucinate
    assert not list(tmp_path.glob("*.png"))       # engine produced no charts
