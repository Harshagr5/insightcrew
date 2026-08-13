# SPDX-License-Identifier: Apache-2.0
"""Pipeline: plan -> compute each step -> critique -> (re-plan if gaps) -> write report.

Orchestration is plain Python; only Planner/Critic/Writer call the model.
"""

from __future__ import annotations

import os

from .contracts import FinalReport, Review, StepResult


class InsightCrew:
    """Compose Planner + deterministic analysis + Critic + Report Writer."""

    def __init__(self, csv_path: str, llm, charts_dir: str = "charts", max_revisions: int = 1):
        # lazy imports so from_agents() tests need neither nooa nor a live model
        import pandas as pd

        from .agents import CriticAgent, PlannerAgent, ReportWriterAgent
        from .datatools import describe_dataset

        os.makedirs(charts_dir, exist_ok=True)
        self.charts_dir = charts_dir
        self._df = pd.read_csv(csv_path)
        schema = describe_dataset(csv_path)

        self.planner = PlannerAgent(llm=llm, schema=schema)
        self.critic = CriticAgent(llm=llm)
        self.writer = ReportWriterAgent(llm=llm)
        self.max_revisions = max_revisions

    @classmethod
    def from_agents(cls, planner, critic, writer, df, charts_dir="charts", max_revisions=1):
        """Build a crew from pre-made agents + a DataFrame (used for testing)."""
        self = cls.__new__(cls)
        self.planner = planner
        self.critic = critic
        self.writer = writer
        self._df = df
        self.charts_dir = charts_dir
        self.max_revisions = max_revisions
        return self

    def _analyze(self, step) -> StepResult:
        """Deterministically execute one planned step (no LLM)."""
        from .analyzer import compute_step

        finding, chart = compute_step(
            self._df, step.dimension, step.metric,
            charts_dir=self.charts_dir, step_id=step.id,
        )
        return StepResult(step_id=step.id, goal=step.goal, finding=finding, chart_path=chart)

    @staticmethod
    def _format_findings(results: list[StepResult]) -> str:
        out = []
        for r in results:
            chart = f" [chart: {r.chart_path}]" if r.chart_path else ""
            out.append(f"Step {r.step_id} - {r.goal}: {r.finding}{chart}")
        return "\n".join(out) if out else "(no findings)"

    @staticmethod
    def _refusal(question: str, note: str) -> FinalReport:
        """A deterministic 'cannot answer' report — no engine, no invented numbers."""
        reason = note or "The question needs data or columns this dataset does not have."
        return FinalReport(
            title="Cannot answer from the available data",
            headline=reason,
            confidence="low",
            body_markdown=(
                f"**Question:** {question}\n\n"
                f"This question can't be answered with the columns and metrics in this "
                f"dataset. {reason}\n\n"
                f"Available breakdowns: region, category, channel, segment, product, month. "
                f"Available metrics: total revenue, total units, average unit price."
            ),
        )

    async def run(self, question: str) -> FinalReport:
        """Run the full pipeline for a question and return a FinalReport."""
        plan = await self.planner.plan(question)
        if not plan.answerable or not plan.steps:
            return self._refusal(question, plan.note)

        results: list[StepResult] = []
        done: set[tuple[str, str]] = set()
        for step in plan.steps:
            results.append(self._analyze(step))
            done.add((step.dimension, step.metric))

        findings = self._format_findings(results)
        review: Review = await self.critic.review(question, findings)

        # Revision is deterministic: the Critic names specific (dimension, metric) steps,
        # and we run exactly those through the engine — so a revision always adds coverage
        # (or we stop, if the Critic only names breakdowns we already have).
        revisions = 0
        while (
            not review.approved
            and review.missing_steps
            and revisions < self.max_revisions
        ):
            added = False
            for step in review.missing_steps:
                key = (step.dimension, step.metric)
                if key in done:
                    continue
                step.id = len(results) + 1  # keep ids/chart names unique
                results.append(self._analyze(step))
                done.add(key)
                added = True
            if not added:
                break  # nothing new proposed; don't loop forever
            findings = self._format_findings(results)
            review = await self.critic.review(question, findings)
            revisions += 1

        charts = [r.chart_path for r in results if r.chart_path]
        return await self.writer.write(question, findings, charts)
