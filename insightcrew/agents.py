# SPDX-License-Identifier: Apache-2.0
"""The three LLM agents: Planner, Critic, Report Writer.

Each uses PredictStrategy (a single structured-output call, no code loop), which is why
the crew stays fast and reliable on small models. Computation lives in analyzer.py.
"""

from __future__ import annotations

from nooa import Agent, strategy
from nooa.strategies import PredictStrategy

from .contracts import AnalysisPlan, FinalReport, Review


class PlannerAgent(Agent):
    """You are a senior data analyst who plans which breakdowns answer a question."""

    def __init__(self, llm, schema: str, **kwargs):
        super().__init__(llm=llm, **kwargs)
        self.schema = schema  # interpolated into the plan() prompt

    @strategy(PredictStrategy())
    async def plan(self, question: str) -> AnalysisPlan:
        """Choose 3-5 analysis steps that together answer the question.

        Dataset schema:
        {self.schema}

        Each step selects a `dimension` to group by and a `metric` to aggregate:
        - dimension: region, category, channel, segment, product, or month
          (use "month" to analyse how something changed over time)
        - metric: total_revenue, total_units, or avg_unit_price

        Give each step a short, human-readable `goal`. Cover the question from a few
        complementary angles — e.g. which group leads on revenue, and how revenue
        moved month over month. Do not repeat the same (dimension, metric) twice.
        """
        ...


class CriticAgent(Agent):
    """You are a rigorous reviewer who checks whether findings answer the question."""

    @strategy(PredictStrategy())
    async def review(self, question: str, findings: str) -> Review:
        """Judge whether the findings fully and concretely answer the question.

        Approve only when the findings are specific (real numbers) and cover every
        part of the question. Otherwise set approved=False and list precise issues
        plus anything still missing. Never invent data of your own.
        """
        ...


class ReportWriterAgent(Agent):
    """You are a data storyteller who writes reports grounded ONLY in given findings."""

    @strategy(PredictStrategy())
    async def write(
        self, question: str, findings: str, chart_paths: list[str]
    ) -> FinalReport:
        """Write the final report answering the question.

        Use ONLY the numbers that appear in the findings below — never invent, round
        beyond, or extrapolate past them. Reference charts by their file path where
        relevant. Keep it tight: a quotable headline, an honest confidence level, and
        a Markdown body with short, skimmable sections.

        Findings:
        {findings}

        Charts available: {chart_paths}
        """
        ...
