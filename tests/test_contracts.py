# SPDX-License-Identifier: Apache-2.0
"""Contracts are the backbone of the pipeline — verify their validation rules."""

import pytest
from pydantic import ValidationError

from insightcrew.contracts import (
    AnalysisPlan,
    AnalysisStep,
    FinalReport,
    Review,
    StepResult,
)


def _step(**kw):
    base = dict(id=1, goal="g", dimension="region", metric="total_revenue")
    base.update(kw)
    return AnalysisStep(**base)


def test_step_requires_positive_id():
    with pytest.raises(ValidationError):
        _step(id=0)


def test_step_rejects_unknown_dimension():
    with pytest.raises(ValidationError):
        _step(dimension="planet")


def test_step_rejects_unknown_metric():
    with pytest.raises(ValidationError):
        _step(metric="total_profit")


def test_plan_requires_at_least_one_step():
    with pytest.raises(ValidationError):
        AnalysisPlan(question="q", steps=[])


def test_plan_caps_step_count():
    steps = [_step(id=i + 1) for i in range(7)]
    with pytest.raises(ValidationError):
        AnalysisPlan(question="q", steps=steps)


def test_report_confidence_is_constrained():
    with pytest.raises(ValidationError):
        FinalReport(title="t", headline="h", confidence="very-high", body_markdown="b")


def test_review_defaults_are_empty_lists():
    r = Review(approved=True)
    assert r.issues == []
    assert r.missing == []


def test_stepresult_chart_is_optional():
    r = StepResult(step_id=1, goal="g", finding="North leads at $189k")
    assert r.chart_path is None
