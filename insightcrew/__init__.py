# SPDX-License-Identifier: Apache-2.0
"""InsightCrew — an autonomous multi-agent data-analysis system built on NVIDIA NOOA."""

from .contracts import AnalysisPlan, AnalysisStep, FinalReport, Review, StepResult
from .orchestrator import InsightCrew

__all__ = [
    "InsightCrew",
    "AnalysisPlan",
    "AnalysisStep",
    "StepResult",
    "Review",
    "FinalReport",
]

__version__ = "0.1.0"
