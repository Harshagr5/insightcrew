# SPDX-License-Identifier: Apache-2.0
"""One-command demo: analyse the bundled sample sales data.

Usage:
    python run_demo.py                      # uses $NOOA_PROVIDER (default: nvidia)
    NOOA_PROVIDER=ollama python run_demo.py # fully local, no API key
"""

from __future__ import annotations

import asyncio
import os

from insightcrew.llm_setup import build_llm
from insightcrew.orchestrator import InsightCrew

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "data", "sample_sales.csv")
QUESTION = "Which region and product category drive our revenue, and what changed over the year?"


async def main() -> None:
    llm = build_llm()
    crew = InsightCrew(csv_path=CSV, llm=llm, charts_dir=os.path.join(HERE, "charts"))
    report = await crew.run(QUESTION)

    print("\n" + "=" * 70)
    print(report.title)
    print("=" * 70)
    print(f"Headline : {report.headline}")
    print(f"Confidence: {report.confidence}\n")
    print(report.body_markdown)


if __name__ == "__main__":
    asyncio.run(main())
