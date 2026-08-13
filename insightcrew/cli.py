# SPDX-License-Identifier: Apache-2.0
"""Command-line entry point: run the crew on a CSV and write a Markdown report."""

from __future__ import annotations

import argparse
import asyncio

from .llm_setup import build_llm
from .orchestrator import InsightCrew


def _render(report) -> str:
    return (
        f"# {report.title}\n\n"
        f"> {report.headline}\n\n"
        f"_Confidence: {report.confidence}_\n\n"
        f"{report.body_markdown}\n"
    )


async def _run(args) -> str:
    llm = build_llm(args.provider)
    crew = InsightCrew(
        csv_path=args.csv,
        llm=llm,
        charts_dir=args.charts_dir,
        max_revisions=args.max_revisions,
    )
    report = await crew.run(args.question)
    return _render(report)


def main() -> None:
    p = argparse.ArgumentParser(
        prog="insightcrew",
        description="Autonomous multi-agent data analysis on NVIDIA NOOA.",
    )
    p.add_argument("--csv", required=True, help="Path to the dataset CSV")
    p.add_argument("--question", required=True, help="What you want to know")
    p.add_argument(
        "--provider",
        default=None,
        help="nvidia | ollama | vllm | gemini | openai (default: $NOOA_PROVIDER or nvidia)",
    )
    p.add_argument("--charts-dir", default="charts", help="Where to save chart images")
    p.add_argument("--out", default="report.md", help="Where to write the Markdown report")
    p.add_argument(
        "--max-revisions",
        type=int,
        default=1,
        help="How many critic-driven revision rounds to allow",
    )
    args = p.parse_args()

    markdown = asyncio.run(_run(args))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(f"Report written to {args.out}")
    print(f"Charts (if any) saved under {args.charts_dir}/")


if __name__ == "__main__":
    main()
