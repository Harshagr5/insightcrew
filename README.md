# InsightCrew

Multi-agent data analysis built on [NVIDIA NOOA](https://github.com/NVIDIA-NeMo/labs-OO-Agents).
Ask a question about a CSV; the crew plans the analysis, computes the numbers with pandas,
checks its own coverage, and writes a short report with charts. A Fun way to work with Agents.

## How it works

LLM agents start by making the decisions; a deterministic pandas engine does the math. The Planner
chooses which `(dimension, metric)` breakdowns answer the question, the engine computes
them and saves charts, the Critic checks whether the findings cover the question, and the
Writer produces the report from the computed numbers only — so figures are never invented.

```
Planner (LLM) → analysis engine (pandas) → Critic (LLM) → Writer (LLM)
        ▲                                        │
        └──────────── re-plan if gaps ───────────┘
```

- Agents: `insightcrew/agents.py`
- Deterministic engine: `insightcrew/analyzer.py`
- Orchestration + revision loop: `insightcrew/orchestrator.py`

## Run

Python 3.12+.

```bash
pip install -e .
export NOOA_PROVIDER=nvidia
export NVIDIA_API_KEY=...          # free key from build.nvidia.com
python run_demo.py
```

Or against your own file:

```bash
insightcrew --csv data/sample_sales.csv --question "Which region drives revenue?"
```

Providers are selected with `NOOA_PROVIDER`: `nvidia` (default), `gemini`, `ollama`,
`openai`. Every agent makes a single structured call, so the whole run takes a few seconds.

## Example

See [`sample_output/`](sample_output) for a generated report and charts. On the bundled
dataset the engine finds North leading region revenue at 189,749 USD (48% of total),
Electronics at 312,236 USD (79%), and total revenue up 104% across the year, peaking in
October.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The agents are faked (no network); the real pandas engine runs on a small in-memory frame,
so both the control flow and the computed numbers are checked.

## Layout

```
insightcrew/   package: contracts, agents, analyzer, orchestrator, llm_setup, cli
tests/         contract, engine, and orchestration tests
data/          sample dataset
notebooks/     Colab runner
sample_output/ an example report + charts
```

Design notes and the reasoning behind the deterministic-engine split are in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
