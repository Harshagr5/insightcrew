# Architecture & design decisions

A short tour of *why* InsightCrew is built the way it is — useful for interviews.

## The pipeline

```
question ─▶ Planner (LLM) ─▶ [deterministic pandas engine] ─▶ Critic (LLM)
               ▲                                                  │
               └───────────── revise if rejected ◀────────────────┘
                                                                  │ approved / out of tries
                                                                  ▼
                                                    Report Writer (LLM) ─▶ FinalReport
```

## Key decisions

**1. LLMs for judgment, real code for computation.** The hardest lesson of the project:
letting an LLM compute numbers produces confident, wrong reports. So the LLM agents only
*decide* and *narrate*, while a deterministic pandas engine does every calculation. The
numbers are therefore exact and reproducible, and the Report Writer is prompted to use
*only* the engine's findings — it has no path to invent a figure.

**2. Constrain the plan to a fixed menu.** Each analysis step is a `(dimension, metric)`
pair drawn from `Literal` types. Because NOOA validates the Planner's structured output
against those types, the plan is *always* valid and directly executable — no free-text
step that the engine might fail to interpret.

**3. One structured call per agent — no code-execution loop.** Every agent uses
`PredictStrategy` (a single structured-output call), not `CodeActStrategy`. This is what
makes the crew reliable on small/free models: there's no multi-iteration REPL to time out
or to fail to converge. The whole run is a handful of fast calls.

**4. Deterministic orchestration, testable everywhere.** The control flow (loop over
steps, feed critic gaps back to the planner, bound revisions) is plain Python. Tests inject
fake agents via `InsightCrew.from_agents(...)` and run the *real* engine on a tiny
DataFrame — so both the flow (including the revision path) and the computed numbers are
verified with no network and no model.

**5. Self-correction with a budget.** The Critic can reject the findings and hand the
Planner a targeted list of what's missing, triggering another round, bounded by
`--max-revisions` so it can't loop forever.

**6. Provider independence.** `llm_setup.build_llm()` is the only vendor-aware code.
Because NOOA wraps LiteLLM, switching between NVIDIA NIM, Gemini, Ollama, or OpenAI is a
one-line env change and touches no agent code.

## What we tried first, and why we changed it

The initial design used a CodeAct analyst agent that wrote and executed its own pandas.
It's the flashy NOOA capability, but on free infrastructure it failed two ways: the large
model was too slow (multi-iteration loops timed out) and the small model couldn't reliably
write working code or return a typed object — and when it did "work", it fabricated numbers
in the final report. Replacing that one component with a deterministic engine kept the
multi-agent LLM design while making the output correct, fast, and safe. Graceful handling
of that failure mode is itself a feature: the system never ships a made-up number.

## Trade-offs & limits

- The set of analyses is bounded by the `dimension`/`metric` menu. Extending it is a small,
  localized change in `contracts.py` (add a `Literal` value) and `analyzer.py` (handle it).
- It no longer *writes* novel analysis code — a deliberate trade of open-endedness for
  correctness and reliability.
- Single dataset per run; no persistence. NOOA's memory subsystem is a natural next step.

## Where to extend

- Add dimensions/metrics (e.g. profit margin, week-over-week growth).
- Add a `ForecastAgent` that plans a forecast the engine computes with statsmodels.
- Swap the CSV loader for a SQL/warehouse source.
- Emit the report as PDF/HTML; turn on NOOA tracing to inspect the run as a call graph.
