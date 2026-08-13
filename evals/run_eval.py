# SPDX-License-Identifier: Apache-2.0
"""Evaluate the Planner (and optionally the Critic) against a labelled question set.

Metrics
-------
Over answerable questions:
  exact-match  — predicted (dimension, metric) set equals an accepted gold plan
  coverage     — predicted set contains all pairs of some accepted gold plan
  wrong-refusal— the system refused a question it should have answered
Over unanswerable questions:
  refusal-acc  — correctly declined
  false-answer — produced a plan anyway (the failure the refusal path exists to prevent)
With --with-critic (answerable questions only):
  critic-fire  — how often the Critic rejects the first pass
  revision-help— of those, how often applying the Critic's named steps reaches coverage

Hosted models are not fully deterministic even at temperature 0, so use --repeat N to
run the suite several times and report mean + range.

Usage
-----
  python evals/run_eval.py --model nvidia_nim/meta/llama-3.1-8b-instruct
  python evals/run_eval.py --repeat 3 --with-critic
  python evals/run_eval.py --verbose             # per-question predicted vs gold
  python evals/run_eval.py --fake                # oracle planner: sanity-checks the scorer
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from insightcrew.contracts import AnalysisPlan, AnalysisStep  # noqa: E402
from insightcrew.datatools import describe_dataset, load_dataframe  # noqa: E402

CSV = os.path.join(ROOT, "data", "sample_sales.csv")
QUESTIONS = os.path.join(HERE, "questions.jsonl")


def load_questions(limit=None):
    rows = []
    with open(QUESTIONS) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def gold_sets(gold):
    return [frozenset((d, m) for d, m in plan) for plan in gold]


def pred_pairs(plan: AnalysisPlan):
    return frozenset((s.dimension, s.metric) for s in plan.steps)


class OraclePlanner:
    """Perfect planner built from gold labels — used only to sanity-check the scorer."""

    def __init__(self, by_question):
        self._by_q = by_question

    async def plan(self, question: str) -> AnalysisPlan:
        row = self._by_q[question]
        if not row["answerable"]:
            return AnalysisPlan(question=question, answerable=False, note="not in data", steps=[])
        steps = [
            AnalysisStep(id=i + 1, goal=f"{d} / {m}", dimension=d, metric=m)
            for i, (d, m) in enumerate(row["gold"][0])
        ]
        return AnalysisPlan(question=question, answerable=True, steps=steps)


def build_real_agents(model, with_critic):
    os.environ.setdefault("NOOA_PROVIDER", "nvidia")
    if model:
        os.environ["NVIDIA_MODEL"] = model
    from insightcrew.agents import CriticAgent, PlannerAgent
    from insightcrew.llm_setup import build_llm

    llm = build_llm(temperature=0.0)
    planner = PlannerAgent(llm=llm, schema=describe_dataset(CSV))
    critic = CriticAgent(llm=llm) if with_critic else None
    return planner, critic


def _empty_stats():
    return {
        "n": 0, "answerable": 0, "unanswerable": 0, "errors": 0,
        "exact": 0, "covered": 0, "wrong_refusal": 0,
        "refused_ok": 0, "false_answer": 0,
        "critic_fired": 0, "critic_seen": 0, "rev_help": 0, "rev_eligible": 0,
    }


async def score_once(questions, planner, critic, df):
    from insightcrew.analyzer import compute_step  # lazy: needs pandas/matplotlib

    s = _empty_stats()
    records = []
    for row in questions:
        s["n"] += 1
        q = row["question"]
        try:
            plan = await planner.plan(q)
        except Exception as e:  # noqa: BLE001
            s["errors"] += 1
            records.append({"id": row["id"], "type": row["type"], "tag": "ERROR", "q": q})
            print(f"  [err] q{row['id']}: {type(e).__name__}: {e}")
            continue

        if row["answerable"]:
            s["answerable"] += 1
            gsets = gold_sets(row["gold"])
            pred = pred_pairs(plan) if plan.answerable else frozenset()
            exact = any(pred == g for g in gsets)
            covered = any(g <= pred for g in gsets)
            s["exact"] += int(exact)
            s["covered"] += int(covered)
            s["wrong_refusal"] += int(not plan.answerable)
            tag = ("WRONG-REFUSAL" if not plan.answerable
                   else "EXACT" if exact else "COVERED" if covered else "UNCOVERED")
            records.append({"id": row["id"], "type": row["type"], "tag": tag, "q": q,
                            "pred": sorted(pred)})

            if critic is not None and plan.answerable and plan.steps:
                s["critic_seen"] += 1
                findings = "\n".join(
                    f"Step {st.id}: " + compute_step(df, st.dimension, st.metric,
                                                     charts_dir="/tmp/eval_charts", step_id=st.id)[0]
                    for st in plan.steps
                )
                review = await critic.review(q, findings)
                if not review.approved:
                    s["critic_fired"] += 1
                    if not covered:
                        s["rev_eligible"] += 1
                        extra = frozenset((x.dimension, x.metric) for x in review.missing_steps)
                        if any(g <= (pred | extra) for g in gsets):
                            s["rev_help"] += 1
        else:
            s["unanswerable"] += 1
            refused = not plan.answerable
            s["refused_ok"] += int(refused)
            s["false_answer"] += int(not refused)
            records.append({"id": row["id"], "type": row["type"],
                            "tag": "REFUSED-OK" if refused else "FALSE-ANSWER", "q": q})
    return s, records


# metric label, numerator key, denominator key
SPECS = [
    ("planner exact-match (answerable)", "exact", "answerable"),
    ("planner coverage (answerable)", "covered", "answerable"),
    ("wrong refusals (answerable)", "wrong_refusal", "answerable"),
    ("refusal accuracy (unanswerable)", "refused_ok", "unanswerable"),
    ("false-answer rate (unanswerable)", "false_answer", "unanswerable"),
    ("critic fire rate", "critic_fired", "critic_seen"),
    ("revision helped (of eligible)", "rev_help", "rev_eligible"),
]


def _rate(s, num, den):
    d = s[den]
    return None if d == 0 else 100 * s[num] / d


def render(model_label, runs):
    lines = [f"### Model: `{model_label}`  (runs={len(runs)}, n={runs[0]['n']}, "
             f"errors={sum(r['errors'] for r in runs)})", ""]
    if len(runs) == 1:
        lines += ["| metric | value |", "|---|---|"]
        s = runs[0]
        for label, num, den in SPECS:
            r = _rate(s, num, den)
            if r is None:
                lines.append(f"| {label} | - |")
            else:
                lines.append(f"| {label} | {r:.0f}% ({s[num]}/{s[den]}) |")
    else:
        lines += ["| metric | mean | range (min–max) |", "|---|---|---|"]
        for label, num, den in SPECS:
            vals = [v for v in (_rate(s, num, den) for s in runs) if v is not None]
            if not vals:
                lines.append(f"| {label} | - | - |")
            else:
                mean = sum(vals) / len(vals)
                lines.append(f"| {label} | {mean:.0f}% | {min(vals):.0f}–{max(vals):.0f}% |")
    return "\n".join(lines)


def render_verbose(records):
    out = ["", "Per-question (last run):", ""]
    for r in records:
        pred = " pred=" + ",".join(f"{d}/{m}" for d, m in r.get("pred", [])) if r.get("pred") else ""
        out.append(f"  q{r['id']:>2} [{r['type']:<12}] {r['tag']:<13} {r['q'][:52]}{pred}")
    return "\n".join(out)


async def evaluate(args):
    questions = load_questions(args.limit)
    by_q = {r["question"]: r for r in questions}
    if args.fake:
        planner, critic, model_label = OraclePlanner(by_q), None, "oracle (fake)"
    else:
        planner, critic = build_real_agents(args.model, args.with_critic)
        model_label = args.model or os.getenv("NVIDIA_MODEL", "default")

    df = load_dataframe(CSV)
    runs, last_records = [], []
    for i in range(args.repeat):
        if args.repeat > 1:
            print(f"  run {i + 1}/{args.repeat} ...")
        stats, records = await score_once(questions, planner, critic, df)
        runs.append(stats)
        last_records = records
    return model_label, runs, last_records


def main():
    p = argparse.ArgumentParser(description="Evaluate InsightCrew's Planner/Critic.")
    p.add_argument("--model", default=None, help="LiteLLM model name")
    p.add_argument("--with-critic", action="store_true", help="also measure critic behaviour")
    p.add_argument("--repeat", type=int, default=1, help="run the suite N times (variance)")
    p.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    p.add_argument("--verbose", action="store_true", help="print per-question predicted vs gold")
    p.add_argument("--fake", action="store_true", help="oracle planner: sanity-check the scorer")
    p.add_argument("--out", default=os.path.join(HERE, "RESULTS.md"))
    args = p.parse_args()

    model_label, runs, records = asyncio.run(evaluate(args))
    table = render(model_label, runs)
    print("\n" + table + "\n")
    if args.verbose:
        print(render_verbose(records))
    with open(args.out, "a") as f:
        f.write(table + "\n\n")
    print(f"(appended to {args.out})")


if __name__ == "__main__":
    main()
