"""Derived evidence views and experiment reports; never an alternative truth source."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from .evidence import read_json
from .replay_buffer.store import open_buffer


def diagnostics(traces, target=0.01):
    kernels = defaultdict(lambda: defaultdict(list))
    for trace in traces:
        kernels[trace["kernel"]][trace["budget"]].append(trace)
    budgets = sorted({trace["budget"] for trace in traces})
    curve = {}
    for budget in budgets:
        curve[budget] = mean(mean(trace["best_quality"] for trace in rows[budget])
                             for rows in kernels.values())
    previous_budget, previous_quality, area = 0, 0, 0
    for budget in budgets:
        area += (budget - previous_budget) * (curve[budget] + previous_quality) / 2
        previous_budget, previous_quality = budget, curve[budget]
    target_costs = []
    for trace in traces:
        if trace["budget"] != max(budgets):
            continue
        hit = next((event for event in trace["events"] if event["best_quality"] >= target), None)
        if hit:
            target_costs.append(trace["budget"] - hit["remaining_budget"])
    return {"quality_at_budget": curve, "quality_auc_normalized": area / max(budgets),
            "target_quality": target, "cost_to_target_observed_mean": mean(target_costs) if target_costs else None,
            "target_hit_worlds": len(target_costs),
            "attempted_nodes": sum(trace["attempted_nodes"] for trace in traces),
            "decision_batches": sum(trace["decision_batches"] for trace in traces)}


def render_report(pool, experiment=None):
    buffer = open_buffer(pool)
    manifest, worlds = buffer.manifest, buffer.worlds
    attempts = [asdict(attempt) for attempt in buffer.attempts]
    lines = ["# KernelWiki-RSI v1 replay report", "", "## Historical coverage", "",
             f"External optimization attempts: {len(attempts)}. Verified replay worlds: {len(worlds)}.", "",
             (f"Primary worlds: {len(manifest.get('cohorts', {}).get('primary', []))}. "
              f"Diagnostic worlds: {len(worlds)}. Reverse traces: {manifest.get('reverse_traces', 0)}."), "",
             f"Explicit out-of-split attempts: {manifest.get('excluded_attempts', 0)} (retained in excluded.jsonl).", "",
             "| Kernel | All attempts | Replay attempts | Qualified node outcomes | Worlds |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for kernel in sorted({attempt["kernel"] for attempt in attempts}):
        rows = [attempt for attempt in attempts if attempt["kernel"] == kernel]
        lines.append(f"| {kernel} | {len(rows)} | {sum(row['disposition'] == 'replay_ready' for row in rows)} | "
                     f"{sum(row['measurement']['status'] == 'qualified' for row in rows)} | "
                     f"{sum(world.kernel == kernel for world in worlds)} |")
    lines += ["", "Missing provenance remains trace-only; failed or unknown supported outcomes still cost budget.",
              "Historical rounds with changed shared context start new worlds. No continuation is invented.", "",
              "## Replay contract", "", "- KnowledgePolicy evolves the exploration decision layer of the Harness.",
              "- Discovery Agent, reference knowledge, interpreter, evaluator, and recorded worlds are frozen.",
              "- Objective = best root-relative qualified gain - 0.01 × charged direction budget.",
              "- Budgets are 1, 2, 4, 6. Macro aggregation gives equal weight to kernels.",
              "- Primary qualification requires branching, continuation, budget coverage and complete outcomes before policy evaluation.",
              "- Inner evolution selects a working policy only; final publication requires new holdout runs and paired real online evidence.",
              "- Historical seven-kernel test runs are inspected regression data, not a fresh holdout.",
              "- Replay validation is not online validation or causal knowledge attribution."]
    if experiment:
        result = read_json(Path(experiment) / "result.json")
        lines += ["", "## Evolution", "", f"Result: **{result['status']}**.",
                  f"Selected candidate: {result.get('selected')}. Published: {result.get('published', False)}.",
                  result.get("reason", result.get("error", "")), ""]
        for split in ("train", "validation", "test"):
            path = Path(experiment) / f"incumbent-{split}.json"
            if path.exists():
                data = read_json(path)
                lines.append(f"Incumbent {split} Objective: {data['summary']['objective']:.8f}.")
        lines += ["", "Detailed proposals, independent reviews, replay trajectories and frozen manifests live beside this report."]
    lines += ["", "## Evidence gaps", "", "```json", json.dumps(manifest.get("gaps", {}), indent=2), "```", ""]
    return "\n".join(lines)
