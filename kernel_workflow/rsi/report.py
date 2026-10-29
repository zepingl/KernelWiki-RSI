"""Derived evidence views and experiment reports; never an alternative truth source."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from .evidence import read_json
from .replay_buffer.store import open_buffer
from .trace.records import OBSERVED_ROLES
from .trace.tree import fold, gap_worklist, knowledge_coverage, knowledge_ledger


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


def trace_metrics(buffer):
    """Design pass conditions over folded trees: grades, multi-direction coverage, observed knowledge use."""
    grades, uses, splits = Counter(), Counter(), {}
    expansions = wide = known = 0
    for view in buffer.trees:
        for node_id, node in view.nodes.items():
            if node.expansion is not None:
                grades[view.grade(node_id)[0]] += 1
            uses.update(use.role for use in node.uses)
        for expansion in view.expansions:
            expansions += 1
            uses.update("planner:" + use.role for use in expansion.uses)
            children = [node_id for node_id, node in view.nodes.items() if node.expansion == expansion.id]
            if expansion.dispatched > 1:
                wide += 1
                known += all(view.outcome(node_id).status != "unknown" for node_id in children)
    primary = set(buffer.manifest.get("tree_cohorts", {}).get("primary", ()))
    for split in ("train", "validation", "test"):
        selected = buffer.select(split, cohort="diagnostic", structure="tree")
        splits[split] = {"trees": len(selected), "primary": sum(view.id in primary for view in selected)}
    share = lambda roles: (sum(uses[role] for role in roles if role.split(":")[-1] in OBSERVED_ROLES)
                           / total if (total := sum(uses[role] for role in roles)) else None)
    direction_roles = [role for role in uses if not role.startswith("planner:")]
    return {"trees": len(buffer.trees), "expansions": expansions, "grades": dict(grades),
            "multi_direction_expansions": wide, "multi_direction_fully_known": known,
            "uses": dict(uses), "direction_uses_observed_share": share(direction_roles),
            "planner_uses_observed_share": share([role for role in uses if role.startswith("planner:")]),
            "splits": splits}


def knowledge_metrics(forest):
    """Knowledge-tree grades and recorded gaps per kind; retired versions are counted separately."""
    grades, gaps, retired = Counter(), Counter(), Counter()
    for view in forest.trees.values():
        if not view.tree.subject.startswith("knowledge:"):
            continue
        kind = view.tree.subject.split(":", 1)[1]
        gone = {item.node for item in view.decisions if item.verdict == "retired"}
        retired[kind] += len(gone)
        grades.update(f"{kind}:{view.grade(node)[0]}" for node in view.nodes if node not in gone)
        gaps.update(f"{kind}:{item.gap.question}" for item in view.gaps)
    return {"grades": dict(sorted(grades.items())), "retired": dict(sorted(retired.items())),
            "gaps": dict(sorted(gaps.items())),
            "external_nodes": sum(len(view.nodes) for view in forest.trees.values() if view.tree.subject == "external")}


def render_trace_report(pool):
    buffer = open_buffer(pool)
    forest = fold(buffer.trace)
    metrics = trace_metrics(buffer)
    ledger = knowledge_ledger(forest)
    coverage = knowledge_coverage(forest)
    worklist = gap_worklist(forest)
    blocked = Counter(reason for row in worklist for reason in row.get("blocked_by", ()))
    recorded = Counter(gap["question"] for row in worklist for gap in row.get("gaps", ()))
    lines = ["# KernelWiki-RSI discovery trace report", "",
             f"Trees: {metrics['trees']}. Expansions: {metrics['expansions']}. "
             f"Generated nodes by grade: {metrics['grades']}.", "",
             f"Multi-direction expansions: {metrics['multi_direction_expansions']}, of which "
             f"{metrics['multi_direction_fully_known']} have every outcome known.", "",
             f"Knowledge uses by role: {metrics['uses']}.", "",
             f"Observed share of direction-level uses: {metrics['direction_uses_observed_share']}. "
             f"Planner uses (tech-lead reads before each plan, repeated per plan) are observed by construction.", "",
             f"KernelWiki coverage of knowledge reads (P1): {coverage['resolved']} resolved, "
             f"{coverage['unregistered']} unregistered, share {coverage['share']}.", "",
             "| Split | Trees | Primary trees |", "| --- | ---: | ---: |"]
    lines += [f"| {split} | {row['trees']} | {row['primary']} |" for split, row in metrics["splits"].items()]
    contrasted = {key: value for key, value in ledger.items() if value["contrasts"]}
    used = {key: value for key, value in ledger.items() if value["roles"]}
    lines += ["", "## Knowledge ledger", "",
              f"KernelWiki nodes: {len(ledger)}. Read by GEAK runs: {len(used)}. "
              f"With paired sibling contrasts: {len(contrasted)}.", "",
              "| KernelWiki node | uses | contrasts | mean gain vs. siblings without it |", "| --- | ---: | ---: | ---: |"]
    for key, value in sorted(contrasted.items(), key=lambda item: -len(item[1]["contrasts"]))[:20]:
        gains = [row["gain"] for row in value["contrasts"]]
        lines.append(f"| `{key}` | {sum(value['roles'].values())} | {len(gains)} | {mean(gains):+.4f} |")
    lines += ["", "Contrasts compare siblings of one expansion (same parent, context and generation); "
              "they are associations, not causal attribution.", "",
              "## Gap worklist", "", "| Blocked by | Nodes |", "| --- | ---: |"]
    lines += [f"| {reason} | {count} |" for reason, count in blocked.most_common()]
    lines += ["", "| Recorded gap | Count |", "| --- | ---: |"]
    lines += [f"| {question} | {count} |" for question, count in recorded.most_common()]
    lines += ["", "```json", json.dumps({**metrics, "knowledge_coverage": coverage,
                                         "knowledge": knowledge_metrics(forest)}, indent=2), "```", ""]
    return "\n".join(lines)
