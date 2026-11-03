"""One online RSI run as a schema-4 code tree whose knowledge uses were identified while the run read them.

Agents read a GEAK-layout view of one KernelWiki commit (knowledge_view). Plan pointers (kk_refs) and files the
agents opened (agent_events logs) name their KernelWiki documents through that view's manifest, so a use is an
observation, not a match. KernelWiki nodes come from projecting the same commit; they must precede these records.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass

from kernelwiki.knowledge import EvidenceRef

from ...discovery import Measurement
from ...evidence import reference
from ...knowledge_view import read_uses, view_uses
from ...trace.gaps import Gap
from ...trace.records import (
    CODE,
    ContextRef,
    Decide,
    Expansion,
    GapRecord,
    Generation,
    Measure,
    Node,
    Plan,
    Slot,
    Tree,
    Use,
)
from ...trace.tree import UNREGISTERED

PLANNER_PHASES = {"plan_round", "analyze"}


@dataclass(frozen=True)
class Step:
    """One online generation: which node it expanded and what came back."""
    request_id: str
    parent: str                 # v1 node id of the expanded node ("<world>/root" for the root)
    generation_result: dict | None
    measurement: Measurement
    source: str | None          # produced candidate source fingerprint, if any
    request: EvidenceRef        # the persisted request (what the generation was given)
    result: EvidenceRef | None  # the lane's result.json, holding the plan and its kk_refs
    events_dir: str             # agent_events logs of this generation


def online_records(*, tree_id, kernel, task, contract, root_source, baseline, steps, stop, view_root, knowledge,
                   generation):
    """`knowledge` is knowledge_index() of the projected view commit: KernelWiki path -> live node id."""
    commit = json.loads((view_root / "manifest.json").read_text())["commit"]
    tree = Tree(tree_id, CODE, contract, task,
                facets={"geak": {"campaign": "online", "kernel": kernel,
                                 "knowledge_identity": f"observed: KernelWiki view manifest @ {commit}"}})
    root = Node(f"{tree_id}/root", tree_id, None, 0, root_source)
    records = [tree, root, Measure(root.id, contract, "director", True, baseline.primary_score, baseline.evidence)]
    names, ancestry = {f"{tree_id}/root": root.id}, defaultdict(list)
    for position, step in enumerate(steps, 1):
        parent = names[step.parent]
        expansion_id = f"{tree_id}/{step.request_id}"
        direction = (((step.generation_result or {}).get("plan") or {}).get("directions") or [{}])[0]
        reads = read_uses(step.events_dir, view_root)
        planner = [use for use in reads if use["phase"] in PLANNER_PHASES]
        engineer = [use for use in reads if use["phase"] not in PLANNER_PHASES]
        pointed = view_uses(step.generation_result, view_root) if step.result else []
        planner_uses, planner_gaps = _uses(expansion_id, planner, knowledge)
        node_id = f"{expansion_id}/0"
        node_uses, node_gaps = _uses(node_id, [*({**use, "evidence": step.result} for use in pointed), *engineer],
                                     knowledge)
        mode = "deep" if direction.get("specialty") == "deep_explore" else "normal"
        records.append(Expansion(
            expansion_id, tree_id, "generate", (parent,),
            Plan(position, ContextRef(tuple(ancestry[parent]), (step.request,)), generation,
                 (Slot(str(direction.get("id") or "d0"), str(direction.get("specialty") or ""), mode),), 1),
            planner_uses))
        records += planner_gaps
        records.append(Node(node_id, tree_id, expansion_id, 0, step.source, node_uses,
                            {"geak": {"attempt": step.request_id}}))
        records += node_gaps
        if step.measurement.status == "qualified" and step.source:
            records.append(Measure(node_id, contract, "verifier", True, step.measurement.primary_score,
                                   step.measurement.evidence))
        elif step.measurement.status == "failed" and step.source:
            records.append(Decide(tree_id, node_id, "incorrect", step.measurement.reason or "measured as failed"))
        names[f"{tree_id}/{step.request_id}"] = node_id
        ancestry[node_id] = [*ancestry[parent], parent]
    if stop:
        records.append(Decide(tree_id, None, "stop", stop))
    return records


def _uses(subject, items, knowledge):
    uses, gaps, seen = [], [], set()
    for item in items:
        target = knowledge.get(item["kernelwiki"]) if item["kernelwiki"] else None
        key = (item["role"], target or item["view_path"])
        if key in seen:
            continue
        seen.add(key)
        evidence = item.get("evidence") or reference(item["events"], start_line=item["line"], end_line=item["line"])
        if target:
            uses.append(Use(target, item["role"], (evidence,)))
        else:
            gaps.append(GapRecord(subject, Gap(f"{subject}#{item['role']}:{item['view_path']}", "needs_online_control",
                                               UNREGISTERED, (item["role"], item["view_path"], item["commit"]))))
    return tuple(uses), gaps

