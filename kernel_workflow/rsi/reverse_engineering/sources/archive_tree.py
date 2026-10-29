"""Rebuild discovery trees from an original GEAK audit archive as append-only trace records.

One tree per campaign x kernel: the campaign relaunches the lane from its best source, so a launch
continues the node whose source it starts on. A plan_round call is an expansion; the engineer calls its
directions reached are the nodes. Commits are inferred from continuation; integrate and adopt are not
reconstructed yet and surface as gaps. A knowledge read becomes a use only when its content resolves to a
KernelWiki node of the given index; any other read is recorded as an unregistered-knowledge gap.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from kernelwiki.knowledge import KernelState, KnowledgeError

from ...evidence import read_json, read_jsonl
from ...trace.gaps import Gap
from ...trace.records import CODE, ContextRef, Decide, Expansion, GapRecord, Generation, Measure, Node, Plan, Slot, Tree, Use
from ...trace.tree import UNREGISTERED

KNOWLEDGE_ROOT = "perf_knowledge/"


def knowledge_id(ref):
    return ref.split(KNOWLEDGE_ROOT, 1)[-1]


def phase(call):
    return str(call.get("context", {}).get("phase", "")).lower()


def launch(call):
    return call.get("path", "").split("/agents/")[0]


def gap(subject, question, reasons, classification="resolvable_local"):
    return GapRecord(subject, Gap(f"{subject}#{len(reasons)}:{question}", classification, question, tuple(reasons)))


@dataclass
class Lineage:
    """Mutable cursor of one campaign x kernel tree while its launches are replayed in order."""
    tree: Tree
    latest: dict
    committed: list
    memory: list = field(default_factory=list)
    rounds: int = 0
    generation: Generation | None = None


class LaneTrees:
    def __init__(self, reader, attempts, *, campaign, tasks, states=None, knowledge=None):
        self.reader, self.campaign, self.tasks, self.states = reader, campaign, tasks, states or {}
        self.knowledge = knowledge or {}
        self.attempts = {attempt.id: attempt for attempt in attempts}
        self.parents_of = {}
        lineage = reader.run / "index/knowledge_lineage.jsonl"
        self.reads, self.origin = defaultdict(list), {}
        for number, row in enumerate(read_jsonl(lineage) if lineage.exists() else (), 1):
            if row.get("kind") == "direct_read" and row.get("card"):
                self.reads[row["call_id"]].append((knowledge_id(row["card"]["ref"]), row["card"].get("snapshot_sha256"),
                                                   reader.ref(lineage, start_line=number, end_line=number)))
            elif row.get("kind") == "direction_transfer":
                candidates = row.get("direction_transfer", {}).get("origin_candidates") or []
                if len(candidates) == 1:
                    self.origin[row["call_id"]] = candidates[0]

    def records(self):
        kernels = defaultdict(lambda: defaultdict(list))
        for call in self.reader.calls.values():
            if call.get("task") in self.tasks and "/agents/" in call.get("path", ""):
                kernels[self.tasks[call["task"]]][launch(call)].append(call)
        records = []
        for kernel, launches in sorted(kernels.items()):
            records.extend(self.kernel_records(kernel, launches))
        return records

    def kernel_records(self, kernel, launches):
        """One tree per campaign x kernel: a launch continues from the node whose source it starts on."""
        records, lineage = [], None
        for name, calls in sorted(launches.items()):
            calls = sorted(calls, key=lambda call: call["call_id"])
            plans = [call for call in calls if phase(call) == "plan_round"]
            engineers = defaultdict(list)
            for call in calls:
                origin = self.origin.get(call["call_id"])
                if phase(call) == "optimize" and origin:
                    engineers[origin["call_id"]].append((origin["direction_index"], call))
            dispatching = [plan for plan in plans if engineers.get(plan["call_id"])]
            if not dispatching:
                continue
            first = min(engineers[dispatching[0]["call_id"]], key=lambda item: item[0])[1]
            attempt = self.attempts.get(first["call_id"])
            if attempt is None or not attempt.parent_source:
                continue
            invocation = read_json(attempt.invocation.path)
            if lineage is None or attempt.parent_source not in lineage.latest:
                # A launch that starts from no reconstructed node (integrate, validate, setup output) opens a
                # new segment rooted at its measured baseline; no edge to the previous segment is invented.
                baseline = self.reader.baseline(attempt.parent_source, first["context"].get("CANONICAL"),
                                                invocation.get("time", first["call_id"]))
                if baseline is None:
                    continue
                tree_id = f"{self.campaign}/{kernel}" + (f"/{Path(name).name}" if lineage else "")
                tree = Tree(tree_id, CODE, baseline.contract, KernelState.from_dict(self.states.get(kernel, {})),
                            facets={"geak": {"campaign": self.campaign, "kernel": kernel}})
                root = Node(f"{tree.id}/root", tree.id, None, 0, attempt.parent_source, facets={"geak": {"launch": name}})
                records += [tree, root, Measure(root.id, tree.contract, "director", True, baseline.primary_score,
                                                baseline.evidence)]
                lineage = Lineage(tree, {root.state: root.id}, [root.id], lineage.memory if lineage else [])
            lineage.generation = Generation(invocation["argv"][invocation["argv"].index("--model") + 1],
                                            invocation["argv"][invocation["argv"].index("--effort") + 1])
            updates = [call["call_id"] for call in calls if phase(call) == "update_memory"]
            for plan in plans:
                plan_ref = self.reader.ref(Path(plan["path"]) / "delivered_result.txt")
                try:
                    result = self.plan(plan)
                except (OSError, ValueError, KeyError) as error:
                    records.append(gap(lineage.tree.id, "unreadable plan result", (plan["call_id"], str(error))))
                    continue
                dispatched = sorted(engineers.get(plan["call_id"], ()), key=lambda item: item[0])
                if dispatched:
                    records.extend(self.expansion(lineage, name, plan, plan_ref, result, dispatched, updates))
                if result.get("stop"):
                    records.append(Decide(lineage.tree.id, None, "stop", str(result.get("reasoning", ""))[:500]))
        return records

    def expansion(self, lineage, name, plan, plan_ref, result, dispatched, updates):
        tree = lineage.tree
        parents = {self.attempts[call["call_id"]].parent_source for _, call in dispatched
                   if call["call_id"] in self.attempts and self.attempts[call["call_id"]].parent_source}
        if len(parents) != 1 or next(iter(parents)) not in lineage.latest:
            return [gap(tree.id, "expansion parent matches no reconstructed node",
                        (plan["call_id"], "integrate and adopt outputs are not reconstructed yet"))]
        parent = lineage.latest[next(iter(parents))]
        records, first = [], {}
        for index, call in dispatched:
            if index in first:
                records.append(gap(tree.id, "direction dispatched more than once", (call["call_id"], first[index])))
            else:
                first[index] = call["call_id"]
        dispatched = [(index, call) for index, call in dispatched if first[index] == call["call_id"]]
        if parent != lineage.committed[-1] and lineage.committed[-1] in self.parents_of.get(parent, ()):
            records.append(Decide(tree.id, parent, "committed", "inferred: a later expansion continues this node"))
            lineage.committed.append(parent)
        # Earlier launches reach a plan through its HISTORY input; within a launch, through update_memory.
        informed = [node for node, after, where in lineage.memory
                    if where != name or any(after < update < plan["call_id"] for update in updates)]
        planned = result["directions"]
        lineage.rounds += 1
        expansion_id = f"{tree.id}/{plan['call_id']}"
        planner_uses, unregistered = self.uses(expansion_id, self.launch_reads(plan), "read")
        expansion = Expansion(
            expansion_id, tree.id, "generate", (parent,),
            Plan(lineage.rounds,
                 ContextRef(informed, (self.reader.ref(Path(plan["path"]) / "delivered_prompt.txt"),)),
                 lineage.generation,
                 tuple(Slot(str(item.get("id") or f"d{index}"), str(item.get("specialty") or ""),
                            "deep" if item.get("specialty") == "deep_explore" else "normal")
                       for index, item in enumerate(planned)),
                 dispatched[-1][0] + 1),
            planner_uses, {"geak": {"launch": name}})
        records += [expansion, *unregistered]
        for index, call in dispatched:
            attempt = self.attempts.get(call["call_id"])
            direction = planned[index] if index < len(planned) else {}
            state = attempt.state.fingerprint if attempt and attempt.state else None
            node_id = f"{expansion.id}/{index}"
            uses, unregistered = [], []
            for items, role in (([(knowledge_id(ref), self.snapshot(ref), plan_ref)
                                  for ref in direction.get("kk_refs") or []], "pointed"),
                                ([(str(ref), None, plan_ref) for ref in direction.get("learned_refs") or []], "cited"),
                                (self.reads[call["call_id"]], "read")):
                resolved, missing = self.uses(node_id, items, role)
                uses.extend(resolved)
                unregistered.extend(missing)
            node = Node(node_id, tree.id, expansion.id, index, state, tuple(uses), {"geak": {"attempt": call["call_id"]}})
            records += [node, *unregistered]
            self.parents_of[node.id] = expansion.parents
            lineage.memory.append((node.id, call["call_id"], name))
            if attempt is None or attempt.disposition != "replay_ready":
                records.append(gap(node.id, "attempt provenance incomplete in the original archive",
                                   attempt.gaps if attempt else ("no inventoried attempt",)))
            if state is None:
                continue
            lineage.latest[state] = node.id
            records.extend(self.outcome(tree, node, name))
        return records

    def plan(self, call):
        text = (self.reader.run / call["path"] / "delivered_result.txt").read_text()
        value = json.loads(text[text.index("{"): text.rindex("}") + 1])
        if not isinstance(value.get("directions"), list):
            raise KnowledgeError("plan result has no directions list")
        return value

    def launch_reads(self, plan):
        return [read for call in self.reader.calls.values()
                if launch(call) == launch(plan) and call["call_id"] <= plan["call_id"]
                and str(call.get("context", {}).get("role", "")) == "tech_lead"
                for read in self.reads[call["call_id"]]]

    def snapshot(self, ref):
        path = self.reader.run / "snapshots/GEAK" / KNOWLEDGE_ROOT / knowledge_id(ref)
        return self.reader.ref(path).sha256 if path.is_file() else None

    def uses(self, subject, items, role):
        """Reads whose content is a KernelWiki node become uses; the rest become unregistered-knowledge gaps."""
        seen, uses, missing = set(), [], []
        for record, revision, evidence in items:
            if (record, revision) in seen:
                continue
            seen.add((record, revision))
            target = self.knowledge.get(revision) if revision else None
            if target:
                uses.append(Use(target, role, (evidence,)))
            else:
                missing.append(GapRecord(subject, Gap(f"{subject}#{role}:{record}@{revision}", "needs_online_control",
                                                      UNREGISTERED, (role, record, revision or "no revision"))))
        return tuple(uses), missing

    def outcome(self, tree, node, launch_name):
        """Runs of the node's source inside its own launch; other launches measure other occurrences.
        Only-failed runs decide incorrect; failed and qualified runs together stay unscored as a gap."""
        measures, failed = [], False
        for row in self.reader.benchmarks[node.state]:
            if not str(row.get("summary_path", "")).startswith(launch_name + "/"):
                continue
            try:
                measured = self.reader.measurement(row)
            except OSError:
                continue
            failed = failed or measured.status == "failed"
            if measured.status == "qualified" and measured.contract == tree.contract:
                call = self.reader.calls.get(row.get("call_id"), {})
                by = {"verify": "verifier", "optimize": "engineer"}.get(phase(call), "director")
                measures.append(Measure(node.id, tree.contract, by, True, measured.primary_score, measured.evidence))
        if failed and measures:
            return [gap(node.id, "conflicting correctness evidence", ("qualified and failed runs of one source",))]
        if failed:
            return [Decide(tree.id, node.id, "incorrect", "correctness failed in every process")]
        return measures


def lane_trees(reader, attempts, *, campaign, tasks, states=None, knowledge=None):
    return LaneTrees(reader, attempts, campaign=campaign, tasks=tasks, states=states, knowledge=knowledge).records()
