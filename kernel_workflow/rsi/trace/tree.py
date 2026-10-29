"""Fold append-only records into trees; grades, outcomes and views are derived, never stored."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from kernelwiki.knowledge import KnowledgeError

from ..discovery import Measurement, fingerprint
from .records import (
    CODE,
    CODE_MEASURERS,
    CODE_OPS,
    CODE_VERDICTS,
    EXTERNAL,
    GATES,
    KNOWLEDGE_OPS,
    KNOWLEDGE_VERDICTS,
    READ_ROLES,
    SOURCE,
    Decide,
    Expansion,
    GapRecord,
    Measure,
    Node,
    Tree,
    from_wire,
    is_knowledge,
)

FAILED_VERDICTS = {"incorrect", "failed"}
UNREGISTERED = "knowledge read outside KernelWiki"


@dataclass
class TreeView:
    tree: Tree
    root: Node | None = None
    expansions: list[Expansion] = field(default_factory=list)
    nodes: dict[str, Node] = field(default_factory=dict)
    measures: dict[str, list[Measure]] = field(default_factory=lambda: defaultdict(list))
    decisions: list[Decide] = field(default_factory=list)
    gaps: list[GapRecord] = field(default_factory=list)

    @property
    def id(self):
        return self.tree.id

    def expansion(self, node):
        return next((item for item in self.expansions if item.id == node.expansion), None)

    def children(self, node_id):
        """Recorded children of a node: expansions in record order, slots in dispatch order."""
        ordered = []
        for expansion in self.expansions:
            if node_id in expansion.parents:
                ordered.extend(sorted((node for node in self.nodes.values() if node.expansion == expansion.id),
                                      key=lambda node: node.slot))
        return ordered

    def ancestors(self, node_ids):
        found, pending = set(), list(node_ids)
        while pending:
            node = self.nodes[pending.pop()]
            if node.id in found:
                continue
            found.add(node.id)
            if node.expansion:
                pending.extend(self.expansion(node).parents)
        return found

    def outcome(self, node_id):
        """Lower median of independent measures under the tree contract; a failure verdict wins over none."""
        node = self.nodes[node_id]
        if any(item.node == node_id and item.verdict in FAILED_VERDICTS for item in self.decisions):
            return Measurement("failed", node.state, reason="decided incorrect or failed")
        scores = sorted((item for item in self.measures[node_id] if item.independent and item.score is not None),
                        key=lambda item: item.score)
        if not scores or node.state is None:
            return Measurement("unknown", node.state, reason="no independent measure under the tree contract")
        chosen = scores[(len(scores) - 1) // 2]
        return Measurement("qualified", node.state, chosen.score, chosen.contract, chosen.evidence)

    def grade(self, node_id):
        """E/A/T from design section 2.5, with the reasons that block the next grade."""
        return (_code_grade if self.tree.subject == CODE else _knowledge_grade)(self, self.nodes[node_id])


def _code_grade(view, node):
    expansion = view.expansion(node) if node.expansion else None
    if node.state is None or (expansion and any(view.nodes[parent].state is None for parent in expansion.parents)):
        return "T", ("parent or produced source state not observed",)
    reasons = []
    if not any(item.independent for item in view.measures[node.id]):
        reasons.append("no independent measure under the tree contract")
    if expansion:
        context = expansion.plan.context
        if not set(context.informed_by) <= view.ancestors(expansion.parents):
            reasons.append("context informed by non-ancestor nodes")
        if not context.evidence or any(item.kind != "observed" for item in context.evidence):
            reasons.append("context evidence not observed")
    return ("A", tuple(reasons)) if reasons else ("E", ())


def _knowledge_grade(view, node):
    if node.state is None:
        return "T", ("knowledge version not observed",)
    expansion = view.expansion(node)
    sources = [use for use in (*node.uses, *expansion.uses) if use.role == SOURCE]
    if not sources:
        return "T", ("no machine-readable source",)
    reasons = []
    if any(item.kind != "observed" for use in sources for item in use.evidence):
        reasons.append("a source is only claimed")
    if not any(item.independent for item in view.measures[node.id]):
        gate = view.tree.params.get("gate")
        reasons.append(f"gate result not recorded (rerunnable: {gate})" if gate else
                       "no gate defined for this knowledge kind")
    return ("A", tuple(reasons)) if reasons else ("E", ())


@dataclass
class Forest:
    trees: dict[str, TreeView] = field(default_factory=dict)

    def code_trees(self):
        """The replayable optimization trees; knowledge and external trees are their evidence."""
        return tuple(view for view in self.trees.values() if view.tree.subject == CODE)


def _check_state(subject, value):
    """Identities are reused, never invented: a content SHA-256, or the commit an external node pins."""
    if subject != EXTERNAL:
        fingerprint(value)
    elif not re.fullmatch(r"[0-9a-f]{7,40}", value):
        raise KnowledgeError("an external node state is the commit it pins")


def fold(records):
    """Apply records in order; every reference must point to an earlier record (design invariant 1)."""
    forest, owner, expansions = Forest(), {}, {}
    for record in records:
        record = record if not isinstance(record, dict) else from_wire(record)
        if isinstance(record, Tree):
            if record.id in forest.trees:
                raise KnowledgeError(f"duplicate tree: {record.id}")
            if record.parent is not None:
                mount = expansions.get(record.parent["expansion"])
                if (record.parent["tree"] not in forest.trees or mount is None or mount.tree != record.parent["tree"]
                        or mount.op != "delegate" or record.parent["slot"] >= mount.dispatched):
                    raise KnowledgeError("a mounted tree must hang on an earlier delegate slot")
            forest.trees[record.id] = TreeView(record)
        elif isinstance(record, Expansion):
            view = _tree(forest, record.tree)
            subject = view.tree.subject
            if record.id in expansions:
                raise KnowledgeError(f"duplicate expansion: {record.id}")
            if (record.op in CODE_OPS) != (subject == CODE) or (record.op in KNOWLEDGE_OPS) != is_knowledge(subject):
                raise KnowledgeError(f"op {record.op} does not belong to a {subject} tree")
            if any(owner.get(node) != record.tree for node in record.parents):
                raise KnowledgeError("expansion parents must be earlier nodes of the same tree")
            if record.plan and any(node not in owner for node in record.plan.context.informed_by):
                raise KnowledgeError("context can only be informed by earlier nodes")
            _check_uses(forest, owner, subject, record.uses)
            expansions[record.id] = record
            view.expansions.append(record)
        elif isinstance(record, Node):
            view = _tree(forest, record.tree)
            subject = view.tree.subject
            if record.id in owner:
                raise KnowledgeError(f"duplicate node: {record.id}")
            if record.state is not None:
                _check_state(subject, record.state)
            if record.expansion is None:
                if subject == CODE and view.root is not None:
                    raise KnowledgeError("a tree has exactly one root")
                if is_knowledge(subject):
                    raise KnowledgeError("every knowledge version is produced by an expansion")
                view.root = view.root or record
            else:
                expansion = expansions.get(record.expansion)
                if subject == EXTERNAL:
                    raise KnowledgeError("an external node is observed, never produced here")
                if expansion is None or expansion.tree != record.tree or record.slot >= expansion.dispatched:
                    raise KnowledgeError("node must fill a dispatched slot of an earlier expansion in its tree")
                if any(node.expansion == record.expansion and node.slot == record.slot for node in view.nodes.values()):
                    raise KnowledgeError("an expansion slot is filled once")
            _check_uses(forest, owner, subject, record.uses)
            owner[record.id] = record.tree
            view.nodes[record.id] = record
        elif isinstance(record, Measure):
            view = _tree(forest, owner.get(record.node))
            if record.contract != view.tree.contract:
                raise KnowledgeError("measure contract must equal its tree contract")
            if (record.by in CODE_MEASURERS) != (view.tree.subject == CODE) or (
                    record.by in GATES) != is_knowledge(view.tree.subject):
                raise KnowledgeError(f"measurer {record.by} does not belong to a {view.tree.subject} tree")
            if view.nodes[record.node].state is None:
                raise KnowledgeError("only an observed state can be measured")
            view.measures[record.node].append(record)
        elif isinstance(record, Decide):
            view = _tree(forest, record.tree)
            if record.node is not None and owner.get(record.node) != record.tree:
                raise KnowledgeError("decision must name an earlier node of its tree")
            if (record.verdict in CODE_VERDICTS) != (view.tree.subject == CODE) or (
                    record.verdict in KNOWLEDGE_VERDICTS) != is_knowledge(view.tree.subject):
                raise KnowledgeError(f"verdict {record.verdict} does not belong to a {view.tree.subject} tree")
            if record.verdict == "committed":
                _check_commit(view, record)
            view.decisions.append(record)
        elif isinstance(record, GapRecord):
            tree = owner.get(record.subject) or (expansions[record.subject].tree if record.subject in expansions
                                                   else record.subject)
            _tree(forest, tree).gaps.append(record)
        else:
            raise KnowledgeError(f"unknown trace record: {type(record).__name__}")
    return forest


def _tree(forest, tree_id):
    if tree_id not in forest.trees:
        raise KnowledgeError(f"record refers to an unknown or later tree: {tree_id}")
    return forest.trees[tree_id]


def _check_uses(forest, owner, subject, uses):
    """Design invariant 9: code reads only KernelWiki knowledge; knowledge cites earlier nodes of any tree."""
    for use in uses:
        if use.target not in owner:
            raise KnowledgeError(f"use must target an earlier node: {use.target}")
        target = forest.trees[owner[use.target]].tree.subject
        if subject == CODE and (use.role not in READ_ROLES or not is_knowledge(target)):
            raise KnowledgeError("a code record only reads KernelWiki knowledge nodes; anything else is a gap")
        if subject != CODE and use.role != SOURCE:
            raise KnowledgeError("a knowledge or external record only cites its sources")


def _check_commit(view, decision):
    """Committed nodes form one chain from the root (design invariant 4)."""
    committed = [item.node for item in view.decisions if item.verdict == "committed"]
    previous = committed[-1] if committed else view.root.id
    node = view.nodes[decision.node]
    if node.expansion is None or previous not in view.expansion(node).parents:
        raise KnowledgeError("a commit must continue the previous commit or the root")
    if decision.threshold is not None:
        before, after = view.outcome(previous), view.outcome(node.id)
        if (before.status == after.status == "qualified"
                and after.primary_score / before.primary_score - 1 < decision.threshold):
            raise KnowledgeError("a commit must meet its recorded threshold")


@dataclass(frozen=True)
class ReplayNode:
    """What the shared Rollout needs from a recorded tree transition."""
    id: str
    attempt_id: str | None
    parent_ids: tuple[str, ...]
    measurement: Measurement

    def continues(self, source):
        return source.id in self.parent_ids


def replay_node(view, node_id):
    node = view.nodes[node_id]
    parents = view.expansion(node).parents if node.expansion else ()
    attempt = node.facets.get("geak", {}).get("attempt") if node.expansion else None
    return ReplayNode(node.id, attempt or (node.id if node.expansion else None), parents, view.outcome(node_id))


def knowledge_ledger(forest):
    """One row per KernelWiki node: where it came from, how it evolved, who read it and with what result."""
    rows = {}
    for view in forest.trees.values():
        if not is_knowledge(view.tree.subject):
            continue
        for node in view.nodes.values():
            expansion = view.expansion(node)
            rows[node.id] = {
                "op": expansion.op, "sources": sorted({use.target for use in (*node.uses, *expansion.uses)}),
                "before": list(expansion.parents), "after": [child.id for child in view.children(node.id)],
                "verdicts": [item.verdict for item in view.decisions if item.node == node.id],
                "roles": defaultdict(int), "contrasts": []}
    for view in forest.trees.values():
        if view.tree.subject != CODE:
            continue
        for expansion in view.expansions:
            for use in expansion.uses:
                rows[use.target]["roles"]["planner:" + use.role] += 1
            siblings = [node for node in view.nodes.values() if node.expansion == expansion.id]
            outcomes = {node.id: view.outcome(node.id) for node in siblings}
            for node in siblings:
                for use in node.uses:
                    rows[use.target]["roles"][use.role] += 1
            for target in {use.target for node in siblings for use in node.uses}:
                users = [node.id for node in siblings if target in {use.target for use in node.uses}]
                others = [node.id for node in siblings if node.id not in users]
                scored = lambda ids: [outcomes[item].primary_score for item in ids
                                      if outcomes[item].status == "qualified"]
                with_score, without_score = scored(users), scored(others)
                if with_score and without_score:
                    rows[target]["contrasts"].append({
                        "expansion": expansion.id, "with": users, "without": others,
                        "gain": (sum(with_score) / len(with_score)) / (sum(without_score) / len(without_score)) - 1})
    return {key: {**value, "roles": dict(value["roles"])} for key, value in sorted(rows.items())}


def knowledge_coverage(forest):
    """P1: share of knowledge reads in code trees that resolve to a KernelWiki node."""
    resolved = sum(1 for view in forest.trees.values() if view.tree.subject == CODE
                   for record in (*view.expansions, *view.nodes.values()) for _ in record.uses)
    unregistered = sum(1 for view in forest.trees.values() for item in view.gaps
                       if item.gap.question == UNREGISTERED)
    total = resolved + unregistered
    return {"resolved": resolved, "unregistered": unregistered, "share": resolved / total if total else None}


def gap_worklist(forest):
    """Every non-E produced node with what blocks it, A before T, plus recorded gaps."""
    rows = []
    for view in forest.trees.values():
        recorded = defaultdict(list)
        for item in view.gaps:
            recorded[item.subject].append(item.gap.to_dict())
        for node in view.nodes.values():
            if node.expansion is None:
                continue
            grade, reasons = view.grade(node.id)
            if grade != "E":
                rows.append({"node": node.id, "grade": grade, "blocked_by": list(reasons),
                             "gaps": recorded.get(node.id, [])})
        rows.extend({"subject": subject, "gaps": items} for subject, items in recorded.items()
                    if subject not in view.nodes)
    return sorted(rows, key=lambda row: (row.get("grade", "Z"), row.get("node") or row["subject"]))
