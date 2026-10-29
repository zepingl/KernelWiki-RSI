"""Observed sources, contexts, measurements, attempts and discovery trees."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field

from kernelwiki.knowledge import (
    EvidenceRef,
    KernelState,
    KnowledgeError,
    finite,
    strict_fields,
)

SCHEMA_VERSION = 2


def fingerprint(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise KnowledgeError("state identity must be a complete SHA-256")
    return value


@dataclass(frozen=True)
class SourceState:
    fingerprint: str
    evidence: tuple[EvidenceRef, ...]

    def __post_init__(self):
        fingerprint(self.fingerprint)
        if not self.evidence or any(item.kind != "observed" for item in self.evidence):
            raise KnowledgeError("source state requires observed snapshots independently of evaluation")

    @classmethod
    def from_dict(cls, value):
        strict_fields(value, cls.__dataclass_fields__, cls.__dataclass_fields__)
        return cls(value["fingerprint"], tuple(EvidenceRef.from_dict(item) for item in value["evidence"]))


@dataclass(frozen=True)
class ContextSnapshot:
    revision: str
    evidence: tuple[EvidenceRef, ...]
    parent_revision: str | None = None

    def __post_init__(self):
        fingerprint(self.revision)
        if self.parent_revision is not None:
            fingerprint(self.parent_revision)
        if not self.evidence or any(item.kind != "observed" for item in self.evidence):
            raise KnowledgeError("context requires observed input and continuation records")

    @classmethod
    def from_dict(cls, value):
        strict_fields(value, cls.__dataclass_fields__, {"revision", "evidence"})
        return cls(value["revision"], tuple(EvidenceRef.from_dict(item) for item in value["evidence"]),
                   value.get("parent_revision"))


@dataclass(frozen=True)
class GenerationSpec:
    mode: str
    model: str
    effort: str
    context_revision: str
    evaluator_revision: str
    worker_limit: int = 2

    def __post_init__(self):
        if self.mode not in {"normal", "deep_explore"}:
            raise KnowledgeError("generation mode must be frozen per world")
        if any(not isinstance(value, str) or not value for key, value in asdict(self).items() if key != "worker_limit"):
            raise KnowledgeError("generation spec must pin agent, context, and evaluator")
        if type(self.worker_limit) is not int or self.worker_limit not in (1, 2):
            raise KnowledgeError("generation must pin one or two workers")

    @property
    def cost(self):
        return 2 if self.mode == "deep_explore" else 1


@dataclass(frozen=True)
class Measurement:
    status: str
    source: str | None = None
    primary_score: float | None = None
    contract: str | None = None
    evidence: tuple[EvidenceRef, ...] = ()
    reason: str = ""

    def __post_init__(self):
        if self.status not in {"qualified", "failed", "unknown"}:
            raise KnowledgeError("invalid measurement status")
        if self.primary_score is not None and finite(self.primary_score) <= 0:
            raise KnowledgeError("primary score must be positive")
        if self.status == "qualified":
            if (self.primary_score is None or not self.contract or not self.source
                    or not re.fullmatch(r"[0-9a-f]{64}", self.source) or not self.evidence
                    or any(item.kind != "observed" for item in self.evidence)):
                raise KnowledgeError("qualified measurement needs source, contract, and observed receipts")
        elif self.primary_score is not None:
            raise KnowledgeError("unqualified measurement cannot supply replay quality")

    @classmethod
    def from_dict(cls, value):
        strict_fields(value, cls.__dataclass_fields__, {"status"})
        return cls(**{**value, "evidence": tuple(EvidenceRef.from_dict(item)
                                               for item in value.get("evidence", []))})


@dataclass(frozen=True)
class DiscoveryNode:
    id: str
    parent_id: str | None
    attempt_id: str | None
    measurement: Measurement
    input_evidence: tuple[EvidenceRef, ...]
    state: SourceState = field(kw_only=True)
    context: ContextSnapshot = field(kw_only=True)
    input_state: str | None = field(default=None, kw_only=True)
    input_context: str | None = field(default=None, kw_only=True)
    candidate_source: str | None = field(default=None, kw_only=True)

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id or not self.input_evidence:
            raise KnowledgeError("node needs an identity and verified input evidence")
        if any(item.kind != "observed" for item in self.input_evidence):
            raise KnowledgeError("node input must be observed, not claimed")
        if self.parent_id is not None and not self.attempt_id:
            raise KnowledgeError("a generated node must identify the external attempt")
        if not isinstance(self.state, SourceState) or not isinstance(self.context, ContextSnapshot):
            raise KnowledgeError("node must own its source and context independently of measurement")
        if self.parent_id is not None:
            fingerprint(self.input_state)
            fingerprint(self.input_context)
            if self.context.parent_revision != self.input_context:
                raise KnowledgeError("continuation context does not inherit its input")
        elif self.input_state is not None or self.input_context is not None or self.context.parent_revision is not None:
            raise KnowledgeError("root cannot inherit an undeclared parent")
        if self.candidate_source is not None:
            fingerprint(self.candidate_source)
        if self.measurement.status == "qualified" and self.measurement.source != self.state.fingerprint:
            raise KnowledgeError("measurement does not belong to the retained source state")

    def continues(self, source):
        return (isinstance(source, DiscoveryNode) and self.parent_id == source.id
                and self.input_state == source.state.fingerprint and self.input_context == source.context.revision)

    @classmethod
    def from_dict(cls, value):
        strict_fields(value, cls.__dataclass_fields__, cls.__dataclass_fields__)
        return cls(value["id"], value["parent_id"], value["attempt_id"],
                   Measurement.from_dict(value["measurement"]),
                   tuple(EvidenceRef.from_dict(item) for item in value["input_evidence"]),
                   state=SourceState.from_dict(value["state"]), context=ContextSnapshot.from_dict(value["context"]),
                   input_state=value["input_state"], input_context=value["input_context"],
                   candidate_source=value["candidate_source"])


@dataclass(frozen=True)
class DiscoveryWorld:
    id: str
    kernel: str
    campaign: str
    state: KernelState
    generation: GenerationSpec
    root: DiscoveryNode
    branches: tuple[tuple[DiscoveryNode, ...], ...]

    def __post_init__(self):
        if not self.id or not self.kernel or not self.campaign:
            raise KnowledgeError("world needs identity, kernel, and campaign")
        if self.root.parent_id is not None or self.root.measurement.status != "qualified":
            raise KnowledgeError("world root needs a verified baseline")
        ids = {self.root.id}
        attempts = set()
        if not isinstance(self.branches, tuple) or not self.branches:
            raise KnowledgeError("world needs recorded branches")
        for branch in self.branches:
            if not isinstance(branch, tuple) or not branch:
                raise KnowledgeError("empty or mutable branch")
            parent = self.root
            for node in branch:
                if node.id in ids or node.attempt_id in attempts or node.parent_id != parent.id:
                    raise KnowledgeError("world is not a root with independent continuation chains")
                if node.input_state != parent.state.fingerprint or node.input_context != parent.context.revision:
                    raise KnowledgeError("child input does not match its parent's source and context")
                if (node.measurement.status == "qualified"
                        and node.measurement.contract != self.root.measurement.contract):
                    raise KnowledgeError("world measurements use different scoring contracts")
                ids.add(node.id)
                attempts.add(node.attempt_id)
                parent = node

    @classmethod
    def from_dict(cls, value):
        strict_fields(value, {*cls.__dataclass_fields__, "schema_version"}, cls.__dataclass_fields__)
        if value.get("schema_version") != SCHEMA_VERSION:
            raise KnowledgeError("legacy discovery schema requires reconstruction from original evidence")
        return cls(value["id"], value["kernel"], value["campaign"],
                   KernelState.from_dict(value["state"]), GenerationSpec(**value["generation"]),
                   DiscoveryNode.from_dict(value["root"]),
                   tuple(tuple(DiscoveryNode.from_dict(node) for node in branch)
                         for branch in value["branches"]))

    def to_dict(self):
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}



@dataclass(frozen=True)
class Attempt:
    id: str
    kernel: str
    campaign: str
    invocation: EvidenceRef
    input: EvidenceRef | None
    parent_source: str | None
    candidate_source: str | None
    measurement: Measurement
    disposition: str
    gaps: tuple[str, ...]
    audit: tuple[EvidenceRef, ...] = ()
    state: SourceState | None = None
    context: ContextSnapshot | None = None
    input_context: str | None = None
    parent_attempt_id: str | None = None

    def __post_init__(self):
        if self.disposition not in {"replay_ready", "trace_only"}:
            raise KnowledgeError("attempt disposition must be replay_ready or trace_only")
        if self.disposition == "replay_ready" and (self.gaps or not self.input or not self.parent_source
                                                  or not self.state or not self.context or not self.input_context):
            raise KnowledgeError("replay-ready attempt has unresolved input provenance")

    @classmethod
    def from_dict(cls, value):
        strict_fields(value, cls.__dataclass_fields__, set(cls.__dataclass_fields__) -
                      {"audit", "state", "context", "input_context", "parent_attempt_id"})
        return cls(**{**value, "invocation": EvidenceRef.from_dict(value["invocation"]),
                      "input": EvidenceRef.from_dict(value["input"]) if value["input"] else None,
                      "measurement": Measurement.from_dict(value["measurement"]),
                      "gaps": tuple(value["gaps"]),
                      "audit": tuple(EvidenceRef.from_dict(item) for item in value.get("audit", [])),
                      "state": SourceState.from_dict(value["state"]) if value.get("state") else None,
                      "context": ContextSnapshot.from_dict(value["context"]) if value.get("context") else None})


def reconstruct(world_id, kernel, campaign, state, generation, root, attempts):
    attempts = tuple(attempts)
    children = defaultdict(list)
    for attempt in attempts:
        if attempt.disposition != "replay_ready":
            raise KnowledgeError("trace-only attempts cannot become replay transitions")
        children[attempt.parent_attempt_id].append(attempt)
    branches, visited = [], set()
    for first in children[None]:
        branch, parent, attempt = [], root, first
        while attempt is not None:
            if attempt.id in visited:
                raise KnowledgeError("cyclic or repeated attempt in discovery history")
            visited.add(attempt.id)
            node = DiscoveryNode(attempt.id, parent.id, attempt.id, attempt.measurement,
                                 tuple(dict.fromkeys((attempt.invocation, attempt.input, *attempt.audit))),
                                 state=attempt.state, context=attempt.context,
                                 input_state=attempt.parent_source, input_context=attempt.input_context,
                                 candidate_source=attempt.candidate_source)
            branch.append(node)
            continuations = children[attempt.id]
            if len(continuations) > 1:
                raise KnowledgeError("multiple historical continuations require independent roots")
            parent, attempt = node, continuations[0] if continuations else None
        branches.append(tuple(branch))
    if len(visited) != len(attempts):
        raise KnowledgeError("continuation references an unsupported parent")
    return DiscoveryWorld(world_id, kernel, campaign, state, generation, root, tuple(branches))
