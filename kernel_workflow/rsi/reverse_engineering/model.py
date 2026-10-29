"""Typed provenance relations; reviews and claims remain separate from observations."""

import json
from dataclasses import asdict, dataclass

from kernelwiki.knowledge import EvidenceRef, KnowledgeError, canonical, strict_fields

from ..discovery import Attempt, DiscoveryNode
from ..trace.gaps import Coverage, Gap


@dataclass(frozen=True)
class Relation:
    source: str
    target: str
    kind: str
    evidence: tuple[EvidenceRef, ...]
    certainty: str = "observed"

    def __post_init__(self):
        object.__setattr__(self, "evidence", tuple(self.evidence))
        if (any(not isinstance(value, str) or not value for value in (self.source, self.target))
                or any(not isinstance(item, EvidenceRef) for item in self.evidence)):
            raise KnowledgeError("relations need identities and typed evidence")
        if self.kind not in {"parent", "measured", "installed", "body_delivered",
                              "direction_forwarded", "implementation_inherited", "method_adopted", "method_reviewed"}:
            raise KnowledgeError("unknown evidence relation")
        if self.certainty not in {"observed", "claim", "review", "unknown"}:
            raise KnowledgeError("unknown relation certainty")
        if self.certainty != "unknown" and not self.evidence:
            raise KnowledgeError("a positive relation needs evidence")
        if self.certainty == "observed" and any(item.kind != "observed" for item in self.evidence):
            raise KnowledgeError("claims and reviews cannot prove an observation")

    @classmethod
    def from_dict(cls, value):
        strict_fields(value, cls.__dataclass_fields__, cls.__dataclass_fields__)
        return cls(value["source"], value["target"], value["kind"],
                   tuple(EvidenceRef.from_dict(item) for item in value["evidence"]), value["certainty"])


@dataclass(frozen=True)
class TraceSubject:
    attempt: Attempt
    parent: DiscoveryNode | None = None
    sources: tuple[EvidenceRef, ...] = ()

    def __post_init__(self):
        if not isinstance(self.attempt, Attempt) or (self.parent is not None and not isinstance(self.parent, DiscoveryNode)):
            raise KnowledgeError("trace subjects need an original attempt and optional direct parent")
        if self.parent is not None:
            if self.attempt.parent_source != self.parent.state.fingerprint:
                raise KnowledgeError("trace parent does not own the dispatched source")
            if self.attempt.input_context and self.attempt.input_context != self.parent.context.revision:
                raise KnowledgeError("trace parent does not own the dispatched context")
            if self.attempt.parent_attempt_id and self.attempt.parent_attempt_id != self.parent.attempt_id:
                raise KnowledgeError("trace parent is not the direct historical parent")
        refs = [self.attempt.invocation, *self.sources, *self.attempt.audit, *self.attempt.measurement.evidence]
        if self.attempt.input:
            refs.append(self.attempt.input)
        for value in (self.attempt.state, self.attempt.context):
            if value:
                refs.extend(value.evidence)
        if self.parent:
            refs.extend((*self.parent.input_evidence, *self.parent.state.evidence,
                         *self.parent.context.evidence, *self.parent.measurement.evidence))
        if any(not isinstance(item, EvidenceRef) or item.kind != "observed" for item in refs):
            raise KnowledgeError("trace subjects must originate from observations")
        object.__setattr__(self, "sources", tuple(dict.fromkeys(refs)))

    def to_dict(self):
        return json.loads(canonical({"attempt": asdict(self.attempt), "parent": asdict(self.parent) if self.parent else None,
                                    "sources": [asdict(item) for item in self.sources]}))

    @classmethod
    def from_dict(cls, value):
        strict_fields(value, {"attempt", "parent", "sources"}, {"attempt", "parent", "sources"})
        return cls(Attempt.from_dict(value["attempt"]),
                   DiscoveryNode.from_dict(value["parent"]) if value["parent"] else None,
                   tuple(EvidenceRef.from_dict(item) for item in value["sources"]))


@dataclass(frozen=True, init=False)
class ReverseReport:
    subject: TraceSubject
    relations: tuple[Relation, ...]
    gaps: tuple[Gap, ...]
    _findings: bytes

    def __init__(self, subject, findings, relations=(), gaps=()):
        if not isinstance(subject, TraceSubject) or not isinstance(findings, dict):
            raise KnowledgeError("reverse reports need a typed subject and structured findings")
        if {"attempt", "relations", "gaps", "report_version", "subject", "sources"} & findings.keys():
            raise KnowledgeError("reverse findings cannot replace authoritative provenance")
        if not isinstance(findings.get("original_inputs"), dict) or not isinstance(findings.get("effect"), dict):
            raise KnowledgeError("reverse findings need original inputs and a separate effect summary")
        relations, gaps = tuple(relations), tuple(gaps)
        if any(not isinstance(item, Relation) for item in relations) or any(not isinstance(item, Gap) for item in gaps):
            raise KnowledgeError("reverse reports require typed relations and gaps")
        if len({item.id for item in gaps}) != len(gaps):
            raise KnowledgeError("reverse reports cannot repeat a gap")
        object.__setattr__(self, "subject", subject)
        object.__setattr__(self, "relations", relations)
        object.__setattr__(self, "gaps", gaps)
        object.__setattr__(self, "_findings", canonical(findings))

    @property
    def findings(self):
        return json.loads(self._findings)

    def to_trace(self):
        return json.loads(canonical({"attempt": asdict(self.subject.attempt), **self.findings,
                                    "relations": [asdict(item) for item in self.relations],
                                    "gaps": [item.to_dict() for item in self.gaps]}))

    def to_dict(self):
        return {"report_version": 1, "subject": self.subject.to_dict(),
                "sources": [asdict(item) for item in self.subject.sources], "trace": self.to_trace()}

    @classmethod
    def from_trace(cls, value, subject=None):
        subject = subject or TraceSubject(Attempt.from_dict(value["attempt"]))
        if Attempt.from_dict(value["attempt"]) != subject.attempt:
            raise KnowledgeError("reverse report changed its original attempt")
        return cls(subject, {key: item for key, item in value.items() if key not in {"attempt", "relations", "gaps"}},
                   tuple(Relation.from_dict(item) for item in value.get("relations", ())),
                   tuple(Gap.from_dict(item) for item in value.get("gaps", ())))

    @classmethod
    def from_dict(cls, value):
        if "report_version" not in value:
            return cls.from_trace(value.get("trace", value))
        if value["report_version"] != 1:
            raise KnowledgeError("unsupported reverse report version")
        subject = TraceSubject.from_dict(value["subject"])
        if tuple(EvidenceRef.from_dict(item) for item in value["sources"]) != subject.sources:
            raise KnowledgeError("reverse report sources differ from its original subject")
        return cls.from_trace(value["trace"], subject)


def subject_for(attempt, worlds, sources=()):
    for world in worlds:
        nodes = {node.id: node for node in (world.root, *(node for branch in world.branches for node in branch))}
        node = next((node for node in nodes.values() if node.attempt_id == attempt.id), None)
        if node:
            return TraceSubject(attempt, nodes[node.parent_id], tuple(sources))
    return TraceSubject(attempt, sources=tuple(sources))
