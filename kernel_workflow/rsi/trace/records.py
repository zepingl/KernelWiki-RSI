"""Six append-only wire records; each value object validates only its own fields."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from kernelwiki.knowledge import EvidenceRef, KernelState, KnowledgeError, finite, strict_fields

from .gaps import Gap

SCHEMA_VERSION = 4
CODE = "kernel_source"
EXTERNAL = "external"
KNOWLEDGE_PREFIX = "knowledge:"
CODE_OPS = {"generate", "integrate", "adopt", "delegate"}
KNOWLEDGE_OPS = {"ingest", "distill", "author", "revise", "merge"}
READ_ROLES = {"pointed", "read", "cited", "adopted", "skill"}
SOURCE = "source"
OBSERVED_ROLES = {"read", "adopted"}
CODE_MEASURERS = {"engineer", "verifier", "director", "e2e_gate"}
GATES = {"schema_gate", "reproduce", "gpu_validate"}
CODE_VERDICTS = {"committed", "incorrect", "failed", "stop"}
KNOWLEDGE_VERDICTS = {"accepted", "rejected", "retired"}


def _text(value, name):
    if not isinstance(value, str) or not value:
        raise KnowledgeError(f"{name} must be non-empty text")
    return value


def _ids(values, name):
    values = tuple(values)
    if any(not isinstance(value, str) or not value for value in values) or len(set(values)) != len(values):
        raise KnowledgeError(f"{name} must be distinct record ids")
    return values


def _refs(values):
    return tuple(item if isinstance(item, EvidenceRef) else EvidenceRef.from_dict(item) for item in values)


def _observed(refs):
    return bool(refs) and all(item.kind == "observed" for item in refs)


def is_knowledge(subject):
    return subject.startswith(KNOWLEDGE_PREFIX)


@dataclass(frozen=True)
class Use:
    """Information flowing into a record from an earlier node of any tree."""
    target: str
    role: str
    evidence: tuple[EvidenceRef, ...]

    def __post_init__(self):
        _text(self.target, "use target")
        object.__setattr__(self, "evidence", _refs(self.evidence))
        if self.role not in READ_ROLES | {SOURCE} or not self.evidence:
            raise KnowledgeError("a use needs a known role and evidence")
        if self.role in OBSERVED_ROLES and not _observed(self.evidence):
            raise KnowledgeError(f"a claim cannot establish a '{self.role}' use")

    @classmethod
    def from_dict(cls, value):
        strict_fields(value, cls.__dataclass_fields__, cls.__dataclass_fields__)
        return cls(**value)


def _uses(values):
    return tuple(item if isinstance(item, Use) else Use.from_dict(item) for item in values)


@dataclass(frozen=True)
class ContextRef:
    informed_by: tuple[str, ...]
    evidence: tuple[EvidenceRef, ...]

    def __post_init__(self):
        object.__setattr__(self, "informed_by", _ids(self.informed_by, "informed_by"))
        object.__setattr__(self, "evidence", _refs(self.evidence))


@dataclass(frozen=True)
class Slot:
    id: str
    specialty: str = ""
    mode: str = "normal"

    def __post_init__(self):
        _text(self.id, "slot id")
        if not isinstance(self.specialty, str) or self.mode not in {"normal", "deep"}:
            raise KnowledgeError("slot needs a specialty and a normal or deep mode")


@dataclass(frozen=True)
class Generation:
    model: str
    effort: str

    def __post_init__(self):
        _text(self.model, "generation model")
        _text(self.effort, "generation effort")


@dataclass(frozen=True)
class Plan:
    """What only a code-optimization decision has: a round, its context, its agent and a dispatched prefix."""
    round: int
    context: ContextRef
    generation: Generation
    slots: tuple[Slot, ...]
    dispatched: int

    def __post_init__(self):
        object.__setattr__(self, "slots", tuple(item if isinstance(item, Slot) else Slot(**item) for item in self.slots))
        if not isinstance(self.context, ContextRef):
            object.__setattr__(self, "context", ContextRef(**self.context))
        if not isinstance(self.generation, Generation):
            object.__setattr__(self, "generation", Generation(**self.generation))
        if type(self.round) is not int or self.round < 0 or type(self.dispatched) is not int or self.dispatched < 1:
            raise KnowledgeError("a plan needs a round and at least one dispatched slot")
        if self.slots and self.dispatched > len(self.slots):
            raise KnowledgeError("dispatched slots exceed the recorded plan")


@dataclass(frozen=True)
class Tree:
    id: str
    subject: str
    contract: str
    task: KernelState = field(default_factory=KernelState)
    params: dict = field(default_factory=dict)
    parent: dict | None = None
    facets: dict = field(default_factory=dict)

    def __post_init__(self):
        _text(self.id, "tree id")
        _text(self.contract, "tree contract")
        if not isinstance(self.subject, str) or (
                self.subject not in {CODE, EXTERNAL} and not re.fullmatch(r"knowledge:[a-z][a-z0-9_]*", self.subject)):
            raise KnowledgeError("tree subject is kernel_source, external or knowledge:<kind>")
        if not isinstance(self.task, KernelState):
            object.__setattr__(self, "task", KernelState.from_dict(self.task))
        if self.parent is not None:
            if self.subject != CODE:
                raise KnowledgeError("only a code tree is mounted on a delegate slot")
            strict_fields(self.parent, {"tree", "expansion", "slot"}, {"tree", "expansion", "slot"})
            if type(self.parent["slot"]) is not int or self.parent["slot"] < 0:
                raise KnowledgeError("a mounted tree needs a delegate slot")
        if not isinstance(self.params, dict) or not isinstance(self.facets, dict):
            raise KnowledgeError("tree params and facets must be objects")


@dataclass(frozen=True)
class Expansion:
    id: str
    tree: str
    op: str
    parents: tuple[str, ...]
    plan: Plan | None = None
    uses: tuple[Use, ...] = ()
    facets: dict = field(default_factory=dict)

    def __post_init__(self):
        _text(self.id, "expansion id")
        _text(self.tree, "expansion tree")
        object.__setattr__(self, "parents", _ids(self.parents, "parents"))
        object.__setattr__(self, "uses", _uses(self.uses))
        if self.plan is not None and not isinstance(self.plan, Plan):
            object.__setattr__(self, "plan", Plan(**self.plan))
        if self.op not in CODE_OPS | KNOWLEDGE_OPS:
            raise KnowledgeError(f"unknown expansion op: {self.op}")
        if (self.plan is None) == (self.op in CODE_OPS):
            raise KnowledgeError("a code expansion carries a plan; a knowledge expansion does not")
        count = len(self.parents)
        if self.op in {"integrate", "merge"} and count < 2:
            raise KnowledgeError(f"{self.op} combines at least two parents")
        if self.op in CODE_OPS - {"integrate"} | {"revise"} and count != 1:
            raise KnowledgeError(f"{self.op} expands exactly one parent")
        if self.op in {"ingest", "distill", "author"} and count > 1:
            raise KnowledgeError(f"{self.op} continues at most one earlier version")

    @property
    def dispatched(self):
        return self.plan.dispatched if self.plan else 1


@dataclass(frozen=True)
class Node:
    id: str
    tree: str
    expansion: str | None
    slot: int
    state: str | None
    uses: tuple[Use, ...] = ()
    facets: dict = field(default_factory=dict)

    def __post_init__(self):
        _text(self.id, "node id")
        _text(self.tree, "node tree")
        object.__setattr__(self, "uses", _uses(self.uses))
        if self.state is not None:
            _text(self.state, "node state")
        if type(self.slot) is not int or self.slot < 0 or (self.expansion is None and self.slot != 0):
            raise KnowledgeError("node slot must be a dispatch position; a node without expansion uses slot 0")


@dataclass(frozen=True)
class Measure:
    """A check that ran: code measurers score the node; knowledge gates only record that they ran."""
    node: str
    contract: str
    by: str
    independent: bool
    score: float | None
    evidence: tuple[EvidenceRef, ...]
    per_case: tuple = ()

    def __post_init__(self):
        _text(self.node, "measured node")
        _text(self.contract, "measurement contract")
        object.__setattr__(self, "evidence", _refs(self.evidence))
        object.__setattr__(self, "per_case", tuple(self.per_case))
        if self.by not in CODE_MEASURERS | GATES or type(self.independent) is not bool:
            raise KnowledgeError("measure needs a known measurer")
        if self.by in GATES and self.score is not None:
            raise KnowledgeError("a knowledge gate passes or fails through a decision, not a score")
        if self.by in CODE_MEASURERS and (self.score is None or finite(self.score) <= 0):
            raise KnowledgeError("a code measure needs a positive score")
        if not self.evidence or (self.independent and not _observed(self.evidence)):
            raise KnowledgeError("an independent measure needs observed receipts")


@dataclass(frozen=True)
class Decide:
    tree: str
    node: str | None
    verdict: str
    reason: str
    threshold: float | None = None

    def __post_init__(self):
        _text(self.tree, "decision tree")
        if self.verdict not in CODE_VERDICTS | KNOWLEDGE_VERDICTS or not isinstance(self.reason, str):
            raise KnowledgeError("decision needs a known verdict and a reason")
        if (self.node is None) != (self.verdict == "stop"):
            raise KnowledgeError("only a stop decision is made without a node")
        if self.threshold is not None:
            finite(self.threshold)


@dataclass(frozen=True)
class GapRecord:
    subject: str
    gap: Gap

    def __post_init__(self):
        _text(self.subject, "gap subject")
        if not isinstance(self.gap, Gap):
            object.__setattr__(self, "gap", Gap.from_dict(self.gap))

    def to_dict(self):
        return {"subject": self.subject, "gap": self.gap.to_dict()}


KINDS = {"tree": Tree, "expansion": Expansion, "node": Node, "measure": Measure, "decide": Decide, "gap": GapRecord}
NAMES = {cls: name for name, cls in KINDS.items()}


def to_wire(record):
    body = record.to_dict() if isinstance(record, GapRecord) else asdict(record)
    return {"kind": NAMES[type(record)], "schema_version": SCHEMA_VERSION, **body}


def from_wire(value):
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION or value.get("kind") not in KINDS:
        raise KnowledgeError(f"trace record needs a known kind and schema version {SCHEMA_VERSION}")
    cls = KINDS[value["kind"]]
    body = {key: item for key, item in value.items() if key not in {"kind", "schema_version"}}
    strict_fields(body, cls.__dataclass_fields__)
    return cls(**body)
