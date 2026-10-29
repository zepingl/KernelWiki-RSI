"""Explicit provenance gaps and the archive coverage that bounds what a search can prove."""

from dataclasses import asdict, dataclass

from kernelwiki.knowledge import EvidenceRef, KnowledgeError, strict_fields


@dataclass(frozen=True)
class Coverage:
    scope: tuple[str, ...]
    inspected: tuple[EvidenceRef, ...] = ()
    remaining: tuple[str, ...] = ()
    unavailable: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ("scope", "inspected", "remaining", "unavailable"):
            value = getattr(self, name)
            if not isinstance(value, (tuple, list)):
                raise KnowledgeError("coverage collections must be explicit sequences")
            object.__setattr__(self, name, tuple(value))
        if not self.scope or any(not isinstance(value, str) or not value for value in self.scope):
            raise KnowledgeError("evidence coverage needs an explicit archive scope")
        if any(not isinstance(item, EvidenceRef) or item.kind not in {"observed", "review"} for item in self.inspected):
            raise KnowledgeError("coverage cannot be established by claims")
        if any(not isinstance(value, str) or not value for value in (*self.remaining, *self.unavailable)):
            raise KnowledgeError("unread coverage paths must be explicit text")
        if len(set(self.remaining)) != len(self.remaining):
            raise KnowledgeError("evidence coverage repeats a pending path")

    @property
    def status(self):
        return "partial" if self.remaining or self.unavailable else "complete"

    def to_dict(self):
        return {**asdict(self), "status": self.status}

    @classmethod
    def from_dict(cls, value):
        strict_fields(value, {*cls.__dataclass_fields__, "status"}, cls.__dataclass_fields__)
        result = cls(tuple(value["scope"]), tuple(EvidenceRef.from_dict(item) for item in value["inspected"]),
                     tuple(value["remaining"]), tuple(value["unavailable"]))
        if value.get("status", result.status) != result.status:
            raise KnowledgeError("coverage status disagrees with its unread evidence")
        return result


@dataclass(frozen=True)
class Gap:
    id: str
    classification: str
    question: str
    reasons: tuple[str, ...] = ()
    coverage: Coverage | None = None

    def __post_init__(self):
        object.__setattr__(self, "reasons", tuple(self.reasons))
        if self.coverage is not None and not isinstance(self.coverage, Coverage):
            raise KnowledgeError("gap coverage must be a verified scope value")
        if any(not isinstance(reason, str) for reason in self.reasons):
            raise KnowledgeError("gap reasons must be explicit text")
        if any(not isinstance(value, str) or not value for value in (self.id, self.question, self.classification)):
            raise KnowledgeError("a provenance gap needs its identity and question")
        if self.classification not in {"resolvable_local", "needs_method_review", "needs_online_control",
                                       "unrecoverable_history"}:
            raise KnowledgeError("unknown provenance gap classification")
        if self.coverage and self.coverage.status == "partial" and self.classification == "unrecoverable_history":
            raise KnowledgeError("partial archive coverage cannot establish absence")

    def to_dict(self):
        result = {"id": self.id, "classification": self.classification, "question": self.question}
        if self.reasons:
            result["reasons"] = list(self.reasons)
        if self.coverage:
            result["coverage"] = self.coverage.to_dict()
        return result

    @classmethod
    def from_dict(cls, value):
        strict_fields(value, cls.__dataclass_fields__, {"id", "classification", "question"})
        return cls(value["id"], value["classification"], value["question"], tuple(value.get("reasons", ())),
                   Coverage.from_dict(value["coverage"]) if value.get("coverage") else None)
