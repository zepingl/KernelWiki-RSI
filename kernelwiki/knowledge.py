"""Knowledge values shared by retrieval, publication, and exploration policies."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath

import yaml


class KnowledgeError(ValueError):
    """A knowledge document violates the versioned public contract."""


TASK_FIELDS = frozenset({"operator", "backend", "gfx", "dtype", "bottleneck", "shape"})
OBSERVATION_FIELDS = frozenset({"node_kind", "measurement_status", "parent_gain",
                                "stagnation", "remaining_budget"})
NUMERIC_FIELDS = frozenset({"parent_gain", "stagnation", "remaining_budget"})
CATEGORICAL_VALUES = {"node_kind": {"root", "leaf"},
                      "measurement_status": {"qualified", "failed", "unknown"}}


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def strict_fields(value, allowed, required=()):
    if not isinstance(value, dict):
        raise KnowledgeError("expected an object")
    if set(value) - set(allowed) or set(required) - set(value):
        raise KnowledgeError(f"unexpected or missing fields: {sorted(value)}")


def identifier(value: str) -> str:
    if (not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._/-]{0,255}", value)
            or any(part in {"", ".", ".."} for part in value.split("/"))):
        raise KnowledgeError("invalid knowledge identifier")
    return value


def relative_path(value: str) -> str:
    if (not isinstance(value, str) or not value or "\\" in value or "\0" in value
            or PurePosixPath(value).is_absolute()
            or any(part in {"", ".", ".."} for part in value.split("/"))):
        raise KnowledgeError("expected a safe relative path")
    return value


def finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise KnowledgeError("expected a finite number")
    return value


@dataclass(frozen=True)
class KernelState:
    operator: str = ""
    backend: str = ""
    gfx: str = ""
    dtype: str = ""
    bottleneck: str = ""
    shape: str = ""

    def __post_init__(self):
        if any(not isinstance(value, str) for value in asdict(self).values()):
            raise KnowledgeError("kernel state fields must be strings; unknown is empty")

    @classmethod
    def from_dict(cls, value):
        strict_fields(value, TASK_FIELDS)
        return cls(**value)


@dataclass(frozen=True)
class EvidenceRef:
    path: str
    sha256: str
    kind: str = "observed"
    start_line: int = 1
    end_line: int = 1

    def __post_init__(self):
        if not isinstance(self.path, str) or not self.path or "\0" in self.path:
            raise KnowledgeError("evidence needs a path")
        if not isinstance(self.sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise KnowledgeError("evidence needs a complete SHA-256")
        if self.kind not in {"observed", "claim", "review"}:
            raise KnowledgeError("invalid evidence kind")
        if (type(self.start_line) is not int or type(self.end_line) is not int
                or not 1 <= self.start_line <= self.end_line):
            raise KnowledgeError("invalid evidence line range")

    @classmethod
    def from_dict(cls, value):
        strict_fields(value, cls.__dataclass_fields__, {"path", "sha256"})
        return cls(**value)


@dataclass(frozen=True)
class Condition:
    field: str
    op: str
    value: str | int | float | tuple[str, ...]

    def __post_init__(self):
        if self.field not in TASK_FIELDS | OBSERVATION_FIELDS:
            raise KnowledgeError(f"policy field is not observable: {self.field}")
        if self.field in NUMERIC_FIELDS:
            if self.op not in {"eq", "gte", "lte"}:
                raise KnowledgeError("numeric conditions support eq/gte/lte")
            finite(self.value)
        elif self.op == "in":
            if (not isinstance(self.value, tuple) or not self.value
                    or any(not isinstance(item, str) or not item for item in self.value)):
                raise KnowledgeError("in condition needs a nonempty tuple of strings")
        elif self.op != "eq" or not isinstance(self.value, str) or not self.value:
            raise KnowledgeError("categorical conditions support eq/in")
        if self.field in CATEGORICAL_VALUES:
            values = self.value if self.op == "in" else (self.value,)
            if any(value not in CATEGORICAL_VALUES[self.field] for value in values):
                raise KnowledgeError("invalid categorical observation value")

    def matches(self, values):
        actual = values.get(self.field)
        if actual is None or actual == "":
            return False
        if self.op == "in":
            return actual in self.value
        if self.op == "gte":
            return actual >= self.value
        if self.op == "lte":
            return actual <= self.value
        return actual == self.value

    @classmethod
    def from_dict(cls, value):
        strict_fields(value, {"field", "op", "value"}, {"field", "op", "value"})
        return cls(value["field"], value["op"],
                   tuple(value["value"]) if isinstance(value["value"], list) else value["value"])


def conditions(values, fields):
    if not isinstance(values, (list, tuple)):
        raise KnowledgeError("conditions must be an array")
    result = tuple(Condition.from_dict(value) for value in values)
    if any(condition.field not in fields for condition in result):
        raise KnowledgeError("condition is outside this rule's scope")
    if len({(condition.field, condition.op) for condition in result}) != len(result):
        raise KnowledgeError("duplicate condition")
    return tuple(sorted(result, key=lambda condition: (condition.field, condition.op)))


@dataclass(frozen=True)
class PolicyRule:
    when: tuple[Condition, ...]
    effect: str

    def __post_init__(self):
        if self.effect not in {"prefer", "defer", "stop"}:
            raise KnowledgeError("policy effect must be prefer, defer, or stop")
        if not isinstance(self.when, tuple) or any(
                not isinstance(condition, Condition) or condition.field not in OBSERVATION_FIELDS
                for condition in self.when):
            raise KnowledgeError("invalid observation conditions")
        if self.effect == "stop" and any(condition.field not in {"stagnation", "remaining_budget"}
                                         for condition in self.when):
            raise KnowledgeError("stop may depend only on global prefix observations")

    @classmethod
    def from_dict(cls, value):
        strict_fields(value, {"when", "effect"}, {"when", "effect"})
        return cls(conditions(value["when"], OBSERVATION_FIELDS), value["effect"])


@dataclass(frozen=True)
class KnowledgeRecord:
    id: str
    title: str
    body: str
    applies_to: tuple[Condition, ...] = ()
    evidence: tuple[EvidenceRef, ...] = ()
    policy_rule: PolicyRule | None = None

    def __post_init__(self):
        identifier(self.id)
        if not isinstance(self.title, str) or not self.title or not isinstance(self.body, str):
            raise KnowledgeError("knowledge needs a title and text body")
        if not isinstance(self.applies_to, tuple) or any(
                not isinstance(condition, Condition) or condition.field not in TASK_FIELDS
                for condition in self.applies_to):
            raise KnowledgeError("invalid task applicability")
        if not isinstance(self.evidence, tuple) or any(
                not isinstance(reference, EvidenceRef) for reference in self.evidence):
            raise KnowledgeError("invalid evidence references")
        if self.policy_rule is not None and not isinstance(self.policy_rule, PolicyRule):
            raise KnowledgeError("invalid policy rule")

    @classmethod
    def from_dict(cls, value):
        strict_fields(value, cls.__dataclass_fields__, {"id", "title", "body"})
        return cls(value["id"], value["title"], value["body"],
                   conditions(value.get("applies_to", []), TASK_FIELDS),
                   tuple(EvidenceRef.from_dict(item) for item in value.get("evidence", [])),
                   PolicyRule.from_dict(value["policy_rule"]) if value.get("policy_rule") is not None else None)

    def to_dict(self):
        return json.loads(canonical(asdict(self)))

    def matches(self, state: KernelState):
        values = asdict(state)
        return all(condition.matches(values) for condition in self.applies_to)


@dataclass(frozen=True)
class KnowledgePolicy:
    schema_version: int = 1
    rule_ids: tuple[str, ...] = ()
    top_k: int = 3
    workers: int = 2

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise KnowledgeError("unsupported policy schema")
        if type(self.top_k) is not int or not 1 <= self.top_k <= 3:
            raise KnowledgeError("top_k must be in [1, 3]")
        if type(self.workers) is not int or not 1 <= self.workers <= 2:
            raise KnowledgeError("workers must be in [1, 2]")
        if not isinstance(self.rule_ids, tuple) or len(set(self.rule_ids)) != len(self.rule_ids):
            raise KnowledgeError("rule_ids must be a unique tuple")
        for rule_id in self.rule_ids:
            identifier(rule_id)

    @classmethod
    def from_dict(cls, value):
        strict_fields(value, cls.__dataclass_fields__, {"schema_version", "rule_ids"})
        if not isinstance(value["rule_ids"], list):
            raise KnowledgeError("rule_ids must be an array")
        return cls(**{**value, "rule_ids": tuple(value["rule_ids"])})

    def to_dict(self):
        return json.loads(canonical(asdict(self)))


def validate_policy(policy: KnowledgePolicy, records: tuple[KnowledgeRecord, ...]):
    by_id = {record.id: record for record in records}
    if len(by_id) != len(records):
        raise KnowledgeError("duplicate knowledge id")
    signatures = {}
    for rule_id in policy.rule_ids:
        record = by_id.get(rule_id)
        if record is None or record.policy_rule is None:
            raise KnowledgeError(f"missing executable rule: {rule_id}")
        signature = (record.applies_to, record.policy_rule.when)
        previous = signatures.setdefault(signature, record.policy_rule.effect)
        if previous != record.policy_rule.effect:
            raise KnowledgeError("identical rule conditions have conflicting effects")


def query(records, state, top_k=3, *, rule_ids=None):
    if type(top_k) is not int or not 1 <= top_k <= 3:
        raise KnowledgeError("top_k must be in [1, 3]")
    matches = [record for record in records if record.matches(state)
               and (rule_ids is None or record.id in rule_ids)]
    return tuple(sorted(matches, key=lambda record: (-len(record.applies_to), record.id))[:top_k])


def read_markdown(path: str, text: str) -> KnowledgeRecord:
    relative_path(path)
    if text.startswith("---\n"):
        parts = text.split("---\n", 2)
        if len(parts) == 3 and re.search(r"^kernelwiki:\s*", parts[1], re.MULTILINE):
            try:
                metadata = yaml.safe_load(parts[1])
            except yaml.YAMLError as error:
                raise KnowledgeError(f"invalid KernelWiki front matter: {path}") from error
            if isinstance(metadata, dict) and "kernelwiki" in metadata:
                return KnowledgeRecord.from_dict({**metadata["kernelwiki"], "body": parts[2]})
    title = next((line.lstrip("# ").strip() for line in text.splitlines()
                  if line.startswith("#")), path)
    path_parts = PurePosixPath(path).parts
    applies = []
    for directory, field in (("operator", "operator"), ("backend", "backend")):
        if directory not in path_parts:
            continue
        position = path_parts.index(directory) + 1
        if position < len(path_parts) - 1:
            applies.append(Condition(field, "eq", path_parts[position]))
        elif field == "backend" and position < len(path_parts) and path_parts[position] != "README.md":
            backend = re.sub(r"_(aiter|notes)$", "", PurePosixPath(path_parts[position]).stem)
            applies.append(Condition(field, "eq", backend))
    record_id = "reference/" + re.sub(r"[^a-z0-9._/-]", "-", path.lower())
    return KnowledgeRecord(record_id, title, text, tuple(applies))


def render_markdown(record: KnowledgeRecord) -> bytes:
    metadata = record.to_dict()
    body = metadata.pop("body")
    return ("---\n" + yaml.safe_dump({"kernelwiki": metadata}, sort_keys=False,
                                     allow_unicode=True) + "---\n" + body).encode("utf-8")
