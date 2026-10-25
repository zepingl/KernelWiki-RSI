"""Read-only verified buffer snapshots; artifact layout stays private."""

import json
from dataclasses import dataclass
from pathlib import Path

from kernelwiki.knowledge import EvidenceRef, KnowledgeError, canonical

from ..discovery import Attempt, DiscoveryWorld
from ..evidence import evidence_refs, read_json, read_jsonl, verify, verify_artifacts
from ..reverse_engineering.model import ReverseReport, subject_for
from .cohorts import cohorts, split_worlds
from .objective import ReplayObjective


@dataclass(frozen=True)
class BufferSnapshot:
    directory: Path
    _manifest: bytes
    worlds: tuple[DiscoveryWorld, ...]
    attempts: tuple[Attempt, ...]
    _traces: tuple[bytes, ...]
    _excluded: tuple[bytes, ...]

    @property
    def manifest(self):
        return json.loads(self._manifest)

    @property
    def budgets(self):
        return tuple(self.manifest["budgets"])

    @property
    def objective(self):
        return ReplayObjective(**self.manifest["objective"])

    @property
    def excluded(self):
        return tuple(json.loads(value) for value in self._excluded)

    def select(self, split=None, *, cohort="primary"):
        manifest = self.manifest
        if cohort not in {"primary", "diagnostic"}:
            raise KnowledgeError("buffer selection requires a primary or diagnostic cohort")
        worlds = self.worlds if split is None else split_worlds(self.worlds, manifest["splits"])[split]
        if cohort == "diagnostic":
            return worlds
        if "cohorts" not in manifest:
            raise KnowledgeError("primary and diagnostic cohorts must be frozen before candidate evaluation")
        selected = set(manifest["cohorts"]["primary"])
        return tuple(world for world in worlds if world.id in selected)

    def attempt(self, attempt_id):
        attempt = next((item for item in self.attempts if item.id == attempt_id), None)
        if attempt is None:
            raise KnowledgeError(f"unknown attempt: {attempt_id}")
        return attempt

    def subject(self, attempt_id):
        return subject_for(self.attempt(attempt_id), self.worlds)

    def parent(self, attempt_id):
        return self.subject(attempt_id).parent

    def reverse_reports(self, kernels=None):
        reports = []
        for payload in self._traces:
            trace = json.loads(payload)
            if kernels is None or trace["attempt"]["kernel"] in kernels:
                reports.append(ReverseReport.from_trace(trace, self.subject(trace["attempt"]["id"])))
        return tuple(reports)


def open_buffer(directory):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    if manifest.get("schema_version") != 2:
        raise KnowledgeError("legacy replay pool requires reconstruction; frozen archives are not requalified")
    verify_artifacts(directory, manifest)
    worlds = tuple(DiscoveryWorld.from_dict(value) for value in read_jsonl(directory / "worlds.jsonl"))
    if "cohorts" in manifest and manifest["cohorts"] != cohorts(worlds, manifest["budgets"]):
        raise KnowledgeError("frozen primary cohort differs from the source worlds")
    checked = set()
    attempts, traces = {}, []
    for value in manifest.get("sources", ()):
        verify(EvidenceRef.from_dict(value))
    for world in worlds:
        for node in (world.root, *(node for branch in world.branches for node in branch)):
            for item in (*node.input_evidence, *node.measurement.evidence,
                         *node.state.evidence, *node.context.evidence):
                if item not in checked:
                    verify(item)
                    checked.add(item)
    if (directory / "attempts.jsonl").exists():
        for value in read_jsonl(directory / "attempts.jsonl"):
            attempt = Attempt.from_dict(value)
            if attempt.id in attempts:
                raise KnowledgeError("duplicate external attempt in replay pool")
            attempts[attempt.id] = attempt
            refs = [attempt.invocation, *attempt.audit, *attempt.measurement.evidence]
            if attempt.input:
                refs.append(attempt.input)
            if attempt.state:
                refs.extend(attempt.state.evidence)
            if attempt.context:
                refs.extend(attempt.context.evidence)
            for item in refs:
                if item not in checked:
                    verify(item)
                    checked.add(item)
        if (directory / "reverse.jsonl").exists():
            if "reverse.jsonl" not in manifest["artifacts"]:
                raise KnowledgeError("reverse evidence must be frozen by the pool manifest")
            reverse_ids = set()
            for trace in read_jsonl(directory / "reverse.jsonl"):
                if trace["attempt"]["id"] in reverse_ids:
                    raise KnowledgeError("duplicate reverse trace in frozen pool")
                reverse_ids.add(trace["attempt"]["id"])
                if Attempt.from_dict(trace["attempt"]) != attempts.get(trace["attempt"]["id"]):
                    raise KnowledgeError("reverse trace does not match the original attempt")
                traces.append(trace)
                for item in evidence_refs(trace):
                    if item not in checked:
                        verify(item)
                        checked.add(item)
    return BufferSnapshot(directory.resolve(), canonical(manifest), worlds, tuple(attempts.values()),
                          tuple(canonical(trace) for trace in traces),
                          tuple(canonical(row) for row in read_jsonl(directory / "excluded.jsonl"))
                          if (directory / "excluded.jsonl").exists() else ())
