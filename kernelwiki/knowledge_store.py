"""Local knowledge snapshots with compare-and-swap publication and rollback."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import yaml

from ._atomic import atomic_bytes, fsync_dir
from .knowledge import (
    EvidenceRef,
    KnowledgeError,
    KnowledgePolicy,
    KnowledgeRecord,
    canonical,
    digest,
    query,
    read_markdown,
    relative_path,
    render_markdown,
    validate_policy,
)


@dataclass(frozen=True)
class KnowledgeSnapshot:
    revision: str
    policy: KnowledgePolicy
    records: tuple[KnowledgeRecord, ...]
    files: tuple[tuple[str, bytes], ...]


@dataclass(frozen=True)
class KnowledgeUpdate:
    policy: KnowledgePolicy
    records: tuple[KnowledgeRecord, ...]
    validation: dict
    evidence: tuple[EvidenceRef, ...] = ()
    # Reference pages written verbatim at their own paths: added (e.g. an ingested learned card) or replaced
    # (e.g. a page amended with field evidence). Neither the old nor the new page may carry a policy rule, so
    # they cannot change policy behavior.
    files: tuple[tuple[str, bytes], ...] = ()


def _manifest(files):
    return {name: {"sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}
            for name, payload in sorted(files.items())}


def _snapshot(files):
    policy = KnowledgePolicy.from_dict(yaml.safe_load(files["knowledge/policy.yaml"]))
    records = tuple(read_markdown(name.removeprefix("knowledge/"), payload.decode())
                    for name, payload in sorted(files.items()) if name.startswith("knowledge/") and name.endswith(".md"))
    validate_policy(policy, records)
    return KnowledgeSnapshot(digest(_manifest(files)), policy, records, tuple(sorted(files.items())))


class KnowledgeClient:
    """The checkout is an editable view; pinned version directories are read truth."""

    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        self.state = self.root / ".kernelwiki-rsi"

    def _safe(self, root, name):
        relative_path(name)
        path = root
        for component in name.split("/"):
            path = path / component
            if path.is_symlink():
                raise KnowledgeError(f"managed paths cannot be symlinks: {path}")
        return path

    def _working_files(self):
        directory = self._safe(self.root, "knowledge")
        files = {}
        for base, directories, names in os.walk(directory, followlinks=False):
            for name in directories + names:
                path = Path(base) / name
                if path.is_symlink():
                    raise KnowledgeError(f"knowledge paths cannot be symlinks: {path}")
            for name in names:
                path = Path(base) / name
                if name.endswith(".md") or path == directory / "policy.yaml":
                    files[str(path.relative_to(self.root))] = path.read_bytes()
        if "knowledge/policy.yaml" not in files:
            raise KnowledgeError("knowledge/policy.yaml is required")
        return files

    def _current(self):
        self._safe(self.root, ".kernelwiki-rsi")
        active = self._safe(self.state, "active.json")
        if active.exists():
            return json.loads(active.read_text())["revision"]
        pending = self._safe(self.state, "pending.json")
        if pending.exists():
            return json.loads(pending.read_text())["from"]
        return None

    def snapshot(self, revision=None):
        self._safe(self.root, ".kernelwiki-rsi")
        revision = revision or self._current()
        if revision is None:
            return _snapshot(self._working_files())
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{64}", revision):
            raise KnowledgeError("revision must be a complete SHA-256")
        directory = self._safe(self.state, f"versions/{revision}")
        manifest = json.loads(self._safe(directory, "manifest.json").read_text())
        if manifest.get("schema_version") != 1 or manifest.get("revision") != revision:
            raise KnowledgeError("invalid knowledge snapshot manifest")
        files = {name: self._safe(directory, name).read_bytes() for name in manifest["files"]}
        if _manifest(files) != manifest["files"] or digest(manifest["files"]) != revision:
            raise KnowledgeError("immutable knowledge snapshot was modified")
        return _snapshot(files)

    def query(self, state, revision=None, top_k=3):
        return query(self.snapshot(revision).records, state, top_k)

    @contextmanager
    def _writer(self):
        try:
            import fcntl
        except ImportError as error:
            raise KnowledgeError("knowledge publication requires POSIX file locking") from error
        self._safe(self.root, ".kernelwiki-rsi").mkdir(exist_ok=True)
        lock_path = self._safe(self.state, "write.lock")
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise KnowledgeError("another knowledge writer holds the lock") from error
            yield
        finally:
            os.close(descriptor)

    def _persist(self, snapshot):
        versions = self._safe(self.state, "versions")
        versions.mkdir(exist_ok=True)
        target = self._safe(versions, snapshot.revision)
        if target.exists():
            self.snapshot(snapshot.revision)
            return
        staging = Path(tempfile.mkdtemp(prefix=".stage-", dir=self.state))
        try:
            for name, payload in snapshot.files:
                atomic_bytes(self._safe(staging, name), payload, durable=True)
            manifest = {"schema_version": 1, "revision": snapshot.revision,
                        "files": _manifest(dict(snapshot.files))}
            atomic_bytes(staging / "manifest.json", canonical(manifest), durable=True)
            os.rename(staging, target)
            fsync_dir(versions, strict=True)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def _check_worktree(self, expected):
        files = {name: payload for name, payload in expected.files if name.startswith("knowledge/")}
        if self._working_files() != files:
            raise KnowledgeError("managed knowledge has user modifications; publication aborted")

    def _sync(self, previous, target, *, recovering=False):
        old = {name: payload for name, payload in previous.files if name.startswith("knowledge/")}
        new = {name: payload for name, payload in target.files if name.startswith("knowledge/")}
        current = self._working_files()
        if recovering:
            if set(current) - (set(old) | set(new)) or any(
                    current.get(name) not in (old.get(name), new.get(name)) for name in set(old) | set(new)):
                raise KnowledgeError("user edits conflict with interrupted knowledge publication")
        elif current != old:
            raise KnowledgeError("knowledge changed during publication")
        for name in sorted(set(old) | set(new)):
            path = self._safe(self.root, name)
            if name not in new:
                if path.exists():
                    path.unlink()
                    fsync_dir(path.parent, strict=True)
            elif current.get(name) != new[name]:
                atomic_bytes(path, new[name], durable=True)

    def _recover(self):
        pending_path = self._safe(self.state, "pending.json")
        if not pending_path.exists():
            return
        pending = json.loads(pending_path.read_text())
        previous, target = self.snapshot(pending["from"]), self.snapshot(pending["to"])
        if self._current() not in {previous.revision, target.revision}:
            raise KnowledgeError("interrupted publication has an unexpected active revision")
        self._sync(previous, target, recovering=True)
        atomic_bytes(self.state / "active.json", canonical({"revision": target.revision}), durable=True)
        pending_path.unlink()
        fsync_dir(self.state, strict=True)

    def recover(self):
        with self._writer():
            self._recover()
            return self.snapshot().revision

    def _activate(self, previous, target):
        self._persist(previous)
        self._persist(target)
        self._check_worktree(previous)
        atomic_bytes(self.state / "pending.json",
                     canonical({"from": previous.revision, "to": target.revision}), durable=True)
        self._sync(previous, target)
        atomic_bytes(self.state / "active.json", canonical({"revision": target.revision}), durable=True)
        (self.state / "pending.json").unlink()
        fsync_dir(self.state, strict=True)
        return target.revision

    def publish(self, candidate: KnowledgeUpdate, expected_revision: str):
        if candidate.validation.get("status") not in {"replay_validated", "evidence_update"}:
            raise KnowledgeError("publication requires replay validation or an explicit evidence update")
        if not candidate.evidence:
            raise KnowledgeError("publication needs supporting evidence")
        if len({record.id for record in candidate.records}) != len(candidate.records):
            raise KnowledgeError("duplicate records in knowledge update")
        with self._writer():
            self._recover()
            previous = self.snapshot()
            if previous.revision != expected_revision:
                raise KnowledgeError("stale expected knowledge revision")
            self._check_worktree(previous)
            records = {record.id: record for record in previous.records}
            if candidate.validation["status"] == "evidence_update" and (
                    candidate.policy != previous.policy or any(
                        record.policy_rule is not None and (
                            record.id not in records
                            or record.policy_rule != records[record.id].policy_rule
                            or record.applies_to != records[record.id].applies_to)
                        or record.id in records and records[record.id].policy_rule is not None
                        and record.policy_rule is None for record in candidate.records)):
                raise KnowledgeError("evidence-only publication cannot change policy behavior")
            paths = {read_markdown(name.removeprefix("knowledge/"), payload.decode()).id: name
                     for name, payload in previous.files if name.startswith("knowledge/") and name.endswith(".md")}
            files = dict(previous.files)
            for record in candidate.records:
                if record.policy_rule is not None and not record.evidence:
                    raise KnowledgeError("executable knowledge requires evidence")
                for item in record.evidence:
                    payload = Path(item.path).read_bytes()
                    if (hashlib.sha256(payload).hexdigest() != item.sha256
                            or item.end_line > max(1, len(payload.splitlines()))):
                        raise KnowledgeError("record evidence changed or its range is invalid")
                if records.get(record.id) != record:
                    name = paths.get(record.id, f"knowledge/general/rsi/{record.id}.md")
                    files[name] = render_markdown(record)
                records[record.id] = record
            for name, payload in candidate.files:
                relative_path(name)
                if not name.startswith("knowledge/") or not name.endswith(".md") or not isinstance(payload, bytes):
                    raise KnowledgeError(f"a verbatim page must be a knowledge/*.md file: {name}")
                pages = [payload] + ([files[name]] if name in files else [])
                if any(read_markdown(name.removeprefix("knowledge/"), page.decode()).policy_rule is not None
                       for page in pages):
                    raise KnowledgeError("a verbatim page cannot carry or replace a policy rule")
                files[name] = payload
            validate_policy(candidate.policy, tuple(records.values()))
            if candidate.policy != previous.policy:    # an unchanged policy keeps its bytes (no reformat churn)
                files["knowledge/policy.yaml"] = yaml.safe_dump(candidate.policy.to_dict(), sort_keys=False).encode()
            files["validation.json"] = canonical(candidate.validation)
            evidence_manifest = []
            for item in candidate.evidence:
                source = Path(item.path)
                payload = source.read_bytes()
                if hashlib.sha256(payload).hexdigest() != item.sha256:
                    raise KnowledgeError("publication evidence changed")
                evidence_manifest.append({"path": item.path, "sha256": item.sha256,
                                          "kind": item.kind, "start_line": item.start_line, "end_line": item.end_line})
                if len(payload) <= 1024 * 1024:
                    files[f"evidence/{item.sha256}"] = payload
            files["evidence/references.json"] = canonical(evidence_manifest)
            return self._activate(previous, _snapshot(files))

    def rollback(self, target_revision, expected_revision):
        with self._writer():
            self._recover()
            previous = self.snapshot()
            if previous.revision != expected_revision:
                raise KnowledgeError("stale expected knowledge revision")
            self._check_worktree(previous)
            return self._activate(previous, self.snapshot(target_revision))
