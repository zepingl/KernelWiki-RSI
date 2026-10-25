"""Immutable, line-addressable evidence; declarations never become observations."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from kernel_workflow.scripts.workspace_sources import check as check_workspace_sources
from kernelwiki._atomic import atomic_bytes, fsync_dir
from kernelwiki.knowledge import (
    EvidenceRef,
    KnowledgeError,
    canonical,
    digest,
    relative_path,
)


def reference(path, *, kind="observed", start_line=1, end_line=None):
    source = Path(path).resolve(strict=True)
    payload = source.read_bytes()
    lines = max(1, len(payload.splitlines()))
    return EvidenceRef(str(source), hashlib.sha256(payload).hexdigest(), kind,
                       start_line, lines if end_line is None else end_line)


def verify(reference: EvidenceRef, root=None) -> bytes:
    path = Path(reference.path)
    if not path.is_absolute():
        if root is None:
            raise KnowledgeError("relative evidence needs an explicit root")
        path = Path(root) / path
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != reference.sha256:
        raise KnowledgeError(f"evidence changed: {path}")
    if reference.end_line > max(1, len(payload.splitlines())):
        raise KnowledgeError(f"evidence range exceeds file: {path}")
    return payload




def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    fsync_dir(target.parent, strict=True)


def artifact_hashes(directory):
    directory = Path(directory)
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise KnowledgeError("immutable artifacts cannot be symlinks")
        if path.is_file() and path != directory / "result.json":
            result[str(path.relative_to(directory))] = reference(path).sha256
    return result


def verify_artifacts(directory, manifest):
    for name, expected in manifest["artifacts"].items():
        relative_path(name)
        path = Path(directory) / name
        if path.is_symlink() or reference(path).sha256 != expected:
            raise KnowledgeError(f"frozen artifact changed: {path}")


def write_jsonl(path, values):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        for value in values:
            stream.write(canonical(asdict(value) if hasattr(value, "__dataclass_fields__") else value) + b"\n")


def read_jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def persist(path, value):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    atomic_bytes(path, canonical(value), durable=True)
    return reference(path)


def snapshot(workspace, source_paths, directory=None):
    workspace = Path(workspace).resolve(strict=True)
    check_workspace_sources(workspace)
    files = {}
    for name in source_paths:
        relative_path(name)
        path = workspace / name
        if not path.exists():
            raise KnowledgeError(f"declared source path is missing: {path}")
        paths = sorted(path.rglob("*")) if path.is_dir() else [path]
        for source in paths:
            if source.is_symlink() or any(parent.is_symlink() for parent in source.parents if parent != workspace):
                raise KnowledgeError("source snapshots cannot follow symlinks")
            if source.is_dir() or any(part in {"__pycache__", ".git"} for part in source.relative_to(workspace).parts):
                continue
            before = source.stat()
            payload = source.read_bytes()
            after = source.stat()
            if before.st_mtime_ns != after.st_mtime_ns or before.st_size != after.st_size:
                raise KnowledgeError("source changed during boundary capture")
            sha = hashlib.sha256(payload).hexdigest()
            files[str(source.relative_to(workspace))] = {"sha256": sha, "size": len(payload)}
            if directory is not None:
                target = Path(directory) / "objects" / sha
                if target.exists():
                    if target.read_bytes() != payload:
                        raise KnowledgeError("immutable captured source object changed")
                else:
                    atomic_bytes(target, payload, durable=True)
    if not files:
        raise KnowledgeError("source snapshot is empty")
    return {"workspace": str(workspace), "files": files, "missing": [], "fingerprint": digest(files)}


def evidence_refs(value):
    found = {}
    def visit(item):
        if isinstance(item, dict):
            fields = {"path", "sha256", "kind", "start_line", "end_line"}
            if fields <= item.keys():
                evidence = EvidenceRef.from_dict({key: item[key] for key in fields})
                found[canonical(asdict(evidence))] = evidence
            else:
                for child in item.values():
                    visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
    visit(value)
    return tuple(found.values())
