"""GEAK's knowledge layout as a view of KernelWiki: one rule table, used in both directions, no content matching.

GEAK agents are prompted with `perf_knowledge/` paths (`operators/<op>/backends/<lang>.md`, ...). KernelWiki keeps
the same documents under its own layout. A run reads a materialized view of one pinned KernelWiki commit, so
every file it can open names its KernelWiki document; history, which read GEAK's own copy, resolves through
the same rules.
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import tarfile
from pathlib import Path

from kernelwiki.knowledge import KnowledgeError

# (GEAK view prefix, KernelWiki prefix), most specific first. "{op}" binds exactly one path segment.
RULES = (
    ("operators/{op}/backends/", "knowledge/operator/{op}/backend/"),
    ("operators/{op}/overview.md", "knowledge/operator/{op}/README.md"),
    ("operators/{op}/", "knowledge/operator/{op}/"),
    ("index/", "knowledge/general/reference/geak_index/"),
    ("backends/", "knowledge/general/libraries/"),
    ("languages/", "knowledge/general/backend/"),
    ("expert_skills/", "knowledge/general/expert_recipes/"),
    ("profiling/", "knowledge/general/profile_bottleneck/"),
    ("case_studies/", "knowledge/general/case_studies/"),
    ("hardware/", "knowledge/general/hardware/"),
    ("landscape/", "knowledge/general/landscape/"),
    ("optimization/", "knowledge/general/optimization/"),
    ("quantization/", "knowledge/general/quantization/"),
    ("reference/", "knowledge/general/reference/"),
    ("workflows/", "knowledge/general/workflows/"),
)
UNMAPPED = "kernelwiki/"            # KernelWiki files no rule covers stay reachable under their own path
POLICY = ("knowledge/policy.yaml", "knowledge/general/rsi/")   # the KnowledgePolicy is not reference knowledge
LEARNED = "/learned_knowledge/"
IMPORT_PREFIXES = ("kernel-", "e2e-", "")   # KernelWiki imported GEAK cards as kernel-<slug> / e2e-<slug>
ALIASED = re.compile(r"knowledge/operator/(?P<op>[^/]+)/backend/(?P<backend>[^/]+)_aiter\.md")


def _apply(path, source, target):
    """Rewrite `path` from prefix `source` to prefix `target`, binding {op} to one segment; None if no match.
    A prefix ending in "/" covers everything below it; any other prefix names exactly one file."""
    pattern = re.escape(source).replace(re.escape("{op}"), "(?P<op>[^/]+)")
    match = re.fullmatch(pattern + ("(?P<rest>.+)" if source.endswith("/") else ""), path)
    if match is None:
        return None
    return target.replace("{op}", match.groupdict().get("op") or "") + (match.groupdict().get("rest") or "")


def to_kernelwiki(view_path):
    """The KernelWiki path a GEAK-layout path names (it may not exist at a given commit)."""
    for view, kernelwiki in RULES:
        if (found := _apply(view_path, view, kernelwiki)) is not None:
            return found
    return "knowledge/" + view_path[len(UNMAPPED):] if view_path.startswith(UNMAPPED) else None


def to_view(kernelwiki_path):
    """The GEAK-layout path under which a KernelWiki file is shown."""
    for view, kernelwiki in sorted(RULES, key=lambda rule: -len(rule[1])):
        if (found := _apply(kernelwiki_path, kernelwiki, view)) is not None:
            return found
    return UNMAPPED + kernelwiki_path.removeprefix("knowledge/")


def resolve(ref, documents):
    """KernelWiki path of a historical GEAK read, or None. `documents` are the KernelWiki paths alive at the
    pinned commit. A bare learned-card slug names the one learned card it was imported as."""
    if "/" not in ref:
        for prefix in IMPORT_PREFIXES:
            found = [path for path in documents if path.endswith(LEARNED + prefix + ref)]
            if found:
                return found[0] if len(found) == 1 else None
        return None
    path = to_kernelwiki(ref)
    return path if path in documents else None


def materialize(repo, commit, destination):
    """Write the GEAK-layout view of KernelWiki `knowledge/` at `commit`, plus manifest.json naming every file's
    KernelWiki path. Reads committed git objects only; the repository is not touched."""
    repo, destination = Path(repo), Path(destination)
    found = subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", commit + "^{commit}"],
                           capture_output=True, text=True)
    if found.returncode:
        raise KnowledgeError(f"KernelWiki must be a git checkout with a committed '{commit}': the view is pinned to it")
    commit = found.stdout.strip()
    archive = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", commit, "--", "knowledge"],
                             check=True, capture_output=True).stdout
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        documents = {member.name: bundle.extractfile(member).read() for member in bundle.getmembers()
                     if member.isfile() and not member.name.startswith(POLICY)}
    shown = {to_view(path): path for path in documents}
    if len(shown) != len(documents):
        raise KnowledgeError("two KernelWiki files map to one view path")
    # Agents are prompted for operators/<op>/backends/<lang>.md. KernelWiki keeps the AITER snapshot of a backend
    # as <lang>_aiter.md; when no <lang>.md exists, the conventional path shows that document too.
    aliases = {}
    for path in documents:
        found = ALIASED.fullmatch(path)
        if found and f"knowledge/operator/{found['op']}/backend/{found['backend']}.md" not in documents:
            aliases[f"operators/{found['op']}/backends/{found['backend']}.md"] = path
    files = {**shown, **aliases}
    for view, path in files.items():
        target = destination / view
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(documents[path])
    manifest = {"schema_version": 1, "repo": str(repo.resolve()), "commit": commit, "files": dict(sorted(files.items())),
                "aliases": dict(sorted(aliases.items()))}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def view_uses(generation_result, view_root):
    """The KernelWiki documents a generation's plan pointed to, named through the view manifest it ran on.
    A pointer outside the view (or a directory) stays unresolved and is reported as such."""
    manifest = json.loads((Path(view_root) / "manifest.json").read_text())
    directions = ((generation_result or {}).get("plan") or {}).get("directions") or []
    uses = []
    for direction in directions:
        for ref in direction.get("kk_refs") or []:
            path = view_path(ref, view_root)
            uses.append({"role": "pointed", "ref": ref, "view_path": path, "kernelwiki": manifest["files"].get(path),
                         "commit": manifest["commit"]})
    return uses


def view_path(ref, view_root):
    """A reference as written by an agent (absolute under the view, or perf_knowledge-relative) -> view path."""
    path, root = str(ref), Path(view_root)
    for prefix in (str(root.resolve()) + "/", str(root) + "/", "perf_knowledge/"):
        if prefix in path:
            path = path.split(prefix, 1)[1]
    return path


def read_uses(events_dir, view_root):
    """Knowledge files agents actually opened, from agent_events stream-json logs: Read and single-file Grep paths,
    and literal view paths in Bash commands, counted only when the tool returned without error. Each names its
    document through the view manifest; a path outside the manifest (a directory, a missing file) stays unresolved."""
    manifest = json.loads((Path(view_root) / "manifest.json").read_text())
    roots = sorted({str(Path(view_root).resolve()), str(view_root)}, key=len, reverse=True)
    literal = re.compile("(?:" + "|".join(re.escape(root) for root in roots) + r")/[^\s'\"`;|&<>()]+")
    uses = []
    for log in sorted(Path(events_dir).glob("*.jsonl")) if Path(events_dir).is_dir() else ():
        lines = log.read_text().splitlines()
        phase = json.loads(lines[0]).get("phase") if lines else None
        calls, failed = {}, set()
        for number, line in enumerate(lines[1:], 2):
            try:
                event = json.loads(line)
            except ValueError:
                continue
            for block in (event.get("message") or {}).get("content") or () if isinstance(event, dict) else ():
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    calls[block.get("id")] = (block.get("name"), block.get("input") or {}, number)
                elif block.get("type") == "tool_result" and block.get("is_error"):
                    failed.add(block.get("tool_use_id"))
        for call_id, (tool, given, number) in calls.items():
            if call_id in failed:
                continue
            if tool == "Read":
                paths = [given.get("file_path", "")]
            elif tool == "Grep":
                paths = [given.get("path", "")]
            elif tool == "Bash":
                paths = literal.findall(given.get("command", ""))
            else:
                continue
            for path in dict.fromkeys(paths):
                if not any(str(path).startswith(root + "/") for root in roots):
                    continue
                view = view_path(path, view_root)
                uses.append({"role": "read", "phase": phase, "tool": tool, "view_path": view,
                             "kernelwiki": manifest["files"].get(view), "commit": manifest["commit"],
                             "events": str(log), "line": number})
    return uses
