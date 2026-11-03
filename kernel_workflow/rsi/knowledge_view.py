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
    pinned commit. A bare learned-card slug names the one learned card with that file name."""
    if "/" not in ref:
        found = [path for path in documents if path.endswith(LEARNED + ref)]
        return found[0] if len(found) == 1 else None
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
    files = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        for member in bundle.getmembers():
            if not member.isfile() or member.name.startswith(POLICY):
                continue
            view = to_view(member.name)
            if view in files:
                raise KnowledgeError(f"two KernelWiki files map to one view path: {view}")
            files[view] = member.name
            target = destination / view
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.extractfile(member).read())
    manifest = {"schema_version": 1, "repo": str(repo.resolve()), "commit": commit, "files": dict(sorted(files.items()))}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def view_uses(generation_result, view_root):
    """The KernelWiki documents a generation's plan pointed to, named through the view manifest it ran on.
    A pointer outside the view (or a directory) stays unresolved and is reported as such."""
    root = Path(view_root)
    files = json.loads((root / "manifest.json").read_text())["files"]
    commit = json.loads((root / "manifest.json").read_text())["commit"]
    directions = ((generation_result or {}).get("plan") or {}).get("directions") or []
    uses = []
    for direction in directions:
        for ref in direction.get("kk_refs") or []:
            path = str(ref)
            for prefix in (str(root.resolve()) + "/", str(root) + "/", "perf_knowledge/"):
                if prefix in path:
                    path = path.split(prefix, 1)[1]
            uses.append({"role": "pointed", "ref": ref, "view_path": path, "kernelwiki": files.get(path),
                         "commit": commit})
    return uses
