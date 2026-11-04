"""KernelWiki page changes proposed as signed, reversible migrations.

Two kinds, each rebuilt and checked byte for byte when a request is made and again when it is published:
- ingest_learned: GEAK learned cards KernelWiki never received. KernelWiki imported GEAK's cards from AMD-AGI/GEAK
  02bfbb0b as `kernel-<slug>.md` under each card's operator; cards whose kernel class has no operator were left
  out. They go under `general/learned_knowledge/` with the same convention: original text, two provenance keys,
  and a footer pinning the GEAK blob, which the KernelWiki projection reads as the source.
- knowledge_iteration: reviewed page revisions, cautions and new learned cards drawn from a frozen replay
  buffer (knowledge_iteration.reproduce).
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from kernelwiki import KnowledgeClient
from kernelwiki.knowledge import EvidenceRef, KnowledgeError

from ..evidence import read_json, reference, verify, write_json
from . import knowledge_iteration

SOURCE_REPOSITORY = "AMD-AGI/GEAK"
LEARNED = "kernel_workflow/knowledge/learned/"
NOT_CARDS = {"README.md", "INDEX.md", "_archive.md"}
TARGET = "knowledge/general/learned_knowledge/kernel-{slug}"
FOOTER = ("\n## Migration provenance\n\nIngested by KernelWiki-RSI on {date} from [{repository} `{short}`]"
          "(https://github.com/{repository}/blob/{commit}/{path}). The 2026-10-21 import left it out because "
          "KernelWiki has no operator page for its kernel class ({kernel_class}), so it is placed under general/. "
          "Performance claims retain their original dates, environments and evidence; this import is not a new "
          "benchmark or a current-best certification.\n")


def git(repo, *args, text=True):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=text).stdout


def kernel_class(source):
    found = re.search(r"^kernel_class:\s*(\S+)", source.decode(errors="replace"), re.M)
    return found[1] if found else "unknown"


def render(source, *, commit, slug, date):
    """GEAK card -> KernelWiki page: provenance keys after the opening '---', footer appended, text unchanged."""
    text = source.decode()
    if not text.startswith("---\n"):
        raise KnowledgeError(f"learned card {slug} has no front matter to extend")
    body = "---\norigin: geak\nevidence_scope: kernel\n" + text[4:]
    body += "" if body.endswith("\n") else "\n"
    return (body + FOOTER.format(date=date, repository=SOURCE_REPOSITORY, short=commit[:12], commit=commit,
                                 path=LEARNED + slug, kernel_class=kernel_class(source))).encode()


def missing_cards(geak, commit, wiki):
    """GEAK learned cards at `commit` that KernelWiki has under none of its import names."""
    cards = [Path(name).name for name in git(geak, "ls-tree", "--name-only", commit, LEARNED).split()
             if name.endswith(".md") and Path(name).name not in NOT_CARDS]
    present = {Path(path).name for path in git(wiki, "ls-tree", "-r", "--name-only", "HEAD", "knowledge").split()
               if "/learned_knowledge/" in path}
    return [slug for slug in cards if not {slug, "kernel-" + slug, "e2e-" + slug} & present]


def propose(geak, commit, wiki, output, *, date=None):
    """Write the proposal: the verbatim GEAK sources, the KernelWiki pages to add, and why each is missing."""
    geak, wiki, destination = Path(geak).resolve(), Path(wiki).resolve(), Path(output).resolve()
    commit = git(geak, "rev-parse", commit).strip()
    date = date or time.strftime("%Y-%m-%d", time.gmtime())
    destination.mkdir(parents=True, exist_ok=False)
    files = []
    for slug in missing_cards(geak, commit, wiki):
        source = git(geak, "show", f"{commit}:{LEARNED}{slug}", text=False)
        (destination / "sources").mkdir(exist_ok=True)
        (destination / "sources" / slug).write_bytes(source)
        path = TARGET.format(slug=slug)
        (destination / "files" / path).parent.mkdir(parents=True, exist_ok=True)
        (destination / "files" / path).write_bytes(render(source, commit=commit, slug=slug, date=date))
        files.append({"path": path, "slug": slug, "kernel_class": kernel_class(source),
                      "source": asdict(reference(destination / "sources" / slug))})
    if not files:
        raise KnowledgeError("every GEAK learned card at this commit is already in KernelWiki")
    value = {"kind": "ingest_learned", "source": {"repository": SOURCE_REPOSITORY, "local": str(geak), "commit": commit},
             "date": date, "wiki": str(wiki), "files": files,
             "rationale": "GEAK learned cards whose kernel class has no KernelWiki operator page were left out of the "
                          "2026-10-21 import; KernelWiki does not hold them under any import name."}
    write_json(destination / "proposal.json", value)
    return value


def reproduce(proposal_dir, wiki):
    """Rebuild every page of a proposal and compare it byte for byte with what was proposed.
    Returns the KernelWiki snapshot, the pages to write, and the evidence."""
    proposal_dir = Path(proposal_dir).resolve()
    value = read_json(proposal_dir / "proposal.json")
    return {"ingest_learned": _reproduce_ingest,
            "knowledge_iteration": knowledge_iteration.reproduce}[value["kind"]](proposal_dir, value, wiki)


def _reproduce_ingest(proposal_dir, value, wiki):
    """Every page comes from the pinned GEAK blob, and every target is still absent from KernelWiki."""
    source = value["source"]
    snapshot = KnowledgeClient(wiki).snapshot()
    present = {name for name, _ in snapshot.files} | {
        path for path in git(wiki, "ls-tree", "-r", "--name-only", "HEAD", "knowledge").split()}
    pages, evidence = [], [reference(proposal_dir / "proposal.json")]
    for item in value["files"]:
        original = verify(EvidenceRef.from_dict(item["source"]))
        if original != git(source["local"], "show", f"{source['commit']}:{LEARNED}{item['slug']}", text=False):
            raise KnowledgeError(f"{item['slug']} differs from GEAK {source['commit'][:12]}")
        page = render(original, commit=source["commit"], slug=item["slug"], date=value["date"])
        if page != (proposal_dir / "files" / item["path"]).read_bytes():
            raise KnowledgeError(f"proposed page for {item['slug']} is not its rendered GEAK source")
        if item["path"] in present:
            raise KnowledgeError(f"{item['path']} already exists in KernelWiki")
        pages.append((item["path"], page))
        evidence.append(EvidenceRef.from_dict(item["source"]))
    return snapshot, tuple(pages), evidence
