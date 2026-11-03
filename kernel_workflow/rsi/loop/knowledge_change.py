"""KernelWiki page changes proposed as signed, reversible migrations.

Two kinds, each rebuilt and checked byte for byte when a request is made and again when it is published:
- ingest_learned: GEAK learned cards KernelWiki never received. KernelWiki imported GEAK's cards from AMD-AGI/GEAK
  02bfbb0b as `kernel-<slug>.md` under each card's operator; cards whose kernel class has no operator were left
  out. They go under `general/learned_knowledge/` with the same convention: original text, two provenance keys,
  and a footer pinning the GEAK blob, which the KernelWiki projection reads as the source.
- amend_pages: an inner round's reviewed field evidence for pages runs actually used, kept in one marked block
  at the end of each page; the rest of the page is untouched, and every citation names a direction that used it.
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
from ..replay_buffer.store import open_buffer
from ..trace.tree import document_ledger, fold

SOURCE_REPOSITORY = "AMD-AGI/GEAK"
LEARNED = "kernel_workflow/knowledge/learned/"
NOT_CARDS = {"README.md", "INDEX.md", "_archive.md"}
SECTION_BEGIN = "<!-- kernelwiki-rsi:field-evidence -->"
SECTION_END = "<!-- /kernelwiki-rsi:field-evidence -->"
SECTION_LIMIT = 1500
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
    return {"ingest_learned": _reproduce_ingest, "amend_pages": _reproduce_amend}[value["kind"]](proposal_dir, value, wiki)


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


def amend(page, section):
    """The page with its field-evidence block replaced (or appended); everything else stays byte for byte."""
    text = page.decode()
    if SECTION_BEGIN in text:
        start = text.index(SECTION_BEGIN)
        end = text.index(SECTION_END, start) + len(SECTION_END)
        start -= text[:start].endswith("\n\n")   # the separator line this function added with the block
        text = text[:start] + text[end:].lstrip("\n")
    text += "" if text.endswith("\n") else "\n"
    return f"{text}\n{SECTION_BEGIN}\n{section.strip()}\n{SECTION_END}\n".encode()


def check_pages(pages, ledger, documents):
    """A field-evidence proposal may touch only pages runs used, cite only directions that used that page, and add
    plain text: no front matter, no markers, no policy, at most SECTION_LIMIT characters."""
    if not isinstance(pages, list) or len({page.get("path") for page in pages if isinstance(page, dict)}) != len(pages):
        raise KnowledgeError("pages must be a list naming each page once")
    for page in pages:
        if set(page) != {"path", "section", "citations"}:
            raise KnowledgeError("a page amendment has exactly path, section and citations")
        path, section, citations = page["path"], page["section"], page["citations"]
        if path not in documents or path not in ledger:
            raise KnowledgeError(f"{path} is not a KernelWiki page that recorded runs used")
        if not isinstance(section, str) or not 0 < len(section.strip()) <= SECTION_LIMIT or any(
                line.strip() in {"---", SECTION_BEGIN, SECTION_END} or line.lstrip().startswith("kernelwiki:")
                for line in section.splitlines()):
            raise KnowledgeError(f"field evidence for {path} must be 1-{SECTION_LIMIT} characters of plain text")
        used = {item["node"] for item in ledger[path]["directions"]}
        if not isinstance(citations, list) or not citations or not set(citations) <= used:
            raise KnowledgeError(f"field evidence for {path} must cite directions that used it")


def round_evidence(evolve_dir):
    """The document ledger of the round's frozen pool and the KernelWiki pages the round saw."""
    evolve_dir = Path(evolve_dir).resolve()
    ledger = document_ledger(fold(open_buffer(evolve_dir / "frozen_pool").trace))
    root = evolve_dir / "frozen_knowledge"
    documents = {str(path.relative_to(root)) for path in root.rglob("*.md")}
    return ledger, documents


def propose_amend(evolve_dir, output):
    """Turn an inner round's reviewed page amendments into a proposal against the pages that round saw."""
    evolve_dir, destination = Path(evolve_dir).resolve(), Path(output).resolve()
    pages = read_json(evolve_dir / "pages.json")["pages"]
    ledger, documents = round_evidence(evolve_dir)
    check_pages(pages, ledger, documents)
    destination.mkdir(parents=True, exist_ok=False)
    for page in pages:
        target = destination / "files" / page["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(amend((evolve_dir / "frozen_knowledge" / page["path"]).read_bytes(), page["section"]))
    value = {"kind": "amend_pages", "evolve": str(evolve_dir), "files": pages,
             "rationale": "Field evidence a reviewed inner round wrote for pages recorded runs used."}
    write_json(destination / "proposal.json", value)
    return value


def _reproduce_amend(proposal_dir, value, wiki):
    """Each page still equals the version the round saw, and its amendment is that version plus the block."""
    evolve_dir = Path(value["evolve"])
    ledger, documents = round_evidence(evolve_dir)
    check_pages(value["files"], ledger, documents)
    snapshot, pages = KnowledgeClient(wiki).snapshot(), []
    current = dict(snapshot.files)
    for page in value["files"]:
        seen = (evolve_dir / "frozen_knowledge" / page["path"]).read_bytes()
        if current.get(page["path"]) != seen:
            raise KnowledgeError(f"{page['path']} changed since the round read it")
        amended = amend(seen, page["section"])
        if amended != (proposal_dir / "files" / page["path"]).read_bytes():
            raise KnowledgeError(f"proposed {page['path']} is not the round's page plus its field evidence")
        pages.append((page["path"], amended))
    evidence = [reference(proposal_dir / "proposal.json"), reference(evolve_dir / "result.json"),
                reference(evolve_dir / "pages.json")]
    return snapshot, tuple(pages), evidence
