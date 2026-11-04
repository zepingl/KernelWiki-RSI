"""What a frozen replay buffer says about KernelWiki, as page changes code can check and reproduce.

Three operations, each about one page or one recorded direction:
- caution: every paired contrast recorded for a page is negative beyond noise (directions that followed it lost
  to siblings in the same round that did not), and the page is not an anti-pattern card. Code puts a banner of
  those numbers under the front matter.
- revise: a page with paired contrasts or heavy use is rewritten by the writer from the directions that used it;
  code keeps the front matter and renders the page's evidence block from the ledger for the cited directions.
- distill: a trunk step that improved without following any page, and whose source diff was archived, becomes
  a learned card; the writer names the technique, code adds the measured effect and provenance.
An agent supplies words only; code chooses the work, supplies every number and checks every result. A page or
direction KernelWiki already carries for this buffer (its stamp or node id is in the text) is not redone.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path

import yaml
from kernelwiki import KnowledgeClient
from kernelwiki.knowledge import KnowledgeError, strict_fields

from ..evidence import read_json, reference
from ..replay_buffer.store import open_buffer
from ..trace.tree import IMPROVED, document_ledger, document_path, fold

CAUTION = ("<!-- kernelwiki-rsi:caution -->", "<!-- /kernelwiki-rsi:caution -->")
EVIDENCE = ("<!-- kernelwiki-rsi:evidence -->", "<!-- /kernelwiki-rsi:evidence -->")
CAUTION_PAIRS = 3          # a caution needs at least this many paired contrasts, all below -IMPROVED
REVISE_PAIRS, REVISE_DIRECTIONS = 3, 10
CITED = 8                  # directions shown to the writer for a page, and the most it may cite
MIN_SIMILARITY = 0.5       # a revised body keeps at least half of the old one
REPORT_CHARS, DIFF_CHARS = 2500, 20000
CARD_FIELDS = {"title", "slug", "operator", "key", "description", "type", "kernel_class", "regime", "keywords",
               "levers", "lever", "apply", "verify", "pitfall"}
CARD_TYPES = {"lever", "method"}


class Evidence:
    """The buffer's page ledger and every recorded direction, with where its report and source diff live."""

    def __init__(self, pool):
        self.pool = str(Path(pool).resolve())
        buffer = open_buffer(self.pool)
        forest = fold(buffer.trace)
        self.ledger = document_ledger(forest)
        attempts = {attempt.id: attempt for attempt in buffer.attempts}
        campaigns = Path(self.pool) / "inputs/campaigns.json"
        runs = {name: Path(item["path"]) for name, item in read_json(campaigns).items()} if campaigns.exists() else {}
        self.directions = {}
        for view in forest.code_trees():
            geak = view.tree.facets.get("geak", {})
            run = runs.get(geak.get("campaign"))
            for node in view.nodes.values():
                expansion = view.expansion(node)
                if expansion is None or expansion.plan is None:
                    continue
                parent = view.nodes[expansion.parents[0]]
                attempt = attempts.get(node.facets.get("geak", {}).get("attempt"))
                self.directions[node.id] = {
                    "node": node.id, "kernel": geak.get("kernel", view.id), "arch": view.tree.task.gfx,
                    "round": expansion.plan.round, "specialty": expansion.plan.slots[node.slot].specialty,
                    "gain": view.gain(node.id), "continued": bool(view.children(node.id)),
                    "knowledge": sorted({document_path(use.target) for use in node.uses}),
                    "report": Path(attempt.input.path).parent / "delivered_result.txt"
                    if attempt and attempt.input else None,
                    "diff": run / "index/source_diffs" / f"{parent.state}__{node.state}.diff"
                    if run and parent.state and node.state else None}

    def stamp(self):
        return f"Replay buffer: `{self.pool}`"


def worklist(evidence, files):
    """Everything the buffer supports that KernelWiki does not carry yet: cautions first (no agent), then
    revisions and distillations interleaved, each by evidence strength."""
    texts = {name: payload.decode(errors="replace") for name, payload in files.items() if name.endswith(".md")}
    cautions, revisions = [], []
    for path, page in evidence.ledger.items():
        if path not in texts or evidence.stamp() in texts[path]:
            continue
        summary = page["summary"]
        if _cautionary(page, texts[path]):
            cautions.append({"op": "caution", "subject": path})
        elif summary["pairs"] >= REVISE_PAIRS or summary["directions"] >= REVISE_DIRECTIONS:
            revisions.append(({"op": "revise", "subject": path}, (summary["pairs"], summary["directions"])))
    revisions = [item for item, _ in sorted(revisions, key=lambda pair: pair[1], reverse=True)]
    written = "\n".join(texts.values())
    distills = [{"op": "distill", "subject": row["node"]} for row in sorted(
        evidence.directions.values(), key=lambda row: row["gain"] or 0, reverse=True)
        if _distillable(row) and f"`{row['node']}`" not in written]
    mixed = [item for pair in zip(revisions, distills) for item in pair]
    shorter = min(len(revisions), len(distills))
    return [*cautions, *mixed, *revisions[shorter:], *distills[shorter:]]


def _cautionary(page, text):
    """Every contrast lost beyond noise. An anti-pattern card is exempt: losing after reading it agrees with it."""
    return (len(page["contrasts"]) >= CAUTION_PAIRS and all(gain < -IMPROVED for gain in page["contrasts"])
            and not re.search(r"^type:\s*anti-pattern\s*$", split(text)[0], re.M))


def _distillable(row):
    return (row["gain"] is not None and row["gain"] > IMPROVED and row["continued"] and not row["knowledge"]
            and row["diff"] is not None and row["diff"].is_file() and row["diff"].stat().st_size > 0)


def pack(evidence, item, files):
    """The one item's evidence the writer sees."""
    if item["op"] == "revise":
        path = item["subject"]
        _, body = split(_unmarked(files[path].decode()))
        return {"op": "revise", "path": path, "page": body, "summary": evidence.ledger[path]["summary"],
                "directions": [_brief(evidence.directions[node], REPORT_CHARS) for node in _shown(evidence, path)]}
    row = evidence.directions[item["subject"]]
    diff = row["diff"].read_text(errors="replace")
    return {"op": "distill", "direction": _brief(row, 2 * REPORT_CHARS), "diff": diff[:DIFF_CHARS],
            "diff_truncated": len(diff) > DIFF_CHARS,
            "operators": sorted({name.split("/")[2] for name in files if name.startswith("knowledge/operator/")})}


def _shown(evidence, path):
    """The page's most negative and most positive scored directions."""
    scored = sorted((row for row in evidence.ledger[path]["directions"] if row["gain"] is not None),
                    key=lambda row: row["gain"])
    return list(dict.fromkeys(row["node"] for row in [*scored[:CITED // 2], *scored[-(CITED // 2):]]))


def _brief(row, chars):
    report = row["report"].read_text(errors="replace")[:chars] if row["report"] and row["report"].is_file() else ""
    return {**{key: row[key] for key in ("node", "kernel", "arch", "round", "specialty", "gain", "continued",
                                         "knowledge")}, "report": report}


def render(evidence, item, files, output, date):
    """The KernelWiki files an item writes, as (path, bytes); KernelError names the first rule a draft breaks."""
    if item["op"] == "caution":
        front, body = split(_unmarked(files[item["subject"]].decode(), CAUTION))
        return ((item["subject"], (front + _caution(evidence, item["subject"]) + "\n\n" + body).encode()),)
    if item["op"] == "revise":
        return (_revise(evidence, item["subject"], files, output),)
    return (_card(evidence, item["subject"], files, output, date),)


def split(text):
    """(front matter with its closing line, body)."""
    if text.startswith("---\n") and (end := text.find("\n---\n", 4)) >= 0:
        return text[:end + 5], text[end + 5:]
    return "", text


def _unmarked(text, *blocks):
    """The text without this module's blocks (default: both), exactly as it was before they were added."""
    for begin, end in blocks or (CAUTION, EVIDENCE):
        if begin in text:
            start, stop = text.index(begin), text.index(end, text.index(begin)) + len(end)
            text = text[:start] + text[stop:].removeprefix("\n\n") if begin == CAUTION[0] else \
                text[:start].rstrip("\n") + "\n" + text[stop:].lstrip("\n")
    return text


def _caution(evidence, path):
    gains = ", ".join(f"{gain:+.1%}" for gain in sorted(evidence.ledger[path]["contrasts"]))
    return (f"{CAUTION[0]}\n> **Caution (KernelWiki-RSI):** in every paired contrast recorded for this page "
            f"({gains}), directions that followed it scored below sibling directions of the same round that did "
            "not. This is an association from recorded GEAK runs, not a cause, and it was not validated online: "
            f"check the page's premise against the current kernel before following it. {evidence.stamp()}.\n"
            f"{CAUTION[1]}")


def _plain(value, name, *, lines=1):
    if not isinstance(value, str) or not value.strip() or len(value.splitlines()) > lines or any(
            marker in value for block in (CAUTION, EVIDENCE) for marker in block):
        raise KnowledgeError(f"{name} must be non-empty plain text" + (" on one line" if lines == 1 else ""))
    return value.strip()


def _revise(evidence, path, files, output):
    strict_fields(output, {"rationale", "body", "citations"}, {"rationale", "body", "citations"})
    front, old = split(_unmarked(files[path].decode()))
    body = _plain(output["body"], "body", lines=10 ** 6)
    if body.startswith("---"):
        raise KnowledgeError("body must not carry front matter; code keeps the page's own")
    similarity = difflib.SequenceMatcher(None, old, body).ratio()
    if similarity < MIN_SIMILARITY:
        raise KnowledgeError(f"body keeps too little of the page (similarity {similarity:.2f} < {MIN_SIMILARITY})")
    citations, used = output["citations"], {row["node"] for row in evidence.ledger[path]["directions"]}
    if not isinstance(citations, list) or not 0 < len(set(citations)) == len(citations) <= CITED \
            or not set(citations) <= used:
        raise KnowledgeError(f"citations must be 1-{CITED} distinct directions that used {path}")
    summary = evidence.ledger[path]["summary"]
    lines = [EVIDENCE[0], "## Evidence (KernelWiki-RSI)", "",
             f"{evidence.stamp()}. Recorded GEAK runs; associations, not causes; not validated online.", "",
             f"- {summary['directions']} recorded directions used this page"
             + (f"; {summary['improved']:.0%} beat their parent by more than {IMPROVED:.1%}" if summary["improved"]
                is not None else "") + f"; {summary['continued']:.0%} were continued by later rounds."]
    if summary["pairs"]:
        lines.append(f"- {summary['pairs']} paired contrasts with sibling directions that did not use it: mean "
                     f"{summary['mean_contrast']:+.1%}, range {summary['min_contrast']:+.1%} to "
                     f"{summary['max_contrast']:+.1%}.")
    lines.append("- Cited directions:")
    for node in citations:
        row = evidence.directions[node]
        gain = "unscored" if row["gain"] is None else f"{row['gain']:+.1%} over its parent"
        lines.append(f"  - `{node}`: {row['kernel']}, round {row['round']}, {row['specialty']}; {gain}"
                     + ("; continued" if row["continued"] else ""))
    lines.append(EVIDENCE[1])
    return path, (front + body + "\n\n" + "\n".join(lines) + "\n").encode()


def _card(evidence, node, files, output, date):
    strict_fields(output, {"rationale", "card"}, {"rationale", "card"})
    card, row = output["card"], evidence.directions[node]
    strict_fields(card, CARD_FIELDS, CARD_FIELDS)
    text = {name: _plain(card[name], name) for name in CARD_FIELDS - {"keywords", "levers"}}
    lists = {name: card[name] for name in ("keywords", "levers")}
    if any(not isinstance(value, list) or not 1 <= len(value) <= 12 or len(set(value)) != len(value)
           or not all(isinstance(entry, str) and entry.strip() and "\n" not in entry for entry in value)
           for value in lists.values()):
        raise KnowledgeError("keywords and levers must be 1-12 distinct one-line strings")
    if text["type"] not in CARD_TYPES:
        raise KnowledgeError(f"type must be one of {sorted(CARD_TYPES)}")
    if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", text["slug"]) or len(text["slug"]) > 70:
        raise KnowledgeError("slug must be lowercase words joined by '-', at most 70 characters")
    if not any(name.startswith(f"knowledge/operator/{text['operator']}/") for name in files):
        raise KnowledgeError(f"operator {text['operator']} is not a KernelWiki operator")
    path = f"knowledge/operator/{text['operator']}/learned_knowledge/kernel-{text['slug']}.md"
    if path in files:
        raise KnowledgeError(f"{path} already exists")
    effect = f"{row['gain']:+.1%} over its parent on {row['kernel']} (one recorded step, round {row['round']})"
    front = {"origin": "kernelwiki-rsi", "evidence_scope": "kernel", "key": text["key"], "type": text["type"],
             "confidence": "★", "effect": effect, "name": text["slug"], "description": text["description"],
             "keywords": lists["keywords"], "kernels": [row["kernel"]], "platforms": [row["arch"]] if row["arch"] else [],
             "kernel_class": text["kernel_class"], "regime": text["regime"], "layer": "learned",
             "lifecycle": "active", "levers": lists["levers"], "origin_kernels": [row["kernel"]]}
    body = (f"# {text['title']}\n- lever: {text['lever']}\n- apply: {text['apply']}\n- verify: {text['verify']}\n"
            f"- pitfall: {text['pitfall']}\n- source: recorded direction `{node}` ({row['kernel']}, round "
            f"{row['round']}, {row['specialty']}): {row['gain']:+.1%} over its parent, continued by later rounds; "
            f"source diff `{row['diff']}`\n\n## Provenance\n\nDistilled by KernelWiki-RSI on {date} from one "
            f"recorded step that followed no KernelWiki page. {evidence.stamp()}. The effect is that step's recorded "
            "measurement on one kernel, not a new benchmark, and the card was not validated online.\n")
    dumped = yaml.safe_dump(front, sort_keys=False, allow_unicode=True, width=10 ** 6, default_flow_style=None)
    return path, f"---\n{dumped}---\n{body}".encode()


def reproduce(proposal_dir, value, wiki):
    """Every file is re-rendered from the recorded drafts against the KernelWiki it was written for, equals the
    proposal byte for byte, and every agent-written item was approved by the reviewer."""
    proposal_dir = Path(proposal_dir).resolve()
    evidence, snapshot = Evidence(value["pool"]), KnowledgeClient(wiki).snapshot()
    files, pages = dict(snapshot.files), []
    for item in value["items"]:
        if item["op"] != "caution" and (item["review"] or {}).get("decision") != "approve":
            raise KnowledgeError(f"{item['subject']} was not approved by the reviewer")
        for path, payload in render(evidence, item, files, item["output"], value["date"]):
            base = proposal_dir / "base" / path
            if files.get(path) != (base.read_bytes() if base.exists() else None):
                raise KnowledgeError(f"{path} changed since its item was written")
            if payload != (proposal_dir / "files" / path).read_bytes():
                raise KnowledgeError(f"proposed {path} is not its rendered item")
            pages.append((path, payload))
    if len({path for path, _ in pages}) != len(pages):
        raise KnowledgeError("two items write the same file")
    return snapshot, tuple(pages), [reference(proposal_dir / "proposal.json")]
