"""Iterate KernelWiki's knowledge on a frozen replay buffer.

Code reads the buffer and chooses the work (knowledge_iteration.worklist). For each item a writer drafts the
change from that item's evidence alone, code checks the draft, and a reviewer judges it; a draft that fails is
redrafted once from the failed check or the reviewer's reasons. Cautions are generated without an agent.
Everything accepted is published as one migration on a KernelWiki experiment branch, signed as automatic: no
person reviewed it and nothing was validated online. Running again continues with what is left; KernelWiki
already carries what was published for this buffer, so a finished buffer has an empty worklist.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from kernelwiki import KnowledgeClient
from kernelwiki.knowledge import KnowledgeError

from ..evidence import write_json
from ..runtime import AgentRole, AgentUnavailable
from . import knowledge_iteration, migration

ROLES = {
    "writer": AgentRole({"type": "object", "required": ["rationale"],
                         "properties": {"rationale": {"type": "string"}, "body": {"type": ["string", "null"]},
                                        "citations": {"type": "array"}, "card": {"type": ["object", "null"]}}},
                        Path(__file__).parent / "roles/writer.md", engineering=False),
    "knowledge_reviewer": AgentRole({"type": "object", "required": ["decision", "reasons"],
                                     "properties": {"decision": {"enum": ["approve", "reject"]},
                                                    "reasons": {"type": "array"}}},
                                    Path(__file__).parent / "roles/knowledge_reviewer.md", engineering=False),
}
AUTO_SIGNER = "auto: kernelwiki-rsi iterate (no human review; not validated online)"
PROTECTED = {"", "main", "master"}
DRAFTS = 2       # one redraft, answering the failed check or the reviewer
WORKERS = 4
CONTENT = {"revise": "body", "distill": "card"}


def iterate(pool, wiki, output, *, agent, max_items=None, progress=None):
    wiki, root = Path(wiki).resolve(), Path(output).resolve()
    branch = migration.git(wiki, "branch", "--show-current")
    if branch in PROTECTED:
        raise KnowledgeError("iterate publishes without a person; run it on a KernelWiki experiment branch")
    root.mkdir(parents=True, exist_ok=False)
    evidence = knowledge_iteration.Evidence(pool)
    snapshot = KnowledgeClient(wiki).snapshot()
    files = dict(snapshot.files)
    items = knowledge_iteration.worklist(evidence, files)
    chosen = items[:max_items] if max_items else items
    write_json(root / "worklist.json", {"pool": evidence.pool, "revision": snapshot.revision, "items": items,
                                        "chosen": len(chosen)})
    date = time.strftime("%Y-%m-%d", time.gmtime())

    def work(numbered):
        number, item = numbered
        directory = root / f"items/{number:02d}-{item['op']}"
        outcome = _work(evidence, files, item, directory, agent, date)
        write_json(directory / "outcome.json", outcome)
        if progress:
            progress({"item": number, "op": item["op"], "subject": item["subject"], "status": outcome["status"]})
        return outcome

    with ThreadPoolExecutor(WORKERS) as workers:
        outcomes = list(workers.map(work, enumerate(chosen, 1)))
    accepted, seen = [], set()
    for outcome in outcomes:
        if outcome["status"] == "accepted" and not seen & set(outcome["paths"]):
            accepted.append(outcome)
            seen.update(outcome["paths"])
    value = {"status": "converged" if not chosen else "published" if accepted else "nothing_accepted",
             "branch": branch, "pool": evidence.pool, "worklist": len(items), "chosen": len(chosen),
             "remaining": len(items) - len(chosen), "agent_calls": sum(outcome["agent_calls"] for outcome in outcomes),
             "items": [{key: outcome[key] for key in ("op", "subject", "status", "paths", "reason")}
                       for outcome in outcomes],
             "revision_before": snapshot.revision}
    if accepted:
        _propose(root / "proposal", evidence, date, wiki, files, accepted)
        migration.request(root / "proposal", wiki, root / "request")
        migration.sign(root / "request", AUTO_SIGNER)
        published = migration.publish(root / "request", wiki)
        value.update(commit=published["commit"], revision=published["revision"],
                     undo=f"python3 -m kernel_workflow.rsi migrate switch --request {root / 'request'} --wiki {wiki} "
                          "--to previous")
    write_json(root / "result.json", value)
    return value


def _work(evidence, files, item, directory, agent, date):
    """One item from draft to verdict. Returns its outcome: accepted with its files, or why not."""
    outcome = {**item, "paths": [], "output": None, "review": None, "reason": None, "agent_calls": 0}
    directory.mkdir(parents=True)
    if item["op"] == "caution":
        written = knowledge_iteration.render(evidence, item, files, None, date)
        return {**outcome, "status": "accepted", "paths": [path for path, _ in written]}
    payload, previous = knowledge_iteration.pack(evidence, item, files), None
    write_json(directory / "pack.json", payload)
    try:
        for draft in range(1, DRAFTS + 1):
            output = agent("writer", {**payload, "previous": previous}, directory / f"agents/writer-{draft}")
            outcome["agent_calls"] += 1
            write_json(directory / f"writer-{draft}.json", output)
            if isinstance(output, dict) and output.get(CONTENT[item["op"]], ...) is None:
                return {**outcome, "status": "declined", "reason": output.get("rationale")}
            try:
                written = knowledge_iteration.render(evidence, item, files, output, date)
            except (KnowledgeError, TypeError, KeyError, AttributeError, ValueError) as error:
                previous = {"draft": output, "failed_check": str(error)}
                outcome["reason"] = f"check: {error}"
                continue
            shown = {key: value for key, value in payload.items() if key != "page"}   # the page is `before`
            review = agent("knowledge_reviewer", {"evidence": shown, "change": [
                {"path": path, "before": files[path].decode() if path in files else None, "after": after.decode()}
                for path, after in written]}, directory / f"agents/reviewer-{draft}")
            outcome["agent_calls"] += 1
            write_json(directory / f"review-{draft}.json", review)
            if isinstance(review, dict) and review.get("decision") == "approve":
                return {**outcome, "status": "accepted", "paths": [path for path, _ in written], "output": output,
                        "review": review}
            previous = {"draft": output, "review": review}
            outcome["reason"] = f"review: {review.get('reasons') if isinstance(review, dict) else review}"
    except (AgentUnavailable, KnowledgeError) as error:   # the agent failed or returned an invalid object
        return {**outcome, "status": "agent_failed", "reason": str(error)}
    return {**outcome, "status": "rejected"}


def _propose(destination, evidence, date, wiki, files, accepted):
    """The proposal migration reproduces: each accepted item's draft and review, the files it read and writes."""
    items = [{key: outcome[key] for key in ("op", "subject", "paths", "output", "review")} for outcome in accepted]
    for item in items:
        for path, payload in knowledge_iteration.render(evidence, item, files, item["output"], date):
            for kind, content in (("base", files.get(path)), ("files", payload)):
                if content is not None:
                    (destination / kind / path).parent.mkdir(parents=True, exist_ok=True)
                    (destination / kind / path).write_bytes(content)
    write_json(destination / "proposal.json", {
        "kind": "knowledge_iteration", "pool": evidence.pool, "date": date, "wiki": str(wiki), "items": items,
        "rationale": "Reviewed KernelWiki changes drawn from what recorded GEAK runs did with these pages."})
