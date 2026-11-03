"""Iterate KernelWiki-RSI on a frozen replay buffer until KernelWiki stops changing.

Each round evolves against the current KernelWiki, then publishes what passed its gate: the policy first (pooled
replay gate), then page field evidence (deterministic citation checks plus the reviewer). A round that publishes
nothing ends the loop. Only a KernelWiki branch other than main is written; every publication is one git commit
and one KernelWiki revision, so `migrate switch` can undo any round. Approvals say no person reviewed them.
"""

from __future__ import annotations

from pathlib import Path

from kernelwiki import KnowledgeClient
from kernelwiki.knowledge import KnowledgeError

from ..evidence import write_json
from . import knowledge_change, migration
from .inner import evolve

AUTO_SIGNER = "auto: kernelwiki-rsi iterate (no human review)"
PROTECTED = {"", "main", "master"}


def iterate(pool, wiki, output, *, rounds, agent=None, max_candidates=4, progress=None):
    wiki, root = Path(wiki).resolve(), Path(output).resolve()
    branch = migration.git(wiki, "branch", "--show-current")
    if branch in PROTECTED:
        raise KnowledgeError("iterate publishes without a person; run it on a KernelWiki experiment branch")
    if type(rounds) is not int or rounds < 1:
        raise KnowledgeError("rounds must be a positive integer")
    root.mkdir(parents=True, exist_ok=False)
    chain = [_version(wiki, 0, "start")]
    summary, status = [], "round_limit"
    for number in range(1, rounds + 1):
        here = root / f"round-{number:02d}"
        result = evolve(pool, wiki, here / "evolve", agent=agent, max_candidates=max_candidates, episodes="step",
                        progress=progress)
        published = []
        if result["status"] == "working_policy_selected":
            published.append(_publish(here / "evolve", here / "policy", wiki))
        if result.get("pages"):
            knowledge_change.propose_amend(here / "evolve", here / "pages-proposal")
            published.append(_publish(here / "pages-proposal", here / "pages", wiki))
        chain += [_version(wiki, number, item["kind"]) for item in published]
        summary.append({"round": number, "evolve": result["status"], "pages": result.get("pages", []),
                        "published": published})
        if progress:
            progress({"round": number, "published": [item["kind"] for item in published]})
        if not published:
            status = "converged"
            break
    value = {"status": status, "branch": branch, "rounds": summary, "chain": chain,
             "undo": "python3 -m kernel_workflow.rsi migrate switch --request <round>/<policy|pages> --wiki "
                     f"{wiki} --to previous, newest publication first"}
    write_json(root / "result.json", value)
    return value


def _publish(source, request_dir, wiki):
    migration.request(source, wiki, request_dir)
    migration.sign(request_dir, AUTO_SIGNER)
    published = migration.publish(request_dir, wiki)
    return {"kind": published["kind"], "request": str(request_dir), "commit": published["commit"],
            "revision": published["revision"], "versions": published["versions"]}


def _version(wiki, number, cause):
    return {"round": number, "cause": cause, "commit": migration.git(wiki, "rev-parse", "HEAD"),
            "revision": KnowledgeClient(wiki).snapshot().revision}
