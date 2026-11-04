"""Evidence-gated KernelWiki migration: request, human approval, publish, rollback and restore.

A change is either a KnowledgePolicy an inner round selected on frozen replay evidence, or knowledge pages
proposed from a pinned source (knowledge_change). A request reproduces the change's gate and lays out exactly what
would move; a person signs it; publication reproduces the gate again, moves KernelWiki with compare-and-swap,
commits the managed `knowledge/` files to git and appends the gate result to each moved knowledge version.
Rollback and restore move between the two revisions of one publication, each as a commit. Nothing here claims
online improvement: fresh holdout and paired online gates stay false.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from kernelwiki import KnowledgeClient
from kernelwiki.knowledge import EvidenceRef, KnowledgeError, canonical
from kernelwiki.knowledge_store import KnowledgeUpdate

from ..evidence import read_json, read_jsonl, reference, verify, verify_artifacts, write_json, write_jsonl
from ..harness import Policy
from ..replay_buffer.cohorts import cohorts
from ..replay_buffer.evaluate import evaluate, qualifies
from ..replay_buffer.objective import ReplayObjective
from ..replay_buffer.store import open_buffer
from ..reverse_engineering.sources.kernelwiki import kernelwiki_trees, knowledge_index
from ..trace.records import Decide, Measure, from_wire, to_wire
from ..trace.tree import fold

from . import knowledge_change

SCOPE = "kernelwiki-policy-migration"
LIMITS = ["Offline replay on recorded episodes only: no fresh holdout and no paired online gate were run.",
          "Each round is scored where history actually was; the policy cannot change later states it would reach.",
          "No GEAK lane reads a schema-2 policy yet; the parent scheduler refuses it explicitly."]
KNOWLEDGE_GATES = {"ingest_learned": "reproduce", "knowledge_iteration": "evidence_gate"}
ITERATION_LIMITS = ["Page evidence summarizes recorded GEAK runs (history read GEAK's copy of these pages); "
                    "it is an association, not causal attribution.",
                    "Text changes cannot be scored by replay: whether agents do better with them is measured by a "
                    "later online comparison of the old and new KernelWiki.",
                    "A new card generalizes one recorded step on one kernel.",
                    "Agent-written items were judged by the reviewer agent; iterate publishes them without a person."]
INGEST_LIMITS = ["Adds knowledge GEAK already held; nothing was newly measured.",
                 "Placed under general/learned_knowledge because KernelWiki has no operator page for these kernel classes.",
                 "No online gate: whether agents use these cards well is measured by later runs."]


def git(wiki, *args):
    return subprocess.run(["git", "-C", str(wiki), *args], check=True, capture_output=True, text=True).stdout.strip()


def reproduce(evolve_dir, wiki):
    """Re-run the selection gate from the evolve round's frozen copies and require identical replay results."""
    evolve_dir = Path(evolve_dir).resolve()
    result = read_json(evolve_dir / "result.json")
    if result["status"] != "working_policy_selected":
        raise KnowledgeError("only a pooled episode round that selected a working policy can request migration")
    verify_artifacts(evolve_dir, result)
    frozen = read_json(evolve_dir / "manifest.json")
    if not frozen.get("episodes"):
        raise KnowledgeError("only a pooled episode round that selected a working policy can request migration")
    if reference(evolve_dir / "frozen_pool/manifest.json").sha256 != frozen["pool"]["sha256"]:
        raise KnowledgeError("frozen pool copy differs from the round manifest")
    snapshot = KnowledgeClient(wiki).snapshot()
    if snapshot.revision != frozen["knowledge_revision"]:
        raise KnowledgeError("KernelWiki moved since the round froze its knowledge; run a new round")
    buffer = open_buffer(evolve_dir / "frozen_pool")
    budgets, objective = tuple(frozen["budgets"]), ReplayObjective(**frozen["objective"])
    fixed = cohorts(buffer.trees, budgets, traces="trace.jsonl", episodes=True)
    if fixed != frozen["cohorts"]:
        raise KnowledgeError("episode cohort no longer reproduces")
    primary = tuple(tree for tree in buffer.trees if tree.id in set(fixed["primary"]))
    incumbent = Policy.from_dict(read_json(evolve_dir / "incumbent.json"), snapshot.records)
    candidate = Policy.from_dict(read_json(evolve_dir / "selected-policy.json"), snapshot.records)
    before = evaluate(primary, incumbent, budgets, objective, episodes=frozen["episodes"])
    after = evaluate(primary, candidate, budgets, objective, episodes=frozen["episodes"])
    recorded = evolve_dir / f"candidate-{result['selected']}-train.json"
    if canonical(before) != canonical(read_json(evolve_dir / "incumbent-train.json")) or \
            canonical(after) != canonical(read_json(recorded)):
        raise KnowledgeError("replay results do not reproduce from the frozen inputs")
    if not (qualifies(after, before, held_out=True)
            and after["summary"]["objective"] > before["summary"]["objective"] + 1e-12):
        raise KnowledgeError("candidate no longer passes the pooled replay gate")
    evidence = [reference(path) for path in (evolve_dir / "result.json", evolve_dir / "manifest.json",
                                             evolve_dir / "incumbent-train.json", recorded,
                                             evolve_dir / f"review-{result['selected']}.json")]
    return snapshot, incumbent, candidate, before, after, evidence


@dataclass(frozen=True)
class Change:
    """One reproducible KernelWiki change and the facts a person signs."""
    kind: str
    source: Path
    snapshot: object
    update: KnowledgeUpdate          # its evidence excludes the request and the approval
    paths: tuple[str, ...]           # the KernelWiki files it moves
    gate: str                        # measurer recorded on each version it creates
    summary: dict
    title: str
    limits: tuple[str, ...]


def change(source, wiki):
    source = Path(source).resolve()
    if (source / "proposal.json").exists():
        snapshot, pages, evidence = knowledge_change.reproduce(source, wiki)
        value = read_json(source / "proposal.json")
        update = KnowledgeUpdate(snapshot.policy, (), {"status": "evidence_update", "kind": value["kind"],
                                                       "gates": {KNOWLEDGE_GATES[value["kind"]]: True,
                                                                 "signed_approval": True, "online": False}},
                                 tuple(evidence), pages)
        if value["kind"] == "ingest_learned":
            summary = {"source_repository": value["source"],
                       "changes": [{"path": item["path"], "change": "added", "slug": item["slug"],
                                    "kernel_class": item["kernel_class"]} for item in value["files"]],
                       "gate": {"rule": "each page re-rendered from the pinned GEAK blob equals the proposal byte for "
                                        "byte; no target exists in KernelWiki"}}
            title, limits = f"ingest {len(pages)} GEAK learned cards", INGEST_LIMITS
        else:
            changes = [{"path": path, "change": item["op"], "subject": item["subject"],
                        "citations": (item["output"] or {}).get("citations"),
                        "review": (item["review"] or {}).get("reasons")}
                       for item in value["items"] for path in item["paths"]]
            summary = {"pool": value["pool"], "changes": changes,
                       "gate": {"rule": "each file re-renders from its recorded draft and the buffer's ledger; every "
                                        "number is generated; every citation used the page; the reviewer approved "
                                        "every agent-written item; cautions are generated from the numbers alone"}}
            counts = {op: sum(item["op"] == op for item in value["items"]) for op in ("revise", "caution", "distill")}
            title = (f"knowledge iteration: revise {counts['revise']} pages, caution {counts['caution']}, "
                     f"distill {counts['distill']} cards")
            limits = ITERATION_LIMITS
        return Change(value["kind"], source, snapshot, update, tuple(path for path, _ in pages),
                      KNOWLEDGE_GATES[value["kind"]], {**summary, "rationale": value["rationale"]}, title, tuple(limits))
    snapshot, incumbent, candidate, before, after, evidence = reproduce(source, wiki)
    previous = {record.id: record for record in snapshot.records}
    changes = [{"id": record.id, "change": "added" if record.id not in previous else "revised",
                "title": record.title, "policy_rule": asdict(record.policy_rule), "body": record.body,
                "evidence": [asdict(item) for item in record.evidence]}
               for record in candidate.records if record.policy_rule and previous.get(record.id) != record]
    kernels = lambda result: {name: {"objective": row["objective"], "full_budget_quality": row["full_budget_quality"]}
                              for name, row in result["summary"]["kernels"].items()}
    update = KnowledgeUpdate(candidate.knowledge, tuple(record for record in candidate.records if record.policy_rule),
                             {"status": "replay_validated", "policy_revision": candidate.revision,
                              "gates": {"pooled_replay": True, "agent_review": True, "human_approval": True,
                                        "fresh_holdout": False, "paired_online": False}, "limits": LIMITS},
                             tuple(evidence))
    summary = {"policy": {"from": incumbent.knowledge.to_dict(), "to": candidate.knowledge.to_dict(),
                          "policy_revision": candidate.revision},
               "changes": changes,
               "replay_gate": {"rule": "higher pooled Objective; pooled and per-kernel full-budget quality not lower; "
                                       "every trace support-complete",
                               "incumbent": {"objective": before["summary"]["objective"],
                                             "full_budget_quality": before["summary"]["full_budget_quality"],
                                             "kernels": kernels(before)},
                               "candidate": {"objective": after["summary"]["objective"],
                                             "full_budget_quality": after["summary"]["full_budget_quality"],
                                             "kernels": kernels(after)}}}
    return Change("policy", source, snapshot, update, ("knowledge/policy.yaml",), "replay_gate", summary,
                  f"migrate KnowledgePolicy to {candidate.revision[:12]}", tuple(LIMITS))


def request(source, wiki, output):
    """Write the migration request a person reviews, and an unsigned approval bound to it."""
    wiki = Path(wiki).resolve()
    moved = change(source, wiki)
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    value = {"scope": SCOPE, "kind": moved.kind, "source": str(moved.source), "wiki": str(wiki),
             "branch": git(wiki, "branch", "--show-current"), "git_head": git(wiki, "rev-parse", "HEAD"),
             "expected_revision": moved.snapshot.revision, **moved.summary,
             "evidence": [asdict(item) for item in moved.update.evidence], "limits": list(moved.limits),
             "drill": "after publishing: roll back to expected_revision, then restore the published revision"}
    write_json(destination / "request.json", value)
    write_json(destination / "approval.json", {"scope": SCOPE, "request": asdict(reference(destination / "request.json")),
                                               "decision": "pending", "approved_by": "", "approved_at": None})
    return value


def sign(request_dir, approved_by, decision="approve"):
    """Record a person's decision on exactly this request; a signed approval is never overwritten."""
    path = Path(request_dir).resolve() / "approval.json"
    approval = read_json(path)
    if approval["decision"] != "pending" or decision not in {"approve", "reject"} or not approved_by:
        raise KnowledgeError("approval is already decided, or the decision or signer is missing")
    verify(EvidenceRef.from_dict(approval["request"]))
    approval.update(decision=decision, approved_by=approved_by, approved_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    os.replace(path, path.with_suffix(".pending.json"))
    write_json(path, approval)
    return approval


def _commit(wiki, message):
    git(wiki, "add", "-A", "--", "knowledge")
    environment = {**os.environ, "GIT_AUTHOR_NAME": "KernelWiki-RSI", "GIT_AUTHOR_EMAIL": "kernelwiki-rsi@invalid",
                   "GIT_COMMITTER_NAME": "KernelWiki-RSI", "GIT_COMMITTER_EMAIL": "kernelwiki-rsi@invalid"}
    subprocess.run(["git", "-C", str(wiki), "commit", "-q", "-m", message, "--", "knowledge"],
                   check=True, capture_output=True, text=True, env=environment)
    return git(wiki, "rev-parse", "HEAD")


def _versions(wiki, commit, paths, prior=()):
    """The projected live version of each moved file at a commit (None once deleted), checked against the file
    on disk and folded with every gate record already appended."""
    records = kernelwiki_trees(wiki, commit)
    live, forest = knowledge_index(records), fold([*records, *prior])
    versions = {}
    for path in paths:
        on_disk = Path(wiki) / path
        if not on_disk.exists():
            if path in live:
                raise KnowledgeError(f"{path} was removed but its version is still live in the projection")
            versions[path] = None
            continue
        node = live[path]
        tree = next(view for view in forest.trees.values() if node in view.nodes)
        if tree.nodes[node].state != hashlib.sha256(on_disk.read_bytes()).hexdigest():
            raise KnowledgeError(f"projected version of {path} differs from the published file")
        versions[path] = (tree.id, tree.tree.contract, node)
    return records, forest, versions


def _gate(kind, gate, versions, evidence, approval):
    return [record for version in versions.values() if version for record in (
        Measure(version[2], version[1], gate, True, None, tuple(evidence)),
        Decide(version[0], version[2], "accepted", f"{kind} gate reproduced; human approval by {approval['approved_by']}"))]


def publish(request_dir, wiki):
    request_dir, wiki = Path(request_dir).resolve(), Path(wiki).resolve()
    destination = request_dir / "publish"
    if destination.exists():
        raise KnowledgeError("this request was already published; use switch to move between its revisions")
    value, approval = read_json(request_dir / "request.json"), read_json(request_dir / "approval.json")
    if approval["request"] != asdict(reference(request_dir / "request.json")):
        raise KnowledgeError("approval was signed for a different request")
    if approval["decision"] != "approve" or not approval["approved_by"] or str(wiki) != value["wiki"]:
        raise KnowledgeError("migration needs a signed approval for this KernelWiki checkout")
    moved = change(value["source"], wiki)
    if moved.snapshot.revision != value["expected_revision"] or \
            [asdict(item) for item in moved.update.evidence] != value["evidence"]:
        raise KnowledgeError("request no longer matches the reproduced gate")
    approvals = [reference(request_dir / "request.json"), reference(request_dir / "approval.json")]
    update = replace(moved.update, evidence=(*approvals, *moved.update.evidence),
                     validation={**moved.update.validation, "approved_by": approval["approved_by"]})
    revision = KnowledgeClient(wiki).publish(update, expected_revision=value["expected_revision"])
    commit = _commit(wiki, f"rsi: {moved.title}\n\nKnowledgeRevision: {revision}\nPrevious: {value['expected_revision']}\n"
                           f"Approved-by: {approval['approved_by']}\nRequest: {approvals[0].path}@{approvals[0].sha256}\n"
                           f"Gate: {moved.gate}; no fresh holdout or online gate")
    _, _, versions = _versions(wiki, commit, moved.paths)
    trace = _gate(moved.kind, moved.gate, versions, update.evidence, approval)
    _versions(wiki, commit, moved.paths, trace)
    destination.mkdir()
    write_jsonl(destination / "trace.jsonl", [to_wire(record) for record in trace])
    result = {"status": "published", "kind": moved.kind, "gate": moved.gate, "previous_revision": value["expected_revision"],
              "revision": revision, "commit": commit, "approved_by": approval["approved_by"],
              "versions": {path: version[2] for path, version in versions.items() if version},
              "trace": asdict(reference(destination / "trace.jsonl"))}
    write_json(destination / "result.json", result)
    return result


def switch(request_dir, wiki, to):
    """Move KernelWiki between the two revisions of one publication; record the move in git and the trace.
    Every move lives under the request, so the publication's whole trace folds in order."""
    request_dir, wiki = Path(request_dir).resolve(), Path(wiki).resolve()
    published = read_json(request_dir / "publish/result.json")
    target, expected = {"previous": (published["previous_revision"], published["revision"]),
                        "published": (published["revision"], published["previous_revision"])}[to]
    traces = [request_dir / "publish/trace.jsonl", *sorted(request_dir.glob("switch-*/trace.jsonl"))]
    prior = [from_wire(value) for path in traces for value in read_jsonl(path)]
    destination = request_dir / f"switch-{len(traces):02d}-{to}"
    destination.mkdir()
    revision = KnowledgeClient(wiki).rollback(target, expected_revision=expected)
    commit = _commit(wiki, f"rsi: {'roll back' if to == 'previous' else 'restore'} {published['kind']} to {target[:12]}\n\n"
                           f"KnowledgeRevision: {revision}\nPrevious: {expected}\nPublication: {published['commit']}")
    paths = tuple(published["versions"])
    _, forest, versions = _versions(wiki, commit, paths, prior)
    if to == "previous":
        retired = {item.node for view in forest.trees.values() for item in view.decisions if item.verdict == "retired"}
        trace = [Decide(next(view.id for view in forest.trees.values() if node in view.nodes), node, "retired",
                        f"rolled back in {commit[:12]}")
                 for node in published["versions"].values() if node not in retired]
    else:
        # Restoring re-applies the signed revision; its gate evidence is the original request, approval and gate.
        value, approval = read_json(request_dir / "request.json"), read_json(request_dir / "approval.json")
        evidence = [reference(request_dir / "request.json"), reference(request_dir / "approval.json"),
                    *(EvidenceRef.from_dict(item) for item in value["evidence"])]
        trace = _gate(published["kind"], published["gate"], versions, evidence, approval)
    _versions(wiki, commit, paths, [*prior, *trace])
    write_jsonl(destination / "trace.jsonl", [to_wire(record) for record in trace])
    result = {"status": f"switched_to_{to}", "revision": revision, "commit": commit,
              "versions": {path: version[2] if version else None for path, version in versions.items()},
              "trace": asdict(reference(destination / "trace.jsonl"))}
    write_json(destination / "result.json", result)
    return result
