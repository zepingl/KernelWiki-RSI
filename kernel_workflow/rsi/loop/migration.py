"""Replay-evidenced KernelWiki migration: request, human approval, publish, rollback and restore.

The inner loop selects a policy on frozen replay evidence. A migration request reproduces that gate and lays
out exactly what would change; a person signs it; publication reproduces the gate again, moves KernelWiki with
compare-and-swap, commits the managed `knowledge/` files to git and appends the gate result to the policy's
knowledge tree. Rollback and restore move between the two revisions of one publication, each as a commit.
This path never claims online improvement: fresh holdout and paired online gates stay false.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import time
from dataclasses import asdict
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
from ..reverse_engineering.sources.kernelwiki import kernelwiki_trees
from ..trace.records import Decide, Measure, from_wire, to_wire
from ..trace.tree import fold

SCOPE = "kernelwiki-policy-migration"
POLICY_TREE = "kernelwiki/policy"
POLICY_NODE = "kernelwiki:knowledge/policy.yaml@{}"
LIMITS = ["Offline replay on recorded episodes only: no fresh holdout and no paired online gate were run.",
          "Each round is scored where history actually was; the policy cannot change later states it would reach.",
          "No GEAK lane reads a schema-2 policy yet; the parent scheduler refuses it explicitly."]


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


def request(evolve_dir, wiki, output):
    """Write the migration request a person reviews, and an unsigned approval bound to it."""
    wiki = Path(wiki).resolve()
    snapshot, incumbent, candidate, before, after, evidence = reproduce(evolve_dir, wiki)
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    previous = {record.id: record for record in snapshot.records}
    changes = [{"id": record.id, "change": "added" if record.id not in previous else "revised",
                "title": record.title, "policy_rule": asdict(record.policy_rule), "body": record.body,
                "evidence": [asdict(item) for item in record.evidence]}
               for record in candidate.records if record.policy_rule and previous.get(record.id) != record]
    kernels = lambda result: {name: {"objective": row["objective"], "full_budget_quality": row["full_budget_quality"]}
                              for name, row in result["summary"]["kernels"].items()}
    value = {"scope": SCOPE, "evolve": str(Path(evolve_dir).resolve()), "wiki": str(wiki),
             "branch": git(wiki, "branch", "--show-current"),
             "git_head": git(wiki, "rev-parse", "HEAD"), "expected_revision": snapshot.revision,
             "policy": {"from": incumbent.knowledge.to_dict(), "to": candidate.knowledge.to_dict(),
                        "policy_revision": candidate.revision},
             "changes": changes,
             "replay_gate": {"rule": "higher pooled Objective; pooled and per-kernel full-budget quality not lower; "
                                     "every trace support-complete",
                             "incumbent": {"objective": before["summary"]["objective"],
                                           "full_budget_quality": before["summary"]["full_budget_quality"],
                                           "kernels": kernels(before)},
                             "candidate": {"objective": after["summary"]["objective"],
                                           "full_budget_quality": after["summary"]["full_budget_quality"],
                                           "kernels": kernels(after)}},
             "evidence": [asdict(item) for item in evidence], "limits": LIMITS,
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


def _policy_node(wiki, commit, prior_trace=()):
    """The projected policy version at a commit, folded with every gate record already appended."""
    records = kernelwiki_trees(wiki, commit)
    node = POLICY_NODE.format(commit[:12])
    state = hashlib.sha256((Path(wiki) / "knowledge/policy.yaml").read_bytes()).hexdigest()
    forest = fold([*records, *prior_trace])
    if forest.trees[POLICY_TREE].nodes[node].state != state:
        raise KnowledgeError("projected policy version differs from the published file")
    return records, node, state


def _gate(node, evidence, approval):
    return [Measure(node, "kernelwiki:policy", "replay_gate", True, None, tuple(evidence)),
            Decide(POLICY_TREE, node, "accepted",
                   f"pooled replay gate reproduced; reviewer approved; human approval by {approval['approved_by']}")]


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
    snapshot, _, candidate, _, _, evidence = reproduce(value["evolve"], wiki)
    if snapshot.revision != value["expected_revision"] or \
            [asdict(item) for item in evidence] != value["evidence"]:
        raise KnowledgeError("request no longer matches the reproduced gate")
    approvals = [reference(request_dir / "request.json"), reference(request_dir / "approval.json")]
    update = KnowledgeUpdate(candidate.knowledge, tuple(record for record in candidate.records if record.policy_rule),
                             {"status": "replay_validated", "policy_revision": candidate.revision,
                              "gates": {"pooled_replay": True, "agent_review": True, "human_approval": True,
                                        "fresh_holdout": False, "paired_online": False},
                              "approved_by": approval["approved_by"], "limits": LIMITS},
                             (*approvals, *evidence))
    revision = KnowledgeClient(wiki).publish(update, expected_revision=value["expected_revision"])
    commit = _commit(wiki, f"rsi: migrate KnowledgePolicy to {candidate.revision[:12]}\n\n"
                           f"KnowledgeRevision: {revision}\nPrevious: {value['expected_revision']}\n"
                           f"Approved-by: {approval['approved_by']}\nRequest: {approvals[0].path}@{approvals[0].sha256}\n"
                           f"Gates: pooled replay, agent review, human approval; no fresh holdout or online gate")
    _, node, state = _policy_node(wiki, commit)
    trace = _gate(node, [*approvals, *evidence], approval)
    _policy_node(wiki, commit, trace)
    destination.mkdir()
    write_jsonl(destination / "trace.jsonl", [to_wire(record) for record in trace])
    result = {"status": "published", "previous_revision": value["expected_revision"], "revision": revision,
              "commit": commit, "policy_node": node, "policy_state": state, "approved_by": approval["approved_by"],
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
    client = KnowledgeClient(wiki)
    revision = client.rollback(target, expected_revision=expected)
    commit = _commit(wiki, f"rsi: {'roll back' if to == 'previous' else 'restore'} KnowledgePolicy to {target[:12]}\n\n"
                           f"KnowledgeRevision: {revision}\nPrevious: {expected}\n"
                           f"Publication: {published['commit']}")
    _, node, state = _policy_node(wiki, commit, prior)
    if to == "previous":
        trace = [Decide(POLICY_TREE, published["policy_node"], "retired", f"rolled back in {commit[:12]}")]
    else:
        approval = read_json(request_dir / "approval.json")
        evidence = [EvidenceRef.from_dict(item) for item in read_json(request_dir / "request.json")["evidence"]]
        trace = _gate(node, [reference(request_dir / "request.json"), reference(request_dir / "approval.json"),
                             *evidence], approval)
    _policy_node(wiki, commit, [*prior, *trace])
    write_jsonl(destination / "trace.jsonl", [to_wire(record) for record in trace])
    result = {"status": f"switched_to_{to}", "revision": revision, "commit": commit, "policy_node": node,
              "policy_state": state, "trace": asdict(reference(destination / "trace.jsonl"))}
    write_json(destination / "result.json", result)
    return result
