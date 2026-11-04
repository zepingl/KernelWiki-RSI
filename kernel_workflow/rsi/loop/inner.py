"""Bounded Knowledge Developer/Reviewer loop over frozen Dream-RSI replay worlds."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from kernelwiki import KnowledgeClient
from kernelwiki.knowledge import (
    KnowledgeError,
    KnowledgePolicy,
    KnowledgeRecord,
    canonical,
    strict_fields,
)

from ..evidence import (
    artifact_hashes,
    evidence_refs,
    reference,
    verify,
    write_json,
)
from ..harness import INTERPRETER_VERSION, Policy
from ..replay_buffer.cohorts import cohorts, kernel_of
from ..replay_buffer.evaluate import evaluate, qualifies
from ..replay_buffer.objective import ReplayObjective
from ..replay_buffer.simulator import PLANNED_ROLES
from ..replay_buffer.store import open_buffer
from ..reverse_engineering.trace import investigation_input
from ..runtime import (
    EFFORT,
    MODEL,
    AgentRole,
    AgentUnavailable,
    FrozenExperimentChanged,
    RuntimeAgent,
    frozen_sources,
)
from ..trace.tree import document_path

ROLES = {
    "preflight": AgentRole({"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}},
                          'Return {"ok":true}. Do not use tools.'),
    "developer": AgentRole({"type": "object", "required": ["kind", "rationale", "policy", "records"],
                            "properties": {"kind": {"enum": ["policy_update", "evidence_update", "needs_online"]},
                                           "rationale": {"type": "string"}, "policy": {"type": "object"},
                                           "records": {"type": "array"}}}, Path(__file__).parent / "roles/developer.md"),
    "reviewer": AgentRole({"type": "object", "required": ["decision", "reasons"],
                           "properties": {"decision": {"enum": ["approve", "reject"]},
                                          "reasons": {"type": "array"}}}, Path(__file__).parent / "roles/reviewer.md"),
}


def _check_frozen(code, paths, pool, pool_ref):
    try:
        verify(pool_ref)
        open_buffer(pool)
        for name, path in paths.items():
            if reference(path).sha256 != code[name]:
                raise KnowledgeError(f"frozen implementation changed: {name}")
    except (KnowledgeError, OSError) as error:
        raise FrozenExperimentChanged(str(error)) from error


def _tree_training_input(trees, result):
    """Recorded episodes: per-step plans (with the KernelWiki pages each direction pointed to), selections and
    revealed outcomes; citable evidence is the measures'."""
    evidence, examples = {}, []
    for tree in sorted(trees, key=lambda item: (kernel_of(item), item.id)):
        for measures in tree.measures.values():
            for measure in measures:
                for item in measure.evidence:
                    evidence[canonical(asdict(item))] = item
        pointed = {(node.expansion, node.slot): sorted({document_path(use.target) for use in node.uses
                                                        if use.role in PLANNED_ROLES})
                   for node in tree.nodes.values() if node.expansion}
        steps = [{"round": position, "parent_quality": tree.outcome(expansion.parents[0]).status,
                  "slots": [{**asdict(slot), "knowledge": pointed.get((expansion.id, index), [])}
                            for index, slot in enumerate(expansion.plan.slots)],
                  "dispatched": expansion.plan.dispatched}
                 for position, expansion in enumerate(tree.expansions, 1)]
        examples.append({"tree": tree.id, "kernel": kernel_of(tree), "state": asdict(tree.tree.task), "steps": steps,
                         "trajectories": [trace for trace in result["traces"] if trace["world_id"] == tree.id],
                         "evidence": [asdict(item) for measure in tree.measures[tree.root.id]
                                      for item in measure.evidence]})
    return {"summary": result["summary"], "qualification": result["qualification"], "examples": examples,
            "reverse_evidence": [], "sample_scope": "all primary training episodes and all frozen budgets"}, evidence


def _training_input(worlds, result, reverse_traces=()):
    evidence, examples = {}, []
    for world in sorted(worlds, key=lambda item: (item.kernel, item.id)):
        for node in (world.root, *(node for branch in world.branches for node in branch)):
            for item in (*node.input_evidence, *node.measurement.evidence, *node.state.evidence, *node.context.evidence):
                evidence[canonical(asdict(item))] = item
        trajectories = [trace for trace in result["traces"] if trace["world_id"] == world.id]
        examples.append({"world": world.id, "kernel": world.kernel, "state": asdict(world.state),
                         "trajectories": trajectories, "evidence": [asdict(item) for item in world.root.input_evidence]})
    reverse = []
    kernels = {world.kernel for world in worlds}
    for report in reverse_traces:
        if report.subject.attempt.kernel not in kernels:
            continue
        value = investigation_input(report)
        for item in evidence_refs(report.to_trace()):
            verify(item)
            evidence[canonical(asdict(item))] = item
        reverse.append(value)
    return {"summary": result["summary"], "qualification": result["qualification"], "examples": examples,
            "reverse_evidence": reverse, "sample_scope": "all primary training worlds and all frozen budgets; no hidden sampling"}, evidence


def _candidate(proposal, incumbent, allowed_evidence):
    strict_fields(proposal, {"kind", "rationale", "policy", "records"},
                  {"kind", "rationale", "policy", "records"})
    if proposal["kind"] != "policy_update":
        if proposal["kind"] not in {"evidence_update", "needs_online"}:
            raise KnowledgeError("invalid proposal kind")
        return None
    if not isinstance(proposal["rationale"], str) or not proposal["rationale"]:
        raise KnowledgeError("candidate needs an evidence-bounded rationale")
    knowledge = KnowledgePolicy.from_dict(proposal["policy"])
    records = {record.id: record for record in incumbent.records}
    changed = tuple(KnowledgeRecord.from_dict(value) for value in proposal["records"])
    if len({record.id for record in changed}) != len(changed):
        raise KnowledgeError("duplicate candidate record")
    for record in changed:
        if record.policy_rule is None or (record.id in records and records[record.id].policy_rule is None):
            raise KnowledgeError("reference knowledge changes need online evaluation")
        if not record.evidence:
            raise KnowledgeError("candidate rule has no training evidence")
        for item in record.evidence:
            if canonical(asdict(item)) not in allowed_evidence:
                raise KnowledgeError("candidate cites evidence outside the frozen training split")
            verify(item)
        records[record.id] = record
    return Policy(knowledge, tuple(records.values()))


def evolve(pool, wiki, output, *, agent=None, publish=False, max_candidates=4, progress=None, policy=None,
           episodes=None):
    if publish:
        raise KnowledgeError("inner evolution never publishes; final cycle qualification requires fresh holdout and paired online evidence")
    if type(max_candidates) is not int or not 1 <= max_candidates <= 4:
        raise KnowledgeError("one round permits at most four new candidates")
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    buffer = open_buffer(pool)
    manifest, worlds = buffer.manifest, buffer.worlds
    if "cohorts" not in manifest:
        raise KnowledgeError("primary and diagnostic cohorts must be frozen before candidate evaluation")
    snapshot = KnowledgeClient(wiki).snapshot()
    incumbent = Policy.from_dict(policy.to_dict(), snapshot.records) if policy else Policy(snapshot.policy, snapshot.records)
    objective = ReplayObjective(**manifest["objective"])
    budgets = tuple(manifest["budgets"])
    if episodes:
        # Policy schema 2 replays recorded expansions; its cohort is fixed from the frozen pool before any candidate.
        frozen_cohorts = cohorts(buffer.trees, budgets, traces="trace.jsonl", episodes=True)
        primary = tuple(tree for tree in buffer.trees if tree.id in set(frozen_cohorts["primary"]))
        worlds = buffer.trees
    else:
        frozen_cohorts = manifest["cohorts"]
        primary = buffer.select(cohort="primary")
    if episodes:
        # Historical kernels were all inspected already; generalization is owned by fresh online holdout runs.
        # Offline, every primary episode trains and gates, and no kernel's full-budget quality may regress.
        partitions = {"train": primary, "validation": primary}
        missing = {}
    else:
        partitions = {name: tuple(world for world in primary if kernel_of(world) in names)
                      for name, names in manifest["splits"].items()}
        missing = {name: sorted(set(manifest["splits"][name]) - {kernel_of(world) for world in partitions[name]})
                   for name in ("train", "validation")}
        missing = {name: values for name, values in missing.items() if values}
    if missing or not partitions["train"] or not partitions["validation"]:
        result = {"status": "insufficient_primary_support", "reason": "branching, continuation, complete outcomes and budget coverage are required",
                  "missing_primary_kernels": missing, "primary_worlds": len(primary), "diagnostic_worlds": len(worlds),
                  "knowledge_revision": snapshot.revision, "policy_revision": incumbent.revision,
                  "opportunities": [], "selected": None, "published": False, "test_evaluated": False, "agent_calls": 0}
        write_json(destination / "selected-policy.json", incumbent.to_dict())
        result["artifacts"] = artifact_hashes(destination)
        write_json(destination / "result.json", result)
        return result
    paths = frozen_sources()
    pool_ref = reference(Path(pool) / "manifest.json")
    frozen = {"schema_version": 2, "pool": asdict(pool_ref), "knowledge_revision": snapshot.revision,
              "policy_revision": incumbent.revision, "interpreter": INTERPRETER_VERSION, "metric": "root-relative-qualified-gain-v2",
              "model": MODEL, "effort": EFFORT, "budgets": budgets, "objective": asdict(objective),
              "splits": manifest["splits"], "cohorts": frozen_cohorts, "max_candidates": max_candidates,
              "episodes": episodes,
              "selection": ("pooled replay gate over every primary episode: higher Objective, no kernel's full-budget "
                            "quality lower; generalization is owned by fresh online holdout") if episodes else
                           "fixed primary validation split",
              "test_role": "historical regression only; no test evaluation or release in an inner round",
              "code": {name: reference(path).sha256 for name, path in paths.items()}}
    write_json(destination / "manifest.json", frozen)
    for name, path in paths.items():
        target = destination / "frozen_code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    for name in ("manifest.json", *manifest["artifacts"]):
        target = destination / "frozen_pool" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((Path(pool) / name).read_bytes())
    for name, payload in snapshot.files:
        path = destination / "frozen_knowledge" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    _check_frozen(frozen["code"], paths, pool, pool_ref)
    write_json(destination / "incumbent.json", incumbent.to_dict())
    incumbent_train = evaluate(partitions["train"], incumbent, budgets, objective, episodes=episodes)
    write_json(destination / "incumbent-train.json", incumbent_train)
    if episodes:
        training, allowed_evidence = _tree_training_input(partitions["train"], incumbent_train)
    else:
        reverse_traces = buffer.reverse_reports(manifest["splits"]["train"])
        training, allowed_evidence = _training_input(partitions["train"], incumbent_train, reverse_traces)
    examples = (lambda result: _tree_training_input(partitions["train"], result)[0]["examples"]) if episodes else \
        (lambda result: _training_input(partitions["train"], result)[0]["examples"])
    agent = agent or RuntimeAgent(roles=ROLES)
    calls = 0

    def request(role, payload, directory):
        nonlocal calls
        calls += 1
        return agent(role, payload, directory)

    receipts, candidates, history = [], [], []
    editing_base, editing_train = incumbent, incumbent_train
    try:
        if request("preflight", {}, destination / "agents/preflight") != {"ok": True}:
            raise AgentUnavailable("experimental Agent preflight did not pass")
        _check_frozen(frozen["code"], paths, pool, pool_ref)
        for index in range(1, max_candidates + 1):
            _check_frozen(frozen["code"], paths, pool, pool_ref)
            receipt = {"opportunity": index, "editing_base_revision": editing_base.revision}
            proposal = review = candidate = None
            base_payload = editing_base.to_dict()
            try:
                payload = {"incumbent": incumbent.to_dict(), "editing_base": base_payload,
                           "editing_base_training": editing_train["summary"], "objective": asdict(objective),
                           "budgets": budgets, "training": training, "history": history}
                proposal = request("developer", payload, destination / f"agents/developer-{index}")
                _check_frozen(frozen["code"], paths, pool, pool_ref)
                write_json(destination / f"candidate-{index}.json", proposal)
                candidate = _candidate(proposal, editing_base, allowed_evidence)
                if candidate is None:
                    receipt.update(status=proposal["kind"], reason=proposal["rationale"])
                else:
                    review = request("reviewer", {"proposal": proposal, "editing_base": base_payload,
                                                "resolved_policy": candidate.to_dict(), "training": training,
                                                "objective": asdict(objective)}, destination / f"agents/reviewer-{index}")
                    _check_frozen(frozen["code"], paths, pool, pool_ref)
                    strict_fields(review, {"decision", "reasons"}, {"decision", "reasons"})
                    if review["decision"] not in {"approve", "reject"} or not isinstance(review["reasons"], list):
                        raise KnowledgeError("invalid reviewer decision")
                    write_json(destination / f"review-{index}.json", review)
                    if review["decision"] == "reject":
                        receipt.update(status="review_rejected", reasons=review["reasons"])
                    else:
                        train_result = evaluate(partitions["train"], candidate, budgets, objective, episodes=episodes)
                        write_json(destination / f"candidate-{index}-train.json", train_result)
                        if train_result["qualification"]["eligible"] and incumbent_train["qualification"]["eligible"]:
                            candidates.append((index, candidate, train_result))
                        receipt.update(status="replayed" if train_result["qualification"]["eligible"] else "unsupported_replay",
                                       training=train_result["summary"], qualification=train_result["qualification"],
                                       training_examples=examples(train_result))
                        if qualifies(train_result, editing_train):
                            editing_base, editing_train = candidate, train_result
            except FrozenExperimentChanged:
                raise
            except (KnowledgeError, ValueError, TypeError, KeyError) as error:
                receipt.update(status="invalid_proposal", reason=str(error))
            receipts.append(receipt)
            _check_frozen(frozen["code"], paths, pool, pool_ref)
            history.append({"opportunity": index, "editing_base": base_payload, "proposal": proposal,
                            "resolved_policy": candidate.to_dict() if candidate else None, "review": review,
                            "feedback": receipt})
            write_json(destination / f"opportunity-{index}.json", history[-1])
            if progress:
                progress({"opportunity": index, "status": receipt["status"]})
    except (AgentUnavailable, FrozenExperimentChanged) as error:
        result = {"status": "agent_failed" if isinstance(error, AgentUnavailable) else "frozen_input_changed",
                  "error": str(error), "opportunities": receipts, "knowledge_revision": snapshot.revision,
                  "policy_revision": incumbent.revision, "published": False, "agent_calls": calls}
        write_json(destination / "result.json", result)
        raise
    _check_frozen(frozen["code"], paths, pool, pool_ref)
    if episodes:
        baseline_validation = incumbent_train
    else:
        baseline_validation = evaluate(partitions["validation"], incumbent, budgets, objective)
        write_json(destination / "incumbent-validation.json", baseline_validation)
    selected = None
    best = baseline_validation["summary"]["objective"]
    for index, candidate, train_result in candidates:
        if episodes:
            validation = train_result
        else:
            validation = evaluate(partitions["validation"], candidate, budgets, objective)
            write_json(destination / f"candidate-{index}-validation.json", validation)
        if (qualifies(validation, baseline_validation, held_out=bool(episodes))
                and validation["summary"]["objective"] > best + 1e-12):
            selected = (index, candidate, validation)
            best = validation["summary"]["objective"]
    working = selected[1] if selected else incumbent
    write_json(destination / "selected-policy.json", working.to_dict())
    result = {"selection": frozen["selection"],
              "status": "working_policy_selected" if selected else "no_change",
              "knowledge_revision": snapshot.revision, "policy_revision": working.revision,
              "selected": selected[0] if selected else None, "published": False,
              "opportunities": receipts, "test_evaluated": False, "agent_calls": calls,
              "test_role": "historical regression; final unseen holdout is owned by the outer cycle",
              "reason": ("selected by the pooled replay gate; no active-pointer change" if episodes else
                         "selected on fixed primary validation only; no active-pointer change") if selected else
                        "no supported candidate improved Objective without full-budget quality regression"}
    if selected:
        result["validation"] = selected[2]["summary"]
    result["artifacts"] = artifact_hashes(destination)
    write_json(destination / "result.json", result)
    return result
