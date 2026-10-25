"""Verify fixed-harness histories and decode GEAK's captured workflow boundaries."""

import difflib
from dataclasses import asdict
from pathlib import Path

from kernelwiki.knowledge import EvidenceRef, KnowledgeError

from ...discovery import Attempt, DiscoveryWorld, reconstruct
from ...evidence import read_json, read_jsonl, reference, verify, verify_artifacts
from ...harness import Policy


def load_online(directory):
    directory = Path(directory)
    result = read_json(directory / "result.json")
    if result.get("status") != "online_executed" or result.get("knowledge_changed") is not False:
        raise KnowledgeError("only completed fixed-harness online histories can enter replay")
    if result.get("worlds") != 1:
        raise KnowledgeError("online history contains no replayable discovery world")
    if not {"manifest.json", "trace.json", "policy.json", "worlds.jsonl"} <= result["artifacts"].keys():
        raise KnowledgeError("online history lacks its required immutable artifacts")
    verify_artifacts(directory, result)
    manifest = read_json(directory / "manifest.json")
    for item in manifest["sources"].values():
        verify(EvidenceRef.from_dict(item))
    worlds = [DiscoveryWorld.from_dict(value) for value in read_jsonl(directory / "worlds.jsonl")]
    if len(worlds) != 1:
        raise KnowledgeError("online history must contain exactly one declared world")
    world = worlds[0]
    policy = Policy.from_dict(read_json(directory / "policy.json"))
    if (result["trace"] != read_json(directory / "trace.json") or result["trace"]["world_id"] != world.id
            or result["trace"]["kernel"] != world.kernel or result["root_source"] != world.root.state.fingerprint
            or result["root_context"] != world.root.context.revision or result["policy_revision"] != policy.revision
            or manifest["generation"] != asdict(world.generation)
            or any(result[key] != manifest[key] for key in ("knowledge_revision", "policy_revision", "reference_revision", "execution"))):
        raise KnowledgeError("online result disagrees with its source, policy, context or execution manifest")
    normalized = []
    for node in (world.root, *(node for branch in world.branches for node in branch)):
        for item in (*node.input_evidence, *node.measurement.evidence, *node.state.evidence, *node.context.evidence):
            verify(item)
        if node.attempt_id:
            normalized.append(Attempt(node.attempt_id, world.kernel, world.campaign,
                                      node.input_evidence[0], node.input_evidence[0],
                                      node.input_state, node.candidate_source, node.measurement,
                                      "replay_ready", (), node.input_evidence,
                                      node.state, node.context, node.input_context,
                                      node.parent_id if node.parent_id != world.root.id else None))
    rebuilt = reconstruct(world.id, world.kernel, world.campaign, world.state, world.generation, world.root, normalized)
    if rebuilt != world:
        raise KnowledgeError("normalized online attempts do not reconstruct the recorded world")
    return result, world, tuple(normalized)


def capture_evidence(subject):
    directory = Path(subject.attempt.invocation.path).parent
    request = read_json(directory / "request.json")
    disposition = read_json(directory / "capture/host-commit-director.json")
    boundaries = [{"boundary": read_json(path), "evidence": asdict(reference(path))}
                  for path in sorted((directory / "capture").glob("boundary-*.json"))]
    plan = next((item for item in boundaries if item["boundary"]["kind"] == "plan_dispatch"), None)
    generated = next((item for item in boundaries if item["boundary"]["kind"] == "candidate_verify"), None)
    diffs = []
    if generated:
        sources = generated["boundary"]["sources"]
        parent_files, candidate_files = sources["parent"]["files"], sources["candidate"]["files"]
        for name in sorted(set(parent_files) | set(candidate_files)):
            if parent_files.get(name) == candidate_files.get(name):
                continue
            bodies, refs = [], []
            for files in (parent_files, candidate_files):
                if name not in files:
                    bodies.append("")
                    continue
                path = directory / "capture/objects" / files[name]["sha256"]
                item = reference(path)
                if item.sha256 != files[name]["sha256"]:
                    raise KnowledgeError("captured online source object changed")
                refs.append(asdict(item))
                bodies.append(verify(item).decode(errors="replace"))
            body = "".join(difflib.unified_diff(bodies[0].splitlines(True), bodies[1].splitlines(True),
                                                fromfile="parent/" + name, tofile="candidate/" + name))
            diffs.append({"path": name, "excerpt": body[:16000], "truncated": len(body) > 16000,
                          "evidence": refs, "source_boundary": generated["evidence"]})

    agent_inputs = [{"input": read_json(path), "evidence": asdict(reference(path))}
                    for path in sorted((directory / "capture").glob("agent-*-input.json"))]
    return {"original_inputs": request, "disposition": disposition, "boundaries": boundaries,
            "plan": plan, "agent_inputs": agent_inputs, "diffs": diffs,
            "disposition_evidence": asdict(reference(directory / "capture/host-commit-director.json"))}
