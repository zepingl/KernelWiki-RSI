"""Numerical replay illustration, never a reconstructed or promotable history."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from kernelwiki.knowledge import Condition, KnowledgePolicy, KnowledgeRecord, PolicyRule

from kernel_workflow.rsi.discovery import DiscoveryWorld
from kernel_workflow.rsi.harness import Policy
from kernel_workflow.rsi.replay_buffer.evaluate import evaluate, qualifies
from kernel_workflow.rsi.replay_buffer.tests.test_simulator import EVIDENCE, node, world

SOURCE_WORLDS = (
    "node039/dsa-sparse-mla/f8dcf1b80680e86f",
    "node039/dsa-sparse-mla/d75526e9a709611e",
)


def construct_world(original):
    root_score = original.root.measurement.primary_score
    measured = sorted(
        (item for branch in original.branches for item in branch
         if item.measurement.status == "qualified" and item.measurement.primary_score > root_score),
        key=lambda item: item.measurement.primary_score,
    )
    if len(measured) < 2:
        raise ValueError("the illustration requires two positive, recorded measurements")
    weak, strong = measured[0], measured[-1]
    fixture = replace(
        world(original.kernel, "constructed/" + original.id),
        generation=replace(world().generation, worker_limit=1),
        root=node("root", None, 1.0),
        branches=(
            (node("A1", "root", weak.measurement.primary_score / root_score),
             node("A2", "A1", strong.measurement.primary_score / root_score)),
            (node("B0", "root", 1.0),),
        ),
    )
    provenance = {
        "world_id": original.id,
        "kernel": original.kernel,
        "root_score": root_score,
        "measurements": {
            "A1": {"attempt_id": weak.attempt_id, "original_parent": weak.parent_id,
                   "primary_score": weak.measurement.primary_score,
                   "quality": weak.measurement.primary_score / root_score - 1},
            "A2": {"attempt_id": strong.attempt_id, "original_parent": strong.parent_id,
                   "primary_score": strong.measurement.primary_score,
                   "quality": strong.measurement.primary_score / root_score - 1},
            "B0": {"primary_score": root_score, "quality": 0,
                   "origin": "reused root measurement, not an additional observed attempt"},
        },
        "topology": "constructed; original parent IDs are preserved separately",
        "dispatch_order": "constructed; weak branch before strong continuation",
        "fixture_source_context": "existing replay unit-test fixture, not the original kernel source/context",
    }
    return fixture, provenance


def policies():
    stop = KnowledgeRecord(
        "demo-stop", "Stop after the first direction", "Handwritten illustration only.", (), EVIDENCE,
        PolicyRule((Condition("remaining_budget", "lte", 1),), "stop"),
    )
    continuation = KnowledgeRecord(
        "demo-continue", "Continue a measured positive leaf", "Handwritten illustration only.", (), EVIDENCE,
        PolicyRule((Condition("node_kind", "eq", "leaf"),
                    Condition("measurement_status", "eq", "qualified"),
                    Condition("parent_gain", "gte", 0.001)), "prefer"),
    )
    return (
        ("neutral", "基线轮转", Policy(KnowledgePolicy(workers=1))),
        ("stop", "过早停止", Policy(KnowledgePolicy(rule_ids=(stop.id,), workers=1), (stop,))),
        ("continue", "续探已测优叶", Policy(KnowledgePolicy(rule_ids=(continuation.id,), workers=1), (continuation,))),
    )


def replay_rounds(fixtures):
    rounds = []
    incumbent_id = "neutral"
    for index in range(len(fixtures)):
        trials = []
        for identifier, title, policy in policies():
            result = evaluate(fixtures[:index + 1], policy, budgets=(2,))
            trials.append({"id": identifier, "title": title, "policy": policy.to_dict(), "result": result})
        incumbent = next(trial for trial in trials if trial["id"] == incumbent_id)
        selected = incumbent
        for trial in trials:
            if qualifies(trial["result"], selected["result"]):
                selected = trial
        rounds.append({
            "round": index + 1, "worlds": index + 1, "incumbent_id": incumbent_id,
            "selected_id": selected["id"], "policy_changed": selected["id"] != incumbent_id,
            "trials": trials, "publication_allowed": False,
        })
        incumbent_id = selected["id"]
    return rounds


def build_case(pool, archive):
    pool, archive = Path(pool), Path(archive)
    manifest = json.loads((pool / "manifest.json").read_text())
    payload = (pool / "worlds.jsonl").read_bytes()
    world_hash = hashlib.sha256(payload).hexdigest()
    if manifest["schema_version"] != 2 or manifest["artifacts"]["worlds.jsonl"] != world_hash:
        raise ValueError("the original schema-2 world archive failed its integrity check")
    indexed = {}
    for line_number, line in enumerate(payload.decode().splitlines(), 1):
        value = json.loads(line)
        if value["id"] in SOURCE_WORLDS:
            indexed[value["id"]] = (DiscoveryWorld.from_dict(value), line_number)
    fixtures, sources = [], []
    for identifier in SOURCE_WORLDS:
        original, line_number = indexed[identifier]
        fixture, provenance = construct_world(original)
        provenance["line"] = line_number
        fixtures.append(fixture)
        sources.append(provenance)
    historical = {}
    for name in ("incumbent-validation", "candidate-1-validation", "incumbent-test", "candidate-1-test"):
        historical[name] = json.loads((archive / "round-opus-01" / (name + ".json")).read_text())["summary"]
    verdict = json.loads((archive / "round-opus-01/result.json").read_text())
    return {
        "scope": "constructed-counterfactual-illustration",
        "execution": "current Python replay/evaluate code; choreographed loop; no online or model execution",
        "publication_allowed": False, "real_agent_calls": 0, "real_gpu_calls": 0,
        "proposals": "handwritten illustrative policies, not new Opus outputs",
        "budget": 2, "workers": 1, "beta_cost": 0.01,
        "sources": {"pool": pool.name, "worlds_sha256": world_hash, "worlds": sources},
        "rounds": replay_rounds(fixtures),
        "current": {key: manifest[key] for key in ("attempts", "replay_ready", "qualified_nodes", "worlds", "reverse_traces")}
                   | {"primary_worlds": len(manifest["cohorts"]["primary"]), "status": "insufficient_primary_support"},
        "historical": historical | {"schema_version": 1, "status": verdict["status"],
                                    "published": verdict["published"], "scope": "archived local validation gain; final held-out rejection"},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--archive", required=True)
    args = parser.parse_args()
    print("window.RSI_CASE = " + json.dumps(build_case(args.pool, args.archive), ensure_ascii=False, indent=2) + ";")


if __name__ == "__main__":
    main()
