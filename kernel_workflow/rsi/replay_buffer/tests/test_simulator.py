import json
from dataclasses import asdict, replace

import pytest
from kernel_workflow.rsi.discovery import (
    ContextSnapshot,
    DiscoveryNode,
    DiscoveryWorld,
    GenerationSpec,
    Measurement,
    SourceState,
)
from kernel_workflow.rsi.harness import Policy
from kernel_workflow.rsi.replay_buffer.cohorts import split_worlds
from kernel_workflow.rsi.replay_buffer.evaluate import evaluate, qualifies
from kernel_workflow.rsi.replay_buffer.objective import ReplayObjective, aggregate
from kernel_workflow.rsi.replay_buffer.simulator import Action, ReplayWorld, replay
from kernelwiki.knowledge import (
    Condition,
    EvidenceRef,
    KernelState,
    KnowledgeError,
    KnowledgePolicy,
    KnowledgeRecord,
    PolicyRule,
    digest,
)

EVIDENCE = (EvidenceRef("fixture.json", "a" * 64),)


def node(name, parent, score):
    measurement = (Measurement("qualified", "b" * 64, score, "fixture-v1", EVIDENCE)
                   if score is not None else Measurement("failed"))
    return DiscoveryNode(name, parent, name if parent else None, measurement, EVIDENCE,
                         state=SourceState("b" * 64, EVIDENCE),
                         context=ContextSnapshot(digest(name), EVIDENCE, digest(parent) if parent else None),
                         input_state="b" * 64 if parent else None,
                         input_context=digest(parent) if parent else None,
                         candidate_source="b" * 64 if parent else None)


def world(kernel="train", name="world", *, deep=False):
    return DiscoveryWorld(name, kernel, "campaign", KernelState(operator="attention", backend="triton"),
                          GenerationSpec("deep_explore" if deep else "normal", "fixed-agent", "high",
                                         "context-v1", "evaluator-v1"), node("root", None, 2.0),
                          ((node("first", "root", 2.02), node("continuation", "first", 2.4)),
                           (node("other", "root", None),)))


def leaf_policy():
    rule = KnowledgeRecord("prefer-leaf", "Continue measured branches", "fixture", (), EVIDENCE,
                           PolicyRule((Condition("node_kind", "eq", "leaf"),), "prefer"))
    return Policy(KnowledgePolicy(rule_ids=(rule.id,), workers=1), (rule,))


def test_world_roundtrip_and_rejects_forks_or_incomparable_measurements():
    expected = world()
    assert DiscoveryWorld.from_dict(json.loads(json.dumps(expected.to_dict()))) == expected
    with pytest.raises(KnowledgeError, match="continuation chains"):
        replace(expected, branches=((node("first", "root", 2.1), node("second", "root", 2.2)),))
    with pytest.raises(KnowledgeError, match="different scoring"):
        replace(expected, branches=((replace(node("first", "root", 2.1),
                                             measurement=Measurement("qualified", "b" * 64, 2.1,
                                                                     "other", EVIDENCE)),),))


def test_prefix_observation_and_whole_batch_before_reveal():
    simulator = ReplayWorld(world(), 4)
    initial = asdict(simulator.observe())
    assert [parent["id"] for parent in initial["parents"]] == ["root"]
    assert "branches" not in initial and "measurement" not in initial
    assert "source" not in json.dumps(initial) and "continuation" not in json.dumps(initial)
    with pytest.raises(KnowledgeError, match="illegal"):
        simulator.step(Action(("root", "root")))
    simulator.step(Action(("root",)))
    simulator.step(Action(("root",)))
    assert simulator.trace()["attempted_nodes"] == 2
    assert simulator.trace()["decision_batches"] == 2
    assert [event["revealed"][0]["node_id"] for event in simulator.trace()["events"]] == ["first", "other"]
    assert simulator.observe().remaining_budget == 2


def test_no_free_internal_success_and_failures_are_charged():
    trace = replay(world(), Policy(), 2)
    assert trace["best_quality"] == pytest.approx(0.01)
    assert trace["charged_budget"] == 2
    assert trace["attempted_nodes"] == 2
    assert trace["stop_reason"] == "budget_exhausted"
    assert ReplayObjective().score(trace)["objective"] == pytest.approx(-0.01)


def test_replay_policy_improvement_and_neutral_baseline():
    candidate = replay(world(), leaf_policy(), 2)
    baseline = replay(world(), Policy(KnowledgePolicy(workers=1)), 2)
    neutral = replay(world(), Policy(leaf_policy().knowledge, leaf_policy().records, neutral=True), 2)
    assert candidate["best_quality"] == pytest.approx(0.2)
    assert neutral == baseline
    assert candidate["events"][1]["action"]["rule_ids"] == ("prefer-leaf",)


def test_reset_each_budget_and_deterministic_support_exhaustion():
    simulator = ReplayWorld(world(), 6)
    simulator.step(Action(("root",)))
    assert simulator.reset().parents[0].id == "root"
    assert simulator.trace()["charged_budget"] == 0
    first = replay(world(), leaf_policy(), 6)
    assert first == replay(world(), leaf_policy(), 6)
    assert first["stop_reason"] == "round_limit"
    assert first["charged_budget"] == 2
    assert first["decision_batches"] == 6
    assert not first["support_complete"]
    assert evaluate([world()], leaf_policy())["traces"][0]["budget"] == 1


def test_deep_cost_limits_batches_and_does_not_look_ahead():
    simulator = ReplayWorld(world(deep=True), 4)
    assert simulator.observe().direction_cost == 2
    with pytest.raises(KnowledgeError, match="illegal"):
        simulator.step(Action(("root", "root")))
    trace = replay(world(deep=True), leaf_policy(), 1)
    assert trace["attempted_nodes"] == 0 and trace["stop_reason"] == "budget_exhausted"
    assert replay(world(deep=True), leaf_policy(), 2)["charged_budget"] == 2


def test_stop_rule_is_executable_and_observable():
    rule = KnowledgeRecord("stop", "Stop after an unproductive batch", "fixture", (), EVIDENCE,
                           PolicyRule((Condition("stagnation", "gte", 1),), "stop"))
    policy = Policy(KnowledgePolicy(rule_ids=("stop",)), (rule,))
    data = replace(world(), branches=((node("failure", "root", None),),))
    result = replay(data, policy, 6)
    assert result["stop_reason"] == "support_exhausted" and result["charged_budget"] == 1


def test_macro_aggregation_does_not_weight_large_kernels_more():
    traces = [replay(world("small", "small"), leaf_policy(), 2)]
    traces += [replay(world("large", f"large-{index}"), Policy(), 2) for index in range(5)]
    summary = aggregate(traces)
    assert summary["objective"] == pytest.approx((0.18 - 0.01) / 2)


def test_splits_and_paired_nonregression():
    splits = {"train": ["train"], "validation": ["validation"], "test": ["test"]}
    pool = [world(kernel, kernel) for kernel in ("train", "validation", "test")]
    assert len(split_worlds(pool, splits)["train"]) == 1
    with pytest.raises(KnowledgeError, match="exactly one split"):
        split_worlds(pool, {**splits, "test": ["train"]})
    with pytest.raises(KnowledgeError, match="lacks verified"):
        split_worlds(pool[:-1], splits)
    candidate = evaluate([world()], leaf_policy(), (1, 2))
    baseline = evaluate([world()], Policy(), (1, 2))
    assert qualifies(candidate, baseline)
    assert not qualifies(baseline, baseline)
    assert qualifies(baseline, baseline, held_out=True)


def test_unsupported_pairs_block_qualification_without_candidate_dependent_drops():
    candidate = evaluate([world()], leaf_policy())
    baseline = evaluate([world()], Policy())
    assert not candidate["qualification"]["eligible"]
    assert not qualifies(candidate, baseline)
    assert candidate["qualification"]["unsupported_pairs"]
    candidate["traces"].pop()
    with pytest.raises(KnowledgeError, match="identical world-budget pairs"):
        qualifies(candidate, baseline)
