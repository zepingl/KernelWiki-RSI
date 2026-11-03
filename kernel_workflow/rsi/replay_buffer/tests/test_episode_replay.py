import pytest
from kernel_workflow.rsi.harness import ExpansionObservation, Policy, SlotView
from kernel_workflow.rsi.replay_buffer.cohorts import cohorts, episode_gaps
from kernel_workflow.rsi.replay_buffer.evaluate import evaluate
from kernel_workflow.rsi.replay_buffer.simulator import episode_replay
from kernel_workflow.rsi.trace.records import (
    CODE,
    ContextRef,
    Decide,
    Expansion,
    Generation,
    Measure,
    Node,
    Plan,
    Slot,
    Tree,
    Use,
)
from kernel_workflow.rsi.trace.tree import fold
from kernelwiki.knowledge import Condition, EvidenceRef, KernelState, KnowledgePolicy, KnowledgeRecord, PolicyRule

EVIDENCE = (EvidenceRef("fixture.json", "a" * 64),)
CARD = "kernelwiki:knowledge/general/tuning.md@abc123456789"
PAGE = "knowledge/general/tuning.md"


def state(token):
    return format(abs(hash(token)) % 16 ** 12, "012x").rjust(64, "0")


def knowledge_records():
    return [Tree("kw", "knowledge:general_page", "kw-gate"), Expansion("kw#author", "kw", "author", ()),
            Node(CARD, "kw", "kw#author", 0, state("card"))]


def lane(name="t", kernel="train", steps=((2.6, 2.2, 2.1), (2.8, 2.5), (3.0,)), **options):
    return fold([*knowledge_records(), *lane_records(name, kernel, steps, **options)]).trees[name]


def lane_records(name, kernel, steps, *, evidence=EVIDENCE, continued=0, deep=(), undispatched=0, pointed=None,
                 unknown=()):
    """root 2.0; each step dispatches its slots, the next step continues `continued` of the previous one."""
    records = [Tree(name, CODE, "lane-v1", KernelState(operator="gemm"),
                    facets={"geak": {"campaign": "fixture", "kernel": kernel}}),
               Node(f"{name}/root", name, None, 0, state(name)),
               Measure(f"{name}/root", "lane-v1", "director", True, 2.0, evidence)]
    parent = f"{name}/root"
    for position, scores in enumerate(steps, 1):
        expansion_id = f"{name}/e{position}"
        slots = tuple(Slot(f"d{index}", "compute", "deep" if (position, index) in deep else "normal")
                      for index in range(len(scores) + (undispatched if position == 1 else 0)))
        records.append(Expansion(expansion_id, name, "generate", (parent,),
                                 Plan(position, ContextRef((), evidence), Generation("m", "high"), slots, len(scores))))
        for index, score in enumerate(scores):
            node_id = f"{expansion_id}/{index}"
            uses = (Use(CARD, "pointed", evidence),) if pointed == (position, index) else ()
            records.append(Node(node_id, name, expansion_id, index, state(node_id), uses))
            if (position, index) not in unknown:
                records.append(Measure(node_id, "lane-v1", "verifier", True, score, evidence))
        parent = f"{expansion_id}/{continued}"
        if position < len(steps):
            records.append(Decide(name, parent, "committed", "fixture"))
    return records


def policy(*rules, max_width=8):
    records = tuple(KnowledgeRecord(f"rule-{index}", "rule", "fixture", (), EVIDENCE, rule)
                    for index, rule in enumerate(rules))
    return Policy(KnowledgePolicy(schema_version=2, rule_ids=tuple(record.id for record in records),
                                  max_width=max_width), records)


def test_without_rules_the_whole_recorded_plan_is_replayed():
    trace = episode_replay(lane(), policy(), 6)
    assert trace["attempted_nodes"] == 6 and trace["charged_budget"] == 6
    assert trace["best_quality"] == pytest.approx(3.0 / 2.0 - 1)
    assert trace["support_complete"] and trace["stop_reason"] == "budget_exhausted"


def test_step_rule_scores_each_round_where_history_was_and_truncate_cuts_lost_continuations():
    tree = lane(continued=1)
    narrow = policy(PolicyRule((), "width", 1))
    step = episode_replay(tree, narrow, 6, "step")
    assert step["attempted_nodes"] == 3 and step["charged_budget"] == 3 and step["support_complete"]
    assert step["stop_reason"] == "recorded_end"
    assert [event["action"] for event in step["events"]] == [[0], [0], [0]]
    truncated = episode_replay(tree, narrow, 6, "truncate")
    assert truncated["stop_reason"] == "continuation_not_selected" and not truncated["support_complete"]


def test_selection_by_pointed_knowledge_and_undispatched_slots():
    tree = lane(pointed=(1, 2))
    # A rule names the page by path, so it matches whichever version the direction pointed to.
    picked = episode_replay(tree, policy(PolicyRule((Condition("knowledge", "contains", PAGE),), "select")), 1)
    assert picked["events"][0]["action"] == [2]
    unsupported = episode_replay(lane(undispatched=1), policy(PolicyRule((Condition("slot_index", "gte", 3),),
                                                                         "select")), 1)
    assert unsupported["unsupported_requests"] == 1 and not unsupported["support_complete"]


def test_stop_skip_and_deep_cost():
    stopped = episode_replay(lane(), policy(PolicyRule((Condition("round", "gte", 2),), "stop")), 6)
    assert stopped["stop_reason"] == "policy_stop" and stopped["charged_budget"] == 3
    deep = lane(deep={(1, 0)})
    skipped = episode_replay(deep, policy(PolicyRule((Condition("mode", "eq", "deep"),), "skip")), 6)
    assert skipped["events"][0]["action"] == [1, 2]
    assert episode_replay(deep, policy(), 6)["events"][0]["remaining_budget"] == 2


def test_interpreter_ignores_v1_rules_and_respects_budget():
    slots = tuple(SlotView(index, "compute", "normal") for index in range(4))
    observation = ExpansionObservation(KernelState(), 1, 0, 2, 0.0, slots)
    assert Policy().select_expansion(observation).slots == (0, 1)


def test_episode_cohort_and_paired_evaluation():
    assert episode_gaps(lane(), 6) == []
    assert episode_gaps(lane(), 7) == ["recorded cost does not cover the largest budget"]
    assert episode_gaps(lane(unknown={(2, 1)}), 6) == ["unknown evaluation outcomes"]
    trees = (lane("a"), lane("b"))
    assert cohorts(trees, (1, 6), traces="trace.jsonl", episodes=True)["primary"] == ["a", "b"]
    result = evaluate(trees, policy(PolicyRule((), "width", 1)), (1, 6), episodes="step")
    assert result["qualification"]["eligible"] and len(result["traces"]) == 4
