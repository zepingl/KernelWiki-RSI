from kernel_workflow.rsi.harness import Policy
from kernel_workflow.rsi.replay_buffer.cohorts import cohorts, split_worlds, tree_gaps
from kernel_workflow.rsi.replay_buffer.evaluate import evaluate
from kernel_workflow.rsi.replay_buffer.simulator import Action, TreeReplay, replay
from kernel_workflow.rsi.replay_buffer.tests.test_simulator import leaf_policy, world
from kernel_workflow.rsi.trace.records import CODE, Decide, Measure, Node, Tree
from kernel_workflow.rsi.trace.tests.test_tree import EVIDENCE, expansion, lane, measure, state
from kernel_workflow.rsi.trace.tree import fold
from kernelwiki.knowledge import KernelState


def revealed(simulator):
    return [row["node_id"] for event in simulator.rollout.events for row in event["revealed"]]


def test_selecting_a_node_reveals_its_recorded_children_in_dispatch_order():
    simulator = TreeReplay(fold(lane()).trees["t"], 6)
    simulator.step(Action(("root",)))
    simulator.step(Action(("root",)))
    assert revealed(simulator) == ["t/e1/0", "t/e1/1"]
    simulator.step(Action(("root",)))
    assert simulator.rollout.events[-1]["unsupported"] == ["root"]
    simulator.step(Action(("node-1",)))
    assert revealed(simulator)[-1] == "t/e2/0" and simulator.done
    assert simulator.trace()["stop_reason"] == "support_exhausted"


def test_tree_cohort_needs_branching_continuation_known_outcomes_and_budget():
    view = fold(lane()).trees["t"]
    assert tree_gaps(view, 3) == []
    assert tree_gaps(view, 6) == ["recorded cost does not cover the largest budget"]
    unknown = fold([record for record in lane() if record != measure("t/e2/0", 2.6)]).trees["t"]
    assert tree_gaps(unknown, 3) == ["unknown evaluation outcomes"]
    single = fold(lane()[:8]).trees["t"]
    assert tree_gaps(single, 2) == ["no supported choice between a new branch and continuation"]
    assert cohorts((view,), (1, 2, 3), traces="trace.jsonl")["primary"] == ["t"]


def test_trees_replay_through_the_shared_evaluator_and_policy():
    view = fold(lane()).trees["t"]
    result = evaluate((view,), leaf_policy(), (1, 2))
    assert result["qualification"]["eligible"]
    full = [trace for trace in result["traces"] if trace["budget"] == 2][0]
    assert full["kernel"] == "train" and full["best_quality"] == 2.6 / 2.0 - 1
    # Continuing past the last recorded leaf is reported, never invented.
    beyond = evaluate((view,), leaf_policy(), (1, 3))
    assert beyond["qualification"]["unsupported_pairs"][0]["budget"] == 3
    assert split_worlds((view, world("validation"), world("test", "w2")), {
        "train": ["train"], "validation": ["validation"], "test": ["test"]})["train"] == (view,)


def test_single_expansion_tree_replays_like_the_equivalent_v1_world():
    v1 = world()
    tree = Tree("w", CODE, "fixture-v1", v1.state, facets={"geak": {"campaign": "campaign", "kernel": "train"}})
    root = Node("w/root", "w", None, 0, state(5))
    first = Node("w/e1/0", "w", "w/e1", 0, state(6))
    other = Node("w/e1/1", "w", "w/e1", 1, state(7))
    records = [tree, root, Measure(root.id, "fixture-v1", "director", True, 2.0, EVIDENCE),
               expansion("e1", root.id, slots=2, tree="w"), first,
               Measure(first.id, "fixture-v1", "verifier", True, 2.02, EVIDENCE), other,
               Decide("w", other.id, "incorrect", "fixture failure")]
    view = fold(records).trees["w"]
    flat = world()
    flat = type(flat)(flat.id, flat.kernel, flat.campaign, flat.state, flat.generation, flat.root,
                      tuple(branch[:1] for branch in flat.branches))
    policy = Policy()
    for budget in (1, 2):
        left, right = replay(view, policy, budget), replay(flat, policy, budget)
        assert (left["best_quality"], left["attempted_nodes"]) == (right["best_quality"], right["attempted_nodes"])
