from dataclasses import replace

import pytest
from kernel_workflow.rsi.discovery import ContextSnapshot, DiscoveryWorld, Measurement
from kernel_workflow.rsi.harness import Policy
from kernel_workflow.rsi.replay_buffer.simulator import Action, ReplayWorld
from kernel_workflow.rsi.replay_buffer.tests.test_simulator import EVIDENCE, node, world
from kernelwiki.knowledge import KnowledgeError, digest


def test_initial_action_is_a_set_even_with_two_workers():
    simulator = ReplayWorld(world(), 6)
    assert Policy().select(simulator.observe()).parents == ("root",)


def test_unsupported_continuation_is_counted_and_not_pruned():
    data = replace(world(), branches=((node("first", "root", 2.1),),
                                      (node("second", "root", 2.2),)))
    simulator = ReplayWorld(data, 2)
    simulator.step(Action(("root",)))
    simulator.step(Action(("node-1",)))
    trace = simulator.trace()
    assert trace["decision_batches"] == 2 and trace["charged_budget"] == 1
    assert trace["stop_reason"] == "round_limit" and not trace["support_complete"]
    assert "node-1" in {parent.id for parent in simulator.observe().parents}


def test_unqualified_nodes_retain_independent_source_identity():
    child = replace(node("first", "root", None), measurement=Measurement("unknown"))
    data = replace(world(), branches=((child, node("continuation", "first", 2.4)),))
    restored = DiscoveryWorld.from_dict(data.to_dict())
    assert restored.branches[0][0].state.fingerprint == "b" * 64
    assert restored.branches[0][1].input_state == restored.branches[0][0].state.fingerprint


def test_matching_source_with_wrong_parent_context_is_rejected():
    child = replace(node("first", "root", 2.1), input_context=digest("unrelated"),
                    context=ContextSnapshot(digest("child"), EVIDENCE, digest("unrelated")))
    with pytest.raises(KnowledgeError, match="parent's source and context"):
        replace(world(), branches=((child,),))


def test_schema_one_cannot_be_silently_requalified():
    value = {**world().to_dict(), "schema_version": 1}
    with pytest.raises(KnowledgeError, match="legacy"):
        DiscoveryWorld.from_dict(value)
