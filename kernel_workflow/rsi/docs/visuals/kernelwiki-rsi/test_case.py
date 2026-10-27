from copy import deepcopy

import pytest
from build_case import construct_world, replay_rounds

from kernel_workflow.rsi.replay_buffer.tests.test_simulator import world


def test_construction_preserves_original_measurements_and_does_not_relabel_history():
    original = world()
    before = deepcopy(original.to_dict())
    fixture, provenance = construct_world(original)
    assert original.to_dict() == before
    assert fixture.id.startswith("constructed/")
    assert "constructed" in provenance["topology"]
    assert fixture.root.measurement.primary_score == 1
    assert fixture.branches[0][0].measurement.primary_score == pytest.approx(1.01)
    assert fixture.branches[0][1].measurement.primary_score == pytest.approx(1.2)
    assert provenance["measurements"]["B0"]["origin"].startswith("reused root")


def test_current_simulator_computes_the_displayed_gain_and_second_round_retains_policy():
    fixture, _ = construct_world(world())
    rounds = replay_rounds((fixture, fixture))
    assert rounds[0]["incumbent_id"] == "neutral"
    assert rounds[0]["selected_id"] == "continue"
    assert rounds[0]["policy_changed"]
    assert rounds[1]["incumbent_id"] == rounds[0]["selected_id"]
    assert not rounds[1]["policy_changed"]
    trials = {trial["id"]: trial["result"] for trial in rounds[0]["trials"]}
    assert trials["neutral"]["summary"]["objective"] == pytest.approx(-0.01)
    assert trials["continue"]["summary"]["objective"] == pytest.approx(0.18)
    assert all(result["traces"][0]["charged_budget"] == 2 for key, result in trials.items() if key != "stop")
    assert all(not item["publication_allowed"] for item in rounds)
