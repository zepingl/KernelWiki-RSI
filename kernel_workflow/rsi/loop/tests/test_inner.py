import json
from dataclasses import replace

import pytest
from kernel_workflow.rsi.discovery import Measurement
from kernel_workflow.rsi.evidence import reference, write_json, write_jsonl
from kernel_workflow.rsi.loop.inner import AgentUnavailable, evolve
from kernel_workflow.rsi.replay_buffer.cohorts import cohorts
from kernel_workflow.rsi.replay_buffer.tests.test_simulator import node, world
from kernelwiki import KnowledgeClient, KnowledgeError


@pytest.fixture
def experiment(tmp_path):
    wiki = tmp_path / "wiki"
    (wiki / "knowledge").mkdir(parents=True)
    (wiki / "knowledge/policy.yaml").write_text("schema_version: 1\nrule_ids: []\ntop_k: 3\nworkers: 2\n")
    pool = tmp_path / "pool"
    pool.mkdir()
    worlds = []
    for kernel in ("train", "validation", "test-secret"):
        path = tmp_path / (kernel + ".json")
        path.write_text(json.dumps({"kernel": kernel, "synthetic_fixture": True}) + "\n")
        evidence = (reference(path),)
        data = world(kernel, kernel)
        first = []
        parent = "root"
        for index, score in enumerate((2.02, 2.4, 2.6, 2.8, 3.0, 3.2)):
            name = f"first-{index}"
            first.append(node(name, parent, score))
            parent = name
        data = replace(data, branches=(tuple(first),
                                       (node("second", "root", None), node("second-child", "second", None)),
                                       (node("third", "root", None),)))
        root = replace(data.root, input_evidence=evidence,
                       measurement=replace(data.root.measurement, evidence=evidence),
                       state=replace(data.root.state, evidence=evidence),
                       context=replace(data.root.context, evidence=evidence))
        branches = tuple(tuple(replace(item, input_evidence=evidence,
                                       measurement=replace(item.measurement, evidence=evidence),
                                       state=replace(item.state, evidence=evidence),
                                       context=replace(item.context, evidence=evidence))
                               for item in branch) for branch in data.branches)
        worlds.append(replace(data, root=root, branches=branches))
    write_jsonl(pool / "worlds.jsonl", [data.to_dict() for data in worlds])
    write_jsonl(pool / "attempts.jsonl", [])
    write_json(pool / "manifest.json", {"schema_version": 2, "splits": {"train": ["train"],
               "validation": ["validation"], "test": ["test-secret"]}, "budgets": [1, 2, 4, 6],
               "objective": {"beta_cost": 0.01, "beta_parallel": 0},
               "cohorts": cohorts(worlds), "artifacts": {name: reference(pool / name).sha256
                                                          for name in ("worlds.jsonl", "attempts.jsonl")}})
    return pool, wiki, worlds


class FakeAgent:
    def __init__(self, kind="policy_update", effect="prefer", invalid=False):
        self.calls = []
        self.kind = kind
        self.effect = effect
        self.invalid = invalid

    def __call__(self, role, payload, directory):
        self.calls.append(role)
        assert "test-secret" not in json.dumps(payload)
        assert '"kernel": "validation"' not in json.dumps(payload)
        if role == "preflight":
            return {"ok": True}
        if role == "reviewer":
            return {"decision": "approve", "reasons": ["synthetic review fixture"]}
        if self.invalid:
            return {"invalid": True}
        return {"kind": self.kind, "rationale": "Synthetic exploration hypothesis",
                "policy": {"schema_version": 1, "rule_ids": ["continue"], "top_k": 3, "workers": 1},
                "records": [{"id": "continue", "title": "Continue", "body": "Synthetic fixture.",
                             "applies_to": [], "evidence": payload["training"]["examples"][0]["evidence"],
                             "policy_rule": {"effect": self.effect,
                                             "when": [{"field": "node_kind", "op": "eq", "value": "leaf"}]}}]}


def test_evolution_selects_working_policy_without_changing_kernelwiki(experiment, tmp_path):
    pool, wiki, _ = experiment
    client = KnowledgeClient(wiki)
    original = client.snapshot().revision
    agent = FakeAgent()
    result = evolve(pool, wiki, tmp_path / "round", agent=agent)
    assert agent.calls.count("developer") == 4 and agent.calls.count("reviewer") == 4
    assert result["status"] == "working_policy_selected" and not result["published"]
    assert result["selected"] == 1 and not result["test_evaluated"]
    assert client.snapshot().policy.rule_ids == ()
    assert client.snapshot().revision == original
    selected = json.loads((tmp_path / "round/selected-policy.json").read_text())
    assert selected["policy"]["rule_ids"] == ["continue"]


def test_inner_loop_cannot_publish_without_outer_final_qualification(experiment, tmp_path):
    pool, wiki, _ = experiment
    agent = FakeAgent()
    with pytest.raises(KnowledgeError, match="final cycle qualification"):
        evolve(pool, wiki, tmp_path / "round", agent=agent, publish=True)
    assert agent.calls == []


@pytest.mark.parametrize("kind", ["evidence_update", "needs_online"])
def test_nonpolicy_proposals_do_not_count_as_self_improvement(experiment, tmp_path, kind):
    pool, wiki, _ = experiment
    agent = FakeAgent(kind=kind)
    result = evolve(pool, wiki, tmp_path / "round", agent=agent)
    assert result["status"] == "no_change" and not result["test_evaluated"]
    assert agent.calls == ["preflight", "developer", "developer", "developer", "developer"]


def test_invalid_outputs_consume_opportunities(experiment, tmp_path):
    pool, wiki, _ = experiment
    agent = FakeAgent(invalid=True)
    result = evolve(pool, wiki, tmp_path / "round", agent=agent)
    assert agent.calls.count("developer") == 4
    assert all(item["status"] == "invalid_proposal" for item in result["opportunities"])
    assert not result["test_evaluated"]


def test_reviewer_rejection_blocks_replay(experiment, tmp_path):
    pool, wiki, _ = experiment
    agent = FakeAgent()

    def rejected(role, payload, directory):
        response = agent(role, payload, directory)
        return {"decision": "reject", "reasons": ["unsupported rule"]} if role == "reviewer" else response

    result = evolve(pool, wiki, tmp_path / "round", agent=rejected, max_candidates=1)
    assert result["opportunities"][0]["status"] == "review_rejected"
    assert not (tmp_path / "round/candidate-1-train.json").exists()


def test_inner_loop_does_not_use_already_inspected_test_data(experiment, tmp_path):
    pool, wiki, worlds = experiment
    test_world = worlds[-1]
    branches = tuple(tuple(replace(item, measurement=Measurement("failed")) for item in branch)
                     for branch in test_world.branches)
    worlds[-1] = replace(test_world, branches=branches)
    (pool / "worlds.jsonl").unlink()
    write_jsonl(pool / "worlds.jsonl", [data.to_dict() for data in worlds])
    manifest = json.loads((pool / "manifest.json").read_text())
    manifest["artifacts"]["worlds.jsonl"] = reference(pool / "worlds.jsonl").sha256
    (pool / "manifest.json").write_text(json.dumps(manifest))
    agent = FakeAgent()
    result = evolve(pool, wiki, tmp_path / "round", agent=agent)
    assert result["status"] == "working_policy_selected" and not result["test_evaluated"]
    assert result["selected"] == 1 and not result["published"]
    assert len(list((tmp_path / "round").glob("candidate-*-test.json"))) == 0
    assert not (wiki / ".kernelwiki-rsi/active.json").exists()


def test_unavailable_agent_stops_without_fallback(experiment, tmp_path):
    pool, wiki, _ = experiment
    calls = []

    def unavailable(role, payload, directory):
        calls.append(role)
        raise AgentUnavailable("unavailable fixture")

    with pytest.raises(AgentUnavailable):
        evolve(pool, wiki, tmp_path / "round", agent=unavailable)
    assert calls == ["preflight"]
    assert json.loads((tmp_path / "round/result.json").read_text())["status"] == "agent_failed"


def test_pool_mutation_during_agent_call_blocks_downstream_gates(experiment, tmp_path):
    pool, wiki, _ = experiment
    agent = FakeAgent()

    def mutating(role, payload, directory):
        response = agent(role, payload, directory)
        if role == "developer":
            (pool / "worlds.jsonl").write_text("modified")
        return response

    from kernelwiki import KnowledgeError
    with pytest.raises(KnowledgeError, match="artifact changed"):
        evolve(pool, wiki, tmp_path / "round", agent=mutating)
    assert not (tmp_path / "round/incumbent-validation.json").exists()
    assert not (wiki / ".kernelwiki-rsi/active.json").exists()


def test_complete_proposals_and_matched_feedback_reach_every_revision(experiment, tmp_path):
    pool, wiki, _ = experiment
    delegate = FakeAgent()
    proposals = []
    def agent(role, payload, directory):
        if role != "developer":
            return delegate(role, payload, directory)
        assert [item["proposal"] for item in payload["history"]] == proposals
        assert [item["opportunity"] for item in payload["history"]] == list(range(1, len(proposals) + 1))
        if not proposals:
            proposal = {"invalid": "complete rejected proposal must still be available"}
        else:
            proposal = delegate(role, payload, directory)
        if len(proposals) >= 2:
            assert payload["editing_base"]["policy"]["rule_ids"] == ["continue"]
            assert payload["history"][0]["feedback"]["status"] == "invalid_proposal"
            assert payload["history"][1]["review"]["decision"] == "approve"
            assert payload["history"][1]["feedback"]["training_examples"][0]["trajectories"]
        proposals.append(proposal)
        return proposal
    result = evolve(pool, wiki, tmp_path / "round", agent=agent)
    assert len(proposals) == 4 and result["selected"] == 2


def test_incomplete_cohort_blocks_agent_calls_not_just_publication(experiment, tmp_path):
    pool, wiki, worlds = experiment
    worlds = [replace(data, branches=(data.branches[0][:1],)) for data in worlds]
    (pool / "worlds.jsonl").unlink()
    write_jsonl(pool / "worlds.jsonl", [data.to_dict() for data in worlds])
    manifest = json.loads((pool / "manifest.json").read_text())
    manifest["cohorts"] = cohorts(worlds)
    manifest["artifacts"]["worlds.jsonl"] = reference(pool / "worlds.jsonl").sha256
    (pool / "manifest.json").write_text(json.dumps(manifest))
    agent = FakeAgent()
    result = evolve(pool, wiki, tmp_path / "round", agent=agent)
    assert result["status"] == "insufficient_primary_support"
    assert agent.calls == [] and not result["published"]


def test_unsupported_candidate_does_not_get_a_cheap_cost_advantage(experiment, tmp_path):
    pool, wiki, worlds = experiment
    worlds = [replace(data, branches=(data.branches[0][:3], *data.branches[1:])) for data in worlds]
    (pool / "worlds.jsonl").unlink()
    write_jsonl(pool / "worlds.jsonl", [data.to_dict() for data in worlds])
    manifest = json.loads((pool / "manifest.json").read_text())
    manifest["cohorts"] = cohorts(worlds)
    manifest["artifacts"]["worlds.jsonl"] = reference(pool / "worlds.jsonl").sha256
    (pool / "manifest.json").write_text(json.dumps(manifest))
    result = evolve(pool, wiki, tmp_path / "round", agent=FakeAgent())
    assert result["status"] == "no_change"
    assert all(item["status"] == "unsupported_replay" for item in result["opportunities"])
