import json
import time
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from kernel_workflow.rsi.evidence import read_json, read_jsonl, reference, write_jsonl
from kernel_workflow.rsi.loop.online import calibrate, explore
from kernel_workflow.rsi.loop.outer import cycle
from kernel_workflow.rsi.loop.qualification import release
from kernel_workflow.rsi.loop.run import IncompleteStage, execute_stage
from kernel_workflow.rsi.replay_buffer.cohorts import cohorts
from kernelwiki import KnowledgeClient, KnowledgeError

from .test_inner import experiment
from .test_online import online_config

__all__ = ["experiment", "online_config"]


@pytest.fixture
def cycle_config(experiment, online_config, tmp_path):
    pool, wiki, worlds = experiment
    _, _, online = online_config
    flattened = []
    for world in worlds:
        first = tuple(replace(node, measurement=replace(node.measurement, primary_score=2.6))
                      for node in world.branches[0])
        flattened.append(replace(world, branches=(first, *world.branches[1:])))
    (pool / "worlds.jsonl").unlink()
    write_jsonl(pool / "worlds.jsonl", [world.to_dict() for world in flattened])
    manifest = read_json(pool / "manifest.json")
    manifest["artifacts"]["worlds.jsonl"] = reference(pool / "worlds.jsonl").sha256
    manifest["cohorts"] = cohorts(flattened)
    (pool / "manifest.json").write_text(json.dumps(manifest))
    paths = []
    for kernel in ("train", "validation", "test-secret"):
        path = tmp_path / (kernel + "-online.yaml")
        path.write_text(yaml.safe_dump({**online, "kernel": kernel, "budget": 6}))
        paths.append(str(path))
    config = {"schema_version": 1, "pool": str(pool), "rounds": 2, "discovery": paths[:2], "holdout": paths[2:],
              "max_candidates": 1, "investigation": {"max_steps": 2, "model_calls": 1},
              "limits": {"direction_units": 36, "agent_calls": 102, "gpus": 2, "wallclock_seconds": 300}}
    path = tmp_path / "cycle.yaml"
    path.write_text(yaml.safe_dump(config))
    return path, wiki, config


class CycleAgent:
    def __init__(self, root):
        self.root = root
        self.calls = []
        self.training = []

    def __call__(self, role, payload, directory):
        self.calls.append(role)
        assert not (self.root / "final").exists()
        assert "test-secret" not in json.dumps(payload)
        if role == "preflight":
            return {"ok": True}
        if role == "reverse_analyst":
            evidence = payload["evidence"]
            return {"method_status": "adopted", "rationale": "Fixture source diff implements the supplied plan, not causal proof.",
                    "evidence": evidence["allowed_evidence"][:1]}
        if role == "reviewer":
            return {"decision": "approve", "reasons": ["bounded fixture scheduling hypothesis"]}
        assert role == "developer"
        assert payload["training"]["reverse_evidence"]
        assert all(item["attempt"]["kernel"] == "train" for item in payload["training"]["reverse_evidence"])
        assert any(item["method_reviews"] for item in payload["training"]["reverse_evidence"])
        self.training.append(payload["training"])
        return {"kind": "policy_update", "rationale": "Stop after a measured plateau in this fixture.",
                "policy": {"schema_version": 1, "rule_ids": ["plateau"], "top_k": 3, "workers": 1},
                "records": [{"id": "plateau", "title": "Plateau", "body": "Fixture evidence only.",
                             "applies_to": [], "evidence": payload["training"]["examples"][0]["evidence"],
                             "policy_rule": {"effect": "stop", "when": [{"field": "stagnation", "op": "gte", "value": 1}]}}]}


@pytest.mark.parametrize("online_regression", [False, True])
def test_two_outer_rounds_feed_new_policy_and_reverse_evidence_back(cycle_config, tmp_path, online_regression):
    path, wiki, config = cycle_config
    if online_regression:
        online = yaml.safe_load(Path(config["holdout"][0]).read_text())
        evaluator = Path(online["workspace"]) / "evaluate.py"
        evaluator.write_text(evaluator.read_text().replace('"optimized_ms": value',
            '"optimized_ms": (3 if "holdout-1-candidate" in str(Path.cwd()) and value < 2 else value)'))
    root = tmp_path / "cycle"
    active = KnowledgeClient(wiki).snapshot().revision
    agent = CycleAgent(root)
    result = cycle(path, wiki, root, agent=agent)
    expected = "final_qualification_failed" if online_regression else "fixture_validated"
    assert result["status"] == expected and not result["published"]
    qualification = read_json(root / "qualification/evidence.json")
    assert qualification["replay_pass"]
    assert qualification["paired_online_pass"] is not online_regression
    assert len(result["rounds"]) == 2 and len(agent.training) == 2
    assert not list(root.glob("round-*-normalized"))
    assert len(list((root / "stages").glob("round-*-pool.done.json"))) == 2
    assert len(list(root.glob("round-*-pool/manifest.json"))) == 2
    assert result["rounds"][0]["input_policy"] != result["rounds"][0]["output_policy"]
    assert result["rounds"][1]["input_policy"] == result["rounds"][0]["output_policy"]
    second = read_json(root / "round-2-discovery-1/result.json")
    assert second["policy_revision"] == result["rounds"][0]["output_policy"]
    assert len(list(read_jsonl(root / "round-2-pool/reverse.jsonl"))) > len(list(read_jsonl(root / "round-1-pool/reverse.jsonl")))
    assert KnowledgeClient(wiki).snapshot().revision == active
    calls = list(agent.calls)
    assert cycle(path, wiki, root, resume=True, agent=agent) == result
    assert agent.calls == calls
    with pytest.raises(KnowledgeError, match="only a real finally qualified cycle"):
        release(root, wiki)
    if not online_regression:
        forged = read_json(root / "result.json")
        forged.update(status="replay_validated", execution_kind="coordinated")
        (root / "result.json").write_text(json.dumps(forged))
        with pytest.raises(KnowledgeError, match="cannot be relabeled"):
            release(root, wiki)
    selected = root / "round-1-evolve/selected-policy.json"
    selected.write_text("changed after completion")
    with pytest.raises(KnowledgeError, match="artifact changed"):
        cycle(path, wiki, root, resume=True, agent=agent)


def test_uncertain_started_external_stage_is_not_repeated(cycle_config, tmp_path):
    path, wiki, _ = cycle_config
    root = tmp_path / "interrupted"
    calls = []

    def interrupted(*args, **kwargs):
        calls.append(args)
        explore(*args, **kwargs)
        raise RuntimeError("fixture interruption after external work, before completion receipt")

    with pytest.raises(RuntimeError, match="fixture interruption"):
        cycle(path, wiki, root, agent=CycleAgent(root), explorer=interrupted)
    with pytest.raises(IncompleteStage, match="no automatic Agent/GPU retry"):
        cycle(path, wiki, root, resume=True, agent=CycleAgent(root), explorer=interrupted)
    assert len(calls) == 1


def test_no_post_calibration_budget_means_no_real_work(cycle_config, tmp_path):
    path, wiki, config = cycle_config
    for name in (*config["discovery"], *config["holdout"]):
        value = yaml.safe_load(Path(name).read_text())
        value["execution"]["kind"] = "coordinated"
        Path(name).write_text(yaml.safe_dump(value))
    config["limits"] = {key: None for key in config["limits"]}
    path.write_text(yaml.safe_dump(config))
    calls = []
    result = cycle(path, wiki, tmp_path / "request", agent=lambda *args: calls.append(args),
                   explorer=lambda *args, **kwargs: calls.append(args))
    assert result["status"] == "calibration_required"
    assert result["agent_calls"] == result["direction_units"] == 0
    assert not calls and not (tmp_path / "request/stages").exists()


def test_stage_budget_and_deadline_are_checked_before_side_effects(tmp_path):
    calls = []
    limits = {"wallclock_seconds": 60, "direction_units": 1, "agent_calls": 2}
    with pytest.raises(KnowledgeError, match="budget cannot admit"):
        execute_stage(tmp_path, "too-large", {}, lambda directory: calls.append(directory), deadline=time.monotonic() + 60,
               limits=limits, reserve={"direction_units": 2, "agent_calls": 4})
    with pytest.raises(KnowledgeError, match="before launch"):
        execute_stage(tmp_path, "late", {}, lambda directory: calls.append(directory), deadline=time.monotonic() - 1, limits=limits)
    assert not calls and not (tmp_path / "stages").exists()


def test_calibration_is_one_normal_direction_and_never_full_cycle(online_config, tmp_path):
    path, wiki, config = online_config
    with pytest.raises(KnowledgeError, match="only one normal direction"):
        calibrate(path, wiki, tmp_path / "invalid")
    config.update(budget=1, gpu_ids=["0"])
    path.write_text(yaml.safe_dump(config))
    result = calibrate(path, wiki, tmp_path / "calibration")
    assert result["status"] == "calibrated" and result["execution_kind"] == "test_fixture"
    assert result["direction_units"] == result["gpus"] == 1
    assert result["agent_calls"] == 2 and not result["published"]


def test_cli_exposes_real_loop_and_bounded_investigation(capsys):
    from kernel_workflow.rsi.__main__ import main
    with pytest.raises(SystemExit) as stopped:
        main(["--help"])
    assert stopped.value.code == 0
    output = capsys.readouterr().out
    assert all(command in output for command in ("investigate", "cycle", "calibrate", "release"))


def test_runtime_agent_deadline_prevents_even_preflight(tmp_path):
    from kernel_workflow.rsi.loop.inner import RuntimeAgent
    agent = RuntimeAgent(node="not-invoked", claude="not-invoked", deadline=time.monotonic() - 1)
    with pytest.raises(KnowledgeError, match="before launch"):
        agent("preflight", {}, tmp_path / "no-call")
    assert not (tmp_path / "no-call").exists()
