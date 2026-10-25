import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from kernel_workflow.rsi.discovery import Attempt, DiscoveryWorld
from kernel_workflow.rsi.evidence import reference, snapshot, write_json, write_jsonl
from kernel_workflow.rsi.harness import Policy
from kernel_workflow.rsi.loop.evaluator import FixedEvaluator
from kernel_workflow.rsi.loop.online import execution_environment, explore
from kernel_workflow.rsi.replay_buffer.build import freeze
from kernel_workflow.rsi.replay_buffer.simulator import replay
from kernel_workflow.rsi.replay_buffer.store import open_buffer
from kernel_workflow.rsi.reverse_engineering.sources.capture import load_online
from kernel_workflow.rsi.reverse_engineering.trace import trace
from kernel_workflow.rsi.runtime import run_process
from kernelwiki.knowledge import KnowledgeError

REPO = Path(__file__).resolve().parents[4]
NODE = shutil.which("node") or "/shared_nfs/zepingl/geak_audit/tools/node"


def test_real_runtime_and_restricted_lane_boundaries():
    if not Path(NODE).is_file():
        pytest.skip("Node is required for standalone runtime integration")
    process = subprocess.run([NODE, str(Path(__file__).parent / "fixtures/runtime_check.mjs"), str(REPO)],
                             capture_output=True, text=True, timeout=30, check=False)
    assert process.returncode == 0, process.stdout + process.stderr


@pytest.fixture
def online_config(tmp_path):
    source = tmp_path / "source"
    (source / "kernel_src").mkdir(parents=True)
    (source / "kernel_src/k.txt").write_text("2")
    evaluator = source / "evaluate.py"
    evaluator.write_text('''import json
from pathlib import Path
value = float(Path("kernel_src/k.txt").read_text())
Path("ut_result.json").write_text(json.dumps({
    "correctness_pass": True, "measurement_valid": True, "performance_scope_complete": True,
    "runtime_fingerprint": "fixture-runtime-v1",
    "per_case": [{"name": "case", "baseline_ms": 2, "optimized_ms": value, "measurement_valid": True}]
}))
''')
    commandment = tmp_path / "COMMANDMENT.md"
    commandment.write_text("# Fixture\nRun the fixed evaluator.\n")
    worker = tmp_path / "fake_claude"
    worker.write_text('''#!/usr/bin/python3
import json
from pathlib import Path
import re
import shutil
import sys
prompt = sys.stdin.read()
if "PHASE=plan_round" in prompt:
    print(json.dumps({"stop": False, "directions": [{"id": "one", "title": "Fixture", "specialty": "compute", "prompt": "Fixture"}]}))
else:
    parent = re.search(r"^- CANONICAL: (.*)$", prompt, re.M)[1]
    workspace = re.search(r"^- KERNEL_PATH: (.*)$", prompt, re.M)[1]
    shutil.copytree(parent, workspace)
    Path(workspace, "kernel_src/k.txt").write_text("1")
    print(json.dumps({"status": "improved", "speedup_geomean": 999, "measurement_valid": True}))
''')
    worker.chmod(0o755)
    config = {"kernel": "fixture", "workspace": str(source), "state": {"operator": "gemm"},
              "execution": {"kind": "test_fixture", "wallclock_limit_seconds": 60},
              "source_paths": ["kernel_src"], "commandment": str(commandment), "gpu_ids": ["0", "1"],
              "budget": 2, "runtime": {"node": NODE, "claude": str(worker), "timeout_seconds": 15},
              "evaluator": {"command": [sys.executable, "evaluate.py"], "result_file": "ut_result.json",
                            "contract_files": ["evaluate.py"], "case_weights": {"case": 1}, "timeout_seconds": 10}}
    path = tmp_path / "online.yaml"
    path.write_text(yaml.safe_dump(config))
    wiki = tmp_path / "online-wiki"
    (wiki / "knowledge").mkdir(parents=True)
    (wiki / "knowledge/policy.yaml").write_text("schema_version: 1\nrule_ids: []\ntop_k: 3\nworkers: 2\n")
    return path, wiki, config


def test_online_to_discovery_world_uses_host_measurements_not_agent_claims(online_config, tmp_path):
    if not Path(NODE).is_file():
        pytest.skip("Node is required for standalone runtime integration")
    path, wiki, config = online_config
    result = explore(path, wiki, tmp_path / "online")
    assert result["trace"]["attempted_nodes"] == 2 and result["trace"]["charged_budget"] == 2
    assert result["trace"]["decision_batches"] == 2
    assert result["trace"]["best_quality"] == pytest.approx(1.0)
    assert not result["knowledge_changed"]
    assert not Path(config["workspace"], "ut_result.json").exists()
    world = DiscoveryWorld.from_dict(json.loads((tmp_path / "online/worlds.jsonl").read_text()))
    assert len(world.branches) == 2
    from kernel_workflow.rsi.__main__ import main
    direct = tmp_path / "standalone-online.json"
    assert main(["trace", "--online", str(tmp_path / "online"), "--attempt",
                 world.branches[0][0].attempt_id, "--output", str(direct)]) == 0
    assert json.loads(direct.read_text())["report_version"] == 1
    for branch in world.branches:
        assert branch[0].measurement.primary_score == 2.0
    assert len(list((tmp_path / "online/generations").glob("*/capture/host-commit-director.json"))) == 2
    pool = tmp_path / "base-pool"
    pool.mkdir()
    write_jsonl(pool / "worlds.jsonl", [])
    write_jsonl(pool / "attempts.jsonl", [])
    write_json(pool / "manifest.json", {"schema_version": 2, "splits": {"train": ["fixture"],
               "validation": [], "test": []}, "budgets": [1, 2, 4, 6],
               "objective": {"beta_cost": 0.01, "beta_parallel": 0},
               "artifacts": {name: reference(pool / name).sha256
                             for name in ("worlds.jsonl", "attempts.jsonl")}})
    freeze(pool, [tmp_path / "online"], tmp_path / "next-pool")
    frozen_worlds = open_buffer(tmp_path / "next-pool").worlds
    assert frozen_worlds == (world,)
    report = trace(open_buffer(tmp_path / "next-pool").subject(world.branches[0][0].attempt_id)).to_trace()
    assert report["effect"]["director_status"] == "not_run"
    assert report["effect"]["retained"] is True
    assert report["effect"]["installed_scope"] == "generation_artifact"
    assert len(report["boundaries"]) == 2
    with pytest.raises(KnowledgeError, match="duplicate world"):
        freeze(tmp_path / "next-pool", [tmp_path / "online"], tmp_path / "duplicate")
    (tmp_path / "online/worlds.jsonl").write_text("modified")
    with pytest.raises(KnowledgeError, match="artifact changed"):
        freeze(pool, [tmp_path / "online"], tmp_path / "tampered")


def test_fixed_evaluator_rejects_contract_and_source_mutations(online_config, tmp_path):
    _, _, config = online_config
    root = Path(config["workspace"])
    evaluator = FixedEvaluator(config["evaluator"], root, config["source_paths"])
    (root / "evaluate.py").write_text("raise SystemExit(0)")
    assert evaluator.measure(root, tmp_path / "changed-contract", "0").status == "unknown"
    (root / "kernel_src/link.txt").symlink_to(root / "evaluate.py")
    with pytest.raises(KnowledgeError, match="symlinks"):
        snapshot(root, config["source_paths"])


def test_evaluator_output_cannot_delete_its_input(online_config):
    _, _, config = online_config
    with pytest.raises(KnowledgeError, match="overwrite"):
        FixedEvaluator({**config["evaluator"], "result_file": "evaluate.py"}, config["workspace"], config["source_paths"])


def test_evaluator_requires_runtime_identity(online_config, tmp_path):
    _, _, config = online_config
    root = Path(config["workspace"])
    path = root / "evaluate.py"
    path.write_text(path.read_text().replace('    "runtime_fingerprint": "fixture-runtime-v1",\n', ""))
    evaluator = FixedEvaluator(config["evaluator"], root, config["source_paths"])
    assert evaluator.measure(root, tmp_path / "missing-runtime", "0").status == "unknown"


def test_unknown_and_failed_measurements_do_not_erase_source(online_config, tmp_path):
    _, _, config = online_config
    root = Path(config["workspace"])
    path = root / "evaluate.py"
    path.write_text(path.read_text().replace('"correctness_pass": True', '"correctness_pass": value > 0'))
    evaluator = FixedEvaluator(config["evaluator"], root, config["source_paths"])
    assert evaluator.measure(root, tmp_path / "root-measure", "0").status == "qualified"
    (root / "kernel_src/k.txt").write_text("-1")
    failed = evaluator.measure(root, tmp_path / "failed", "0")
    assert failed.status == "failed" and failed.reason == "correctness_failed"
    assert failed.source == snapshot(root, config["source_paths"])["fingerprint"]
    path.write_text("raise SystemExit(1)")
    unknown = evaluator.measure(root, tmp_path / "unknown", "0")
    assert unknown.status == "unknown" and unknown.source == failed.source


def test_runtime_is_pinned_across_parent_and_candidate(online_config, tmp_path, monkeypatch):
    _, _, config = online_config
    root = Path(config["workspace"])
    path = root / "evaluate.py"
    path.write_text("import os\n" + path.read_text().replace('"fixture-runtime-v1"', 'os.environ["FIXTURE_RUNTIME"]'))
    evaluator = FixedEvaluator(config["evaluator"], root, config["source_paths"])
    monkeypatch.setenv("FIXTURE_RUNTIME", "runtime-A")
    assert evaluator.measure(root, tmp_path / "runtime-a", "0").status == "qualified"
    monkeypatch.setenv("FIXTURE_RUNTIME", "runtime-B")
    result = evaluator.measure(root, tmp_path / "runtime-b", "0")
    assert result.status == "unknown" and "differs from the root" in result.reason


def test_candidate_baseline_drift_cannot_improve_quality(online_config, tmp_path, monkeypatch):
    _, _, config = online_config
    root = Path(config["workspace"])
    path = root / "evaluate.py"
    path.write_text("import os\n" + path.read_text().replace('"baseline_ms": 2',
                                                           '"baseline_ms": float(os.environ["FIXTURE_BASELINE"])'))
    evaluator = FixedEvaluator(config["evaluator"], root, config["source_paths"])
    monkeypatch.setenv("FIXTURE_BASELINE", "2")
    baseline = evaluator.measure(root, tmp_path / "fixed-root", "0")
    monkeypatch.setenv("FIXTURE_BASELINE", "200")
    candidate = evaluator.measure(root, tmp_path / "drifted-baseline", "0")
    assert baseline.primary_score == candidate.primary_score == 1.0
    assert baseline.contract == candidate.contract


def test_failed_candidate_continuation_online_freeze_replay(online_config, tmp_path):
    path, wiki, config = online_config
    config["budget"] = 3
    config["state"]["backend"] = "triton"
    path.write_text(yaml.safe_dump(config))
    evaluator = Path(config["workspace"], "evaluate.py")
    evaluator.write_text(evaluator.read_text().replace('"correctness_pass": True', '"correctness_pass": value > 0'))
    worker = Path(config["runtime"]["claude"])
    worker.write_text(worker.read_text().replace('write_text("1")',
                                               'write_text("-1" if "/node-1/" in workspace else "1")'))
    result = explore(path, wiki, tmp_path / "online")
    world = DiscoveryWorld.from_dict(json.loads((tmp_path / "online/worlds.jsonl").read_text()))
    failed, continuation = world.branches[0]
    assert failed.measurement.status == "failed"
    assert failed.state.fingerprint != world.root.state.fingerprint
    assert continuation.input_state == failed.state.fingerprint
    assert continuation.input_context == failed.context.revision
    request = json.loads((tmp_path / "online/generations/node-3/request.json").read_text())
    observation = request["parent"]["context"]["branch_observations"][0]
    assert observation["measurement"]["status"] == "failed"
    assert observation["generation_result"]["plan"]["directions"][0]["title"] == "Fixture"
    args = json.loads((tmp_path / "online/generations/node-3/args.json").read_text())
    assert args["rsi_context"]["KK_LANGUAGE"] == "triton"
    assert json.loads(json.dumps(replay(world, Policy(), 3))) == json.loads(json.dumps(result["trace"]))
    pool = tmp_path / "base-pool"
    write_jsonl(pool / "worlds.jsonl", [])
    write_jsonl(pool / "attempts.jsonl", [])
    write_json(pool / "manifest.json", {"schema_version": 2, "splits": {"train": ["fixture"],
               "validation": [], "test": []}, "budgets": [1, 2, 4, 6],
               "objective": {"beta_cost": 0.01, "beta_parallel": 0},
               "artifacts": {name: reference(pool / name).sha256 for name in ("worlds.jsonl", "attempts.jsonl")}})
    freeze(pool, [tmp_path / "online"], tmp_path / "next-pool")
    worlds = open_buffer(tmp_path / "next-pool").worlds
    attempts = [Attempt.from_dict(json.loads(line))
                for line in (tmp_path / "next-pool/attempts.jsonl").read_text().splitlines()]
    frozen_child = next(item for item in attempts if item.id == continuation.attempt_id)
    assert frozen_child.parent_source == failed.state.fingerprint
    assert frozen_child.parent_attempt_id == failed.attempt_id
    assert worlds == (world,)


def test_archived_online_inputs_survive_later_config_edits(online_config, tmp_path):
    path, wiki, _ = online_config
    explore(path, wiki, tmp_path / "online")
    path.write_text("later unrelated experiment configuration")
    result, _, _ = load_online(tmp_path / "online")
    assert result["status"] == "online_executed"


def test_working_policy_changes_only_scheduling_not_root_context(online_config, tmp_path):
    path, wiki, _ = online_config
    baseline = explore(path, wiki, tmp_path / "baseline-policy")
    working = Policy.from_dict({"policy": {"schema_version": 1, "rule_ids": [], "top_k": 3, "workers": 1}, "records": []})
    candidate = explore(path, wiki, tmp_path / "working-policy", policy=working)
    assert baseline["policy_revision"] != candidate["policy_revision"]
    for key in ("reference_revision", "root_source", "root_context", "evaluator_protocol", "evaluator_runtime"):
        assert baseline[key] == candidate[key]
    assert not (tmp_path / "working-policy/frozen_knowledge/knowledge/policy.yaml").exists()


def test_coordinated_execution_fails_closed_without_placement_and_readonly_inputs(online_config, monkeypatch):
    _, wiki, config = online_config
    config["execution"]["kind"] = "coordinated"
    names = ("NRC_RUN_ID", "NRC_NODE", "NRC_CPUSET", "NRC_CPU_CORES", "NRC_PHYSICAL_GPU_PCI", "NRC_LOGICAL_GPU_IDS")
    for name in names:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(KnowledgeError, match="missing live NRC"):
        execution_environment(config, wiki)
    for name, value in zip(names, ("fixture-run", "fixture-node", "0-3", "4", "pci-a,pci-b", "0,1")):
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0")
    with pytest.raises(KnowledgeError, match="second device"):
        execution_environment(config, wiki)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES")
    with pytest.raises(KnowledgeError, match="read-only"):
        execution_environment(config, wiki)
    monkeypatch.setattr("kernel_workflow.rsi.loop.online.os.statvfs", lambda path: SimpleNamespace(f_flag=os.ST_RDONLY))
    environment = execution_environment(config, wiki)
    assert environment["device"] == {"node": "fixture-node", "pci": "pci-a"}


def test_expired_deadline_starts_no_subprocess_or_output(tmp_path):
    with pytest.raises(KnowledgeError, match="budget exhausted before launch"):
        run_process([sys.executable, "-c", "raise SystemExit(99)"], tmp_path, tmp_path / "late",
                    timeout=20, deadline=time.monotonic() - 1)
    assert not (tmp_path / "late").exists()


def test_failed_root_keeps_frozen_code_and_config_without_starting_generation(online_config, tmp_path):
    path, wiki, config = online_config
    config["evaluator"]["case_weights"] = {"missing_case": 1}
    path.write_text(yaml.safe_dump(config))
    original_config = path.read_bytes()
    output = tmp_path / "failed-root"
    with pytest.raises(KnowledgeError, match="root baseline did not qualify"):
        explore(path, wiki, output)
    path.write_text("changed after the failed baseline\n")
    assert (output / "frozen_inputs/online-config").read_bytes() == original_config
    assert (output / "frozen_inputs/GEAK/kernel_workflow/rsi/glm7_evaluator.py").is_file()
    assert (output / "baseline/measurement.json").is_file()
    assert not (output / "generations").exists()


def test_evaluator_checks_physical_device_and_protocol(online_config, tmp_path):
    _, _, config = online_config
    root = Path(config["workspace"])
    device = {"node": "fixture-node", "pci": "0000:01:00.0"}
    bound = FixedEvaluator(config["evaluator"], root, config["source_paths"], device=device)
    unbound = FixedEvaluator(config["evaluator"], root, config["source_paths"])
    assert bound.protocol != unbound.protocol
    result = bound.measure(root, tmp_path / "device-missing", "0")
    assert result.status == "unknown" and "physical GPU" in result.reason
