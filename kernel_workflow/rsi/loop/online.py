"""Explicit online discovery using the same policy and a fixed GEAK generation worker."""

from __future__ import annotations

import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import yaml
from kernelwiki import KnowledgeClient
from kernelwiki.knowledge import KernelState, KnowledgeError, digest, read_markdown

from ..discovery import (
    ContextSnapshot,
    DiscoveryNode,
    DiscoveryWorld,
    GenerationSpec,
    Measurement,
    SourceState,
)
from ..evidence import (
    persist,
    read_json,
    reference,
    snapshot,
    verify,
    write_json,
    write_jsonl,
)
from ..harness import Policy, Rollout
from ..reverse_engineering.sources.capture import load_online
from ..runtime import EFFORT, MODEL, REPO_ROOT, WORKFLOW_ROOT, run_process
from .evaluator import FixedEvaluator


class OnlineExecutor:
    def __init__(self, config, output, evaluator, reference_knowledge):
        self.config = config
        self.output = Path(output).resolve()
        self.evaluator = evaluator
        self.reference_knowledge = str(reference_knowledge)

    def execute(self, parent, generation_spec):
        if generation_spec.model != MODEL or generation_spec.effort != EFFORT:
            raise KnowledgeError("online discovery must use the frozen Opus 5.5/high Agent")
        directory = self.output / "generations" / parent["request_id"]
        directory.mkdir(parents=True, exist_ok=False)
        before = snapshot(parent["workspace"], self.config["source_paths"], directory / "capture")
        if before["fingerprint"] != parent["source"]:
            raise KnowledgeError("selected parent source changed before generation")
        request = persist(directory / "request.json", {"parent": parent, "generation": asdict(generation_spec),
                                                       "source": before})
        context = {"CANONICAL": parent["workspace"], "EVAL_DIR": str(directory),
                   "COMMANDMENT": str(Path(self.config["commandment"]).resolve()), "BASELINE_PER_CASE": [],
                   "profileSummary": None, "history": {"insights": parent["context"]["branch_observations"]},
                   "KERNEL_KNOWLEDGE_DIR": self.reference_knowledge,
                   "KK_OPERATOR": self.config["state"].get("operator", ""),
                   "KK_LANGUAGE": self.config["state"].get("backend", ""),
                   "KK_REFS": [], "GFX": self.config["state"].get("gfx", ""), "DEVICE_TARGET": "",
                   "PHYSICAL_CU_COUNT": 0, "ROOFLINE_STATUS": "unknown"}
        if self.config.get("codebase_context"):
            shutil.copyfile(self.config["codebase_context"], directory / "codebase_context.md")
        args = {"kernel_path": parent["workspace"], "workflow_dir": str(WORKFLOW_ROOT),
                "rsi_generation": generation_spec.mode, "rsi_context": context, "gpu_ids": str(parent["gpu_id"]),
                "warm_start": "off", "update_experience": "off", "use_learned_kb": "false",
                "use_expert_skills": "false", "candidate_floor": 1e-12,
                "rsi_capture": {"directory": str(directory / "capture"),
                                "allowed_roots": [str(directory), parent["workspace"]],
                                "source_paths": self.config["source_paths"]}}
        write_json(directory / "args.json", args)
        runtime = self.config["runtime"]
        registry = {"default_profile": "rsi", "profiles": {"rsi": {"agent": "claude"}},
                    "models": {MODEL: {"id": MODEL}}, "agents": {"claude": {
                    "bin": runtime["claude"], "prompt": "stdin", "dialect": "anthropic", "model_flag": "--model",
                    "args": ["-p", "--bare", "--tools", "Bash,Read,Write", "--dangerously-skip-permissions",
                             "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                             "--setting-sources", "", "--disable-slash-commands", "--no-session-persistence",
                             "--effort", EFFORT]}}}
        write_json(directory / "registry.json", registry)
        command = [runtime["node"], str(REPO_ROOT / "interface/runtime/engine/run_workflow.mjs"),
                   str(WORKFLOW_ROOT / "kernel_lane.js"), "--registry", str(directory / "registry.json"),
                   "--agent", "claude", "--model", MODEL, "--concurrency", "1", "--args-file",
                   str(directory / "args.json"), "--result-file", str(directory / "result.json"),
                   "--agent-timeout-ms", str(runtime.get("timeout_seconds", 3600) * 1000)]
        process = run_process(command, directory, directory / "runtime",
                              timeout=runtime.get("timeout_seconds", 3600) + 30,
                              environment={"GEAK_SCHEMA_RETRIES": "0", "GEAK_KB_READ_ONLY": "1"},
                              deadline=self.evaluator.deadline)
        after = snapshot(parent["workspace"], self.config["source_paths"], directory / "capture")
        if after["fingerprint"] != before["fingerprint"]:
            raise KnowledgeError("generation changed its immutable parent")
        result = read_json(directory / "result.json") if (directory / "result.json").exists() else None
        measurement = Measurement("unknown", before["fingerprint"], evidence=(request, process["evidence"]),
                                  reason="generation_did_not_return_a_candidate")
        retained = Path(parent["workspace"])
        candidate_source = None
        retention = "restored_parent"
        verify_reference = None
        if process["exit_code"] == 0 and result and result.get("status") == "generated":
            verify_reference = result.get("result", {}).get("evidence")
            if not verify_reference:
                raise KnowledgeError("generation is missing durable candidate boundary evidence")
            from kernelwiki.knowledge import EvidenceRef
            boundary = json.loads(verify(EvidenceRef.from_dict(verify_reference)))
            candidate = Path(result["candidate_workspace"]).resolve()
            if directory not in candidate.parents:
                raise KnowledgeError("candidate escaped its generation workspace")
            candidate_source = snapshot(candidate, self.config["source_paths"], directory / "capture")
            if candidate_source["fingerprint"] != boundary["sources"]["candidate"]["fingerprint"]:
                raise KnowledgeError("candidate changed after its generation boundary")
            retained = directory / "verify"
            materialize = run_process(["bash", str(WORKFLOW_ROOT / "scripts/materialize_workspace.sh"),
                                       "--src", str(candidate), "--dst", str(retained)], directory,
                                      directory / "materialize", timeout=120, deadline=self.evaluator.deadline)
            if materialize["exit_code"] != 0:
                raise KnowledgeError("verification workspace materialization failed")
            if snapshot(retained, self.config["source_paths"])["fingerprint"] != candidate_source["fingerprint"]:
                raise KnowledgeError("verification does not contain the generated candidate")
            retention = "candidate"
        return {"parent": parent, "before": before, "request": request, "process": process,
                "result": result, "retained": retained, "candidate_source": candidate_source,
                "retention": retention, "verify_reference": verify_reference, "measurement": measurement}

    def finish(self, outcome):
        parent, before, request = outcome["parent"], outcome["before"], outcome["request"]
        process, result, retained = outcome["process"], outcome["result"], outcome["retained"]
        candidate_source, retention = outcome["candidate_source"], outcome["retention"]
        measurement = outcome["measurement"]
        directory = self.output / "generations" / parent["request_id"]
        if candidate_source:
            measurement = self.evaluator.measure(retained, directory / "evaluation", self.config["gpu_ids"][0])
            persist(directory / "capture/host-candidate-verify.json", {
                "schema_version": 1, "kind": "candidate_verify", "stage": "evaluated",
                "generation_boundary": outcome["verify_reference"], "measurement": asdict(measurement),
                "sources": {"candidate": candidate_source, "verified": snapshot(retained, self.config["source_paths"])}})
        retained_source = snapshot(retained, self.config["source_paths"], directory / "capture")
        disposition = persist(directory / "capture/host-commit-director.json", {
            "schema_version": 1, "kind": "commit_director", "installed_scope": "generation_artifact",
            "installed_source": retained_source, "retention": retention,
            "candidate_source": candidate_source, "parent_source": before,
            "director_status": "not_run", "measurement_status": measurement.status})
        audit = tuple(reference(path) for path in sorted((directory / "capture").glob("*.json")))
        details = {"attempt": parent["request_id"], "parent_source": before["fingerprint"],
                   "candidate_source": candidate_source["fingerprint"] if candidate_source else None,
                   "retained_source": retained_source["fingerprint"], "retention": retention,
                   "generation_result": result, "measurement": asdict(measurement),
                   "process": {key: value for key, value in process.items() if key != "evidence"},
                   "stdout": (directory / "runtime/stdout.log").read_text(errors="replace"),
                   "stderr": (directory / "runtime/stderr.log").read_text(errors="replace"),
                   "evidence": [asdict(item) for item in audit]}
        return {"id": parent["request_id"], "workspace": str(retained), "measurement": measurement,
                "state": SourceState(retained_source["fingerprint"], (disposition,)),
                "candidate_source": candidate_source["fingerprint"] if candidate_source else None,
                "input_evidence": (request, process["evidence"], *audit), "details": details}


def execution_environment(config, wiki):
    execution = config.get("execution", {})
    limit = execution.get("wallclock_limit_seconds")
    if type(limit) is not int or limit < 1:
        raise KnowledgeError("online execution requires an explicit positive wall-clock limit")
    if execution.get("kind") == "test_fixture":
        return {"kind": "test_fixture", "wallclock_limit_seconds": limit, "device": None}
    if execution.get("kind") != "coordinated":
        raise KnowledgeError("real online work requires the existing node resource coordinator")
    required = ("NRC_RUN_ID", "NRC_NODE", "NRC_CPUSET", "NRC_CPU_CORES", "NRC_PHYSICAL_GPU_PCI", "NRC_LOGICAL_GPU_IDS")
    if any(not os.environ.get(key) for key in required):
        raise KnowledgeError("missing live NRC allocation/physical-device environment; no GPU work started")
    logical = os.environ["NRC_LOGICAL_GPU_IDS"].split(",")
    physical = os.environ["NRC_PHYSICAL_GPU_PCI"].split(",")
    if list(map(str, config["gpu_ids"])) != logical or len(logical) != len(physical):
        raise KnowledgeError("online GPU IDs must be the coordinator's logical device order")
    if any(os.environ.get(key) for key in ("ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")):
        raise KnowledgeError("a second device visibility filter would invalidate coordinator placement")
    roots = {Path(config["workspace"]), REPO_ROOT, Path(wiki), Path(config["commandment"])}
    if config.get("codebase_context"):
        roots.add(Path(config["codebase_context"]))
    if any(not os.statvfs(path).f_flag & os.ST_RDONLY for path in roots):
        raise KnowledgeError("real online source, GEAK, reference knowledge and context mounts must be read-only")
    return {"kind": "coordinated", "wallclock_limit_seconds": limit,
            "run_id": os.environ["NRC_RUN_ID"], "node": os.environ["NRC_NODE"],
            "cpuset": os.environ["NRC_CPUSET"], "physical_gpus": physical,
            "device": {"node": os.environ["NRC_NODE"], "pci": physical[0]}}


def fixed_inputs(config_path, config):
    from ..runtime import frozen_sources
    paths = frozen_sources()
    for label, directory in (("online-roles", WORKFLOW_ROOT / "roles"),
                             ("online-workflow-knowledge", WORKFLOW_ROOT / "knowledge")):
        paths.update({f"{label}/{path.relative_to(directory)}": path
                      for path in directory.rglob("*") if path.is_file()})
    for name in ("materialize_workspace.sh", "gpu_lock.sh", "workspace_sources.py"):
        paths["online-scripts/" + name] = WORKFLOW_ROOT / "scripts" / name
    paths["online-config"] = Path(config_path)
    for key in ("commandment", "codebase_context"):
        if config.get(key):
            paths[key] = Path(config[key])
    return paths


def explore(config_path, wiki, output, *, executor_factory=OnlineExecutor, revision=None, policy=None, deadline=None):
    config = yaml.safe_load(Path(config_path).read_text())
    environment = execution_environment(config, wiki)
    started = time.monotonic()
    deadline = min(deadline, started + environment["wallclock_limit_seconds"]) if deadline is not None else started + environment["wallclock_limit_seconds"]
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    budget = config.get("budget", 6)
    if type(budget) is not int or budget < 1:
        raise KnowledgeError("online budget must be a positive integer")
    gpu_ids = config["gpu_ids"]
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise KnowledgeError("online workers require explicitly assigned distinct GPUs")
    state = KernelState.from_dict(config["state"])
    wiki_snapshot = KnowledgeClient(wiki).snapshot(revision)
    policy = Policy.from_dict(policy.to_dict(), wiki_snapshot.records) if policy else Policy(wiki_snapshot.policy, wiki_snapshot.records)
    reference_files = {}
    for name, payload in wiki_snapshot.files:
        if not name.startswith("knowledge/") or not name.endswith(".md"):
            continue
        if read_markdown(name.removeprefix("knowledge/"), payload.decode()).policy_rule is not None:
            continue
        path = destination / "frozen_knowledge" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        reference_files[name] = reference(path).sha256
    (destination / "frozen_knowledge/knowledge").mkdir(parents=True, exist_ok=True)
    reference_revision = digest(reference_files)
    persist(destination / "policy.json", policy.to_dict())
    frozen_paths = fixed_inputs(config_path, config)
    knowledge_root = destination / "frozen_knowledge"
    frozen_paths.update({f"online-kernelwiki-knowledge/{path.relative_to(knowledge_root)}": path
                         for path in knowledge_root.rglob("*") if path.is_file()})
    frozen = {name: reference(path) for name, path in frozen_paths.items()}
    archived = {}
    for name, item in frozen.items():
        target = destination / "frozen_inputs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(verify(item))
        archived[name] = asdict(reference(target))
    original_workspace = Path(config["workspace"]).resolve(strict=True)
    if original_workspace == destination or original_workspace in destination.parents:
        raise KnowledgeError("online output must be outside its source workspace")
    root_workspace = str(destination / "root")
    materialize = run_process(["bash", str(WORKFLOW_ROOT / "scripts/materialize_workspace.sh"),
                               "--src", str(original_workspace), "--dst", root_workspace], destination,
                              destination / "materialize-root", timeout=120, deadline=deadline)
    if materialize["exit_code"] != 0:
        raise KnowledgeError("online root materialization failed")
    evaluator = FixedEvaluator(config["evaluator"], root_workspace, config["source_paths"],
                               deadline=deadline, device=environment["device"])
    baseline = evaluator.measure(root_workspace, destination / "baseline", gpu_ids[0])
    for item in frozen.values():
        verify(item)
    if baseline.status != "qualified":
        raise KnowledgeError("online root baseline did not qualify")
    generation = GenerationSpec(config.get("mode", "normal"), MODEL, EFFORT,
                                digest({"reference_knowledge": reference_revision,
                                        "state": config["state"], "commandment": reference(config["commandment"]).sha256,
                                        "codebase_context": reference(config["codebase_context"]).sha256
                                        if config.get("codebase_context") else None}), evaluator.revision,
                                min(2, len(gpu_ids)))
    persist(destination / "manifest.json", {"schema_version": 2, "config": config,
                                            "generation": asdict(generation), "knowledge_revision": wiki_snapshot.revision,
                                            "policy_revision": policy.revision, "reference_revision": reference_revision,
                                            "execution": environment, "evaluator_protocol": evaluator.protocol,
                                            "evaluator_runtime": evaluator.runtime,
                                            "sources": archived})
    executor = executor_factory(config, destination, evaluator, destination / "frozen_knowledge/knowledge")
    world_id = "online/" + digest(str(destination))[:24]
    root_source = snapshot(root_workspace, config["source_paths"], destination / "root-capture")
    source_ref = persist(destination / "root-capture/source.json", root_source)
    initial_context = {"generation": {key: value for key, value in asdict(generation).items() if key != "evaluator_revision"},
                       "reference_revision": reference_revision, "source": root_source["fingerprint"],
                       "branch_observations": []}
    context_ref = persist(destination / "root-context.json", initial_context)
    root = DiscoveryNode(world_id + "/root", None, None, baseline, baseline.evidence,
                         state=SourceState(root_source["fingerprint"], (source_ref,)),
                         context=ContextSnapshot(digest(initial_context), (context_ref,)))
    rollout = Rollout(state, generation, root, budget)
    workspaces = {"root": root_workspace}
    contexts = {"root": initial_context}
    positions, branches = {}, []
    while not rollout.stop:
        if time.monotonic() >= deadline:
            rollout.stop = "wallclock_limit"
            break
        for item in frozen.values():
            verify(item)
        action = policy.select(rollout.observe())
        rollout.validate(action)
        if not action.parents:
            rollout.step(action, ())
            break
        requests = [{"id": rollout.nodes[parent].id, "workspace": workspaces[parent], "context": contexts[parent],
                     "source": rollout.nodes[parent].state.fingerprint,
                     "context_revision": rollout.nodes[parent].context.revision,
                     "request_id": f"node-{rollout.attempted + index + 1}", "gpu_id": gpu_ids[index]}
                    for index, parent in enumerate(action.parents)]
        with ThreadPoolExecutor(max_workers=len(requests)) as workers:
            futures = [workers.submit(executor.execute, request, generation) for request in requests]
            outcomes = [future.result() for future in futures]
        outcomes = [executor.finish(outcome) for outcome in outcomes]
        for item in frozen.values():
            verify(item)
        nodes = []
        for parent, outcome in zip(action.parents, outcomes):
            parent_node = rollout.nodes[parent]
            measurement = outcome["measurement"]
            details = {**outcome["details"], "root_relative_gain":
                       measurement.primary_score / baseline.primary_score - 1
                       if measurement.status == "qualified" else None}
            context = {**initial_context, "parent_revision": parent_node.context.revision,
                       "branch_observations": [*contexts[parent]["branch_observations"], details]}
            context_ref = persist(destination / "generations" / outcome["id"] / "continuation.json", context)
            node_id = world_id + "/" + outcome["id"]
            node = DiscoveryNode(node_id, parent_node.id, node_id, measurement, outcome["input_evidence"],
                                 state=outcome["state"],
                                 context=ContextSnapshot(digest(context), (context_ref,), parent_node.context.revision),
                                 input_state=parent_node.state.fingerprint,
                                 input_context=parent_node.context.revision,
                                 candidate_source=outcome["candidate_source"])
            nodes.append(node)
            if parent == "root":
                positions[outcome["id"]] = len(branches)
                branches.append([node])
            else:
                positions[outcome["id"]] = positions[parent]
                branches[positions[parent]].append(node)
            workspaces[outcome["id"]] = outcome["workspace"]
            contexts[outcome["id"]] = context
        rollout.step(action, tuple(nodes))
    trace = rollout.trace(world_id, config["kernel"], "online")
    write_json(destination / "trace.json", trace)
    if branches:
        world = DiscoveryWorld(world_id, config["kernel"], "online", state, generation,
                               root, tuple(tuple(branch) for branch in branches))
        write_jsonl(destination / "worlds.jsonl", [world.to_dict()])
    result = {"status": "online_executed", "trace": trace, "knowledge_revision": wiki_snapshot.revision,
              "policy_revision": policy.revision, "reference_revision": reference_revision,
              "root_source": root.state.fingerprint, "root_context": root.context.revision,
              "evaluator_protocol": evaluator.protocol, "evaluator_runtime": evaluator.runtime,
              "execution": environment, "elapsed_seconds": time.monotonic() - started,
              "agent_calls": len(list((destination / "generations").glob("*/capture/agent-*-input.json"))),
              "knowledge_changed": False, "worlds": 1 if branches else 0,
              "artifacts": {name: reference(destination / name).sha256 for name in
                            (("manifest.json", "trace.json", "policy.json", "worlds.jsonl") if branches else
                             ("manifest.json", "trace.json", "policy.json"))}}
    write_json(destination / "result.json", result)
    return result


def calibrate(config_path, wiki, output, *, explorer=explore):
    config = yaml.safe_load(Path(config_path).read_text())
    execution = config.get("execution", {})
    if (config.get("budget") != 1 or config.get("mode", "normal") != "normal" or len(config.get("gpu_ids", [])) != 1
            or type(execution.get("wallclock_limit_seconds")) is not int
            or not 1 <= execution["wallclock_limit_seconds"] <= 1800):
        raise KnowledgeError("calibration permits only one normal direction, one GPU, and at most 1800 seconds")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + execution["wallclock_limit_seconds"]
    write_json(root / "started.json", {"config": asdict(reference(config_path)), "started_at": time.time(),
                                       "direction_units": 1, "gpus": 1, "wallclock_limit_seconds": execution["wallclock_limit_seconds"]})
    try:
        result = explorer(config_path, wiki, root / "online", policy=Policy(), deadline=deadline)
        load_online(root / "online")
        calibrated = {"status": "calibrated" if result["trace"]["charged_budget"] == 1 else "calibration_incomplete",
                      "execution_kind": result["execution"]["kind"], "direction_units": result["trace"]["charged_budget"],
                      "agent_calls": result["agent_calls"], "gpus": 1,
                      "wallclock_limit_seconds": execution["wallclock_limit_seconds"],
                      "elapsed_seconds": time.monotonic() - started, "completed_at": time.time(),
                      "online": asdict(reference(root / "online/result.json")), "published": False,
                      "next_action": "request an explicit post-calibration full-cycle resource budget",
                      "artifacts": {"started.json": reference(root / "started.json").sha256,
                                    "online/result.json": reference(root / "online/result.json").sha256}}
        write_json(root / "result.json", calibrated)
        return calibrated
    except Exception as error:
        write_json(root / "failure.json", {"status": "calibration_failed", "error": str(error),
                                           "elapsed_seconds": time.monotonic() - started,
                                           "requested_generations": len(list((root / "online/generations").glob("*/request.json"))),
                                           "agent_calls": len(list((root / "online/generations").glob("*/capture/agent-*-input.json"))),
                                           "next_action": "inspect the existing attempt; do not automatically retry"})
        raise
