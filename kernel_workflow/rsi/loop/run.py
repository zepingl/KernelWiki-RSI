"""Frozen cycle inputs, explicit resource envelopes and safe stage receipts."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import yaml
from kernelwiki import KnowledgeClient
from kernelwiki.knowledge import EvidenceRef, KnowledgeError, digest, strict_fields

from ..evidence import (
    artifact_hashes,
    read_json,
    reference,
    snapshot,
    verify,
    verify_artifacts,
    write_json,
)
from ..harness import Policy
from ..replay_buffer.store import open_buffer
from ..reverse_engineering.sources.capture import load_online
from ..runtime import RuntimeAgent, frozen_sources, remaining_seconds
from .online import fixed_inputs


class IncompleteStage(KnowledgeError):
    """An operation may have run; never repeat its external effects automatically."""


def _cost(result):
    return {"direction_units": result.get("trace", {}).get("charged_budget", 0),
            "agent_calls": result.get("agent_calls", result.get("model_calls", 0))}


def execute_stage(root, name, inputs, operation, *, deadline, limits, reserve=None, result_name="result.json"):
    started = root / "stages" / (name + ".started.json")
    done = root / "stages" / (name + ".done.json")
    directory = root / name
    if done.exists():
        receipt = read_json(done)
        if receipt["inputs"] != inputs or read_json(started)["inputs"] != inputs:
            raise KnowledgeError("resume inputs differ from the completed stage")
        result = json.loads(verify(EvidenceRef.from_dict(receipt["result"])))
        verify_artifacts(directory, result)
        if receipt["cost"] != _cost(result):
            raise KnowledgeError("completed stage resource accounting differs from its observed result")
        return result
    if started.exists() or directory.exists():
        raise IncompleteStage(f"uncertain stage {name}: inspect {started}; no automatic Agent/GPU retry")
    remaining_seconds(deadline, limits["wallclock_seconds"])
    usage = spent(root)
    reserve = reserve or {"direction_units": 0, "agent_calls": 0}
    if any(usage[key] + reserve.get(key, 0) > limits[key] for key in ("direction_units", "agent_calls")):
        raise KnowledgeError(f"cycle resource budget cannot admit stage {name}")
    write_json(started, {"stage": name, "inputs": inputs, "started_at": time.time(), "reserved": reserve})
    beginning = time.monotonic()
    try:
        operation(directory)
        result = read_json(directory / result_name)
        verify_artifacts(directory, result)
        cost = _cost(result)
        if any(type(cost[key]) is not int or cost[key] < 0 or cost[key] > reserve.get(key, 0) for key in cost):
            raise KnowledgeError("stage exceeded its reserved resource envelope")
        write_json(done, {"stage": name, "inputs": inputs, "result": asdict(reference(directory / result_name)),
                          "cost": cost, "elapsed_seconds": time.monotonic() - beginning, "completed_at": time.time()})
        return result
    except Exception as error:
        write_json(root / "stages" / (name + ".failed.json"),
                   {"stage": name, "error": str(error), "elapsed_seconds": time.monotonic() - beginning,
                    "next_action": "inspect the existing stage; an uncertain external operation is never retried"})
        raise


def spent(root):
    result = {"direction_units": 0, "agent_calls": 0}
    for path in (root / "stages").glob("*.done.json"):
        receipt = read_json(path)
        reserved = read_json(path.with_name(path.name.replace(".done.json", ".started.json")))["reserved"]
        for key, value in receipt["cost"].items():
            if key not in result or type(value) is not int or not 0 <= value <= reserved.get(key, 0):
                raise KnowledgeError("invalid completed stage resource accounting")
            result[key] += value
    return result


def online_budget(config):
    cost = 2 if config.get("mode", "normal") == "deep_explore" else 1
    return {"direction_units": config["budget"], "agent_calls": 2 * (config["budget"] // cost)}


def load_settings(config_path, manifest):
    path = Path(config_path).resolve()
    config = yaml.safe_load(path.read_text())
    strict_fields(config, {"schema_version", "pool", "rounds", "discovery", "holdout", "max_candidates",
                           "investigation", "limits", "calibration", "approval", "agent_timeout_seconds"},
                  {"schema_version", "pool", "discovery", "holdout", "limits"})
    if config["schema_version"] != 1:
        raise KnowledgeError("unsupported outer-cycle configuration")
    rounds, candidates = config.get("rounds", 2), config.get("max_candidates", 4)
    if type(rounds) is not int or rounds < 2 or type(candidates) is not int or not 1 <= candidates <= 4:
        raise KnowledgeError("the outer cycle needs at least two rounds and one to four inner opportunities")
    investigation = config.get("investigation", {"max_steps": 2, "model_calls": 0})
    strict_fields(investigation, {"max_steps", "model_calls"}, {"max_steps", "model_calls"})
    if (type(investigation["max_steps"]) is not int or not 1 <= investigation["max_steps"] <= 16
            or type(investigation["model_calls"]) is not int
            or not 0 <= investigation["model_calls"] <= investigation["max_steps"]):
        raise KnowledgeError("reverse investigation requires bounded steps and model calls")
    entries = {}
    for group in ("discovery", "holdout"):
        if not isinstance(config[group], list) or not config[group] or any(not isinstance(name, str) for name in config[group]):
            raise KnowledgeError("cycle discovery and holdout must list their explicit configuration files")
        entries[group] = []
        for name in config[group]:
            source = (path.parent / name).resolve()
            value = yaml.safe_load(source.read_text())
            if type(value.get("budget")) is not int or value["budget"] < max(manifest["budgets"]):
                raise KnowledgeError("cycle discovery must cover every frozen replay budget")
            if value.get("mode", "normal") not in {"normal", "deep_explore"}:
                raise KnowledgeError("invalid cycle generation mode")
            if len(value.get("gpu_ids", [])) not in (1, 2):
                raise KnowledgeError("cycle needs one or two explicitly placed workers")
            entries[group].append({"path": str(source), "config": value})
        kernels = [entry["config"]["kernel"] for entry in entries[group]]
        expected = manifest["splits"]["test"] if group == "holdout" else manifest["splits"]["train"] + manifest["splits"]["validation"]
        if len(set(kernels)) != len(kernels) or set(kernels) != set(expected):
            raise KnowledgeError(f"{group} must cover exactly its frozen kernel split, without duplicates")
    kinds = {entry["config"].get("execution", {}).get("kind") for group in entries.values() for entry in group}
    if len(kinds) != 1 or not kinds <= {"coordinated", "test_fixture"}:
        raise KnowledgeError("cycle cannot mix coordinated work with synthetic fixtures")
    if type(config.get("agent_timeout_seconds", 300)) is not int or config.get("agent_timeout_seconds", 300) <= 0:
        raise KnowledgeError("Agent timeout must be a positive number of seconds")
    for key in ("calibration", "approval"):
        if config.get(key):
            config[key] = str((path.parent / config[key]).resolve())
    return {**config, "pool": str((path.parent / config["pool"]).resolve()), "rounds": rounds,
            "max_candidates": candidates, "investigation": investigation, "entries": entries, "kind": kinds.pop()}


def _projection(settings):
    development = [online_budget(entry["config"]) for entry in settings["entries"]["discovery"]]
    holdout = [online_budget(entry["config"]) for entry in settings["entries"]["holdout"]]
    rounds = settings["rounds"]
    generations = rounds * sum(item["agent_calls"] // 2 for item in development)
    inner = rounds * (1 + 2 * settings["max_candidates"])
    analyst = generations * settings["investigation"]["model_calls"]
    return {"direction_units": rounds * sum(item["direction_units"] for item in development)
                               + 2 * sum(item["direction_units"] for item in holdout),
            "agent_calls": 2 * generations + 2 * sum(item["agent_calls"] for item in holdout) + inner + analyst,
            "gpus": max(len(entry["config"]["gpu_ids"]) for entries in settings["entries"].values() for entry in entries),
            "inner_agent_calls": inner, "reverse_agent_calls": analyst,
            "note": "upper bounds include two fresh holdout arms; unused opportunities are not spent"}


def authorize(settings, wiki_revision, pool_ref, output):
    projection = _projection(settings)
    code = {name: reference(path).sha256 for name, path in frozen_sources().items()}
    plan_revision = digest({"settings": {key: value for key, value in settings.items() if key != "approval"},
                            "wiki_revision": wiki_revision, "pool": asdict(pool_ref), "implementation": code})
    limits = settings["limits"]
    valid_limits = set(limits) == {"direction_units", "agent_calls", "gpus", "wallclock_seconds"} and all(
        type(value) is int and value > 0 for value in limits.values())
    if settings["kind"] == "test_fixture":
        if not valid_limits or any(limits[key] < projection[key] for key in ("direction_units", "agent_calls", "gpus")):
            raise KnowledgeError("fixture resource envelope is smaller than the planned cycle")
        return None
    calibration_ref = None
    calibration = None
    if settings.get("calibration"):
        calibration_ref = reference(settings["calibration"])
        calibration = read_json(calibration_ref.path)
        verify_artifacts(Path(calibration_ref.path).parent, calibration)
        observed, world, _ = load_online(Path(calibration_ref.path).parent / "online")
        if (calibration.get("status") != "calibrated" or calibration.get("execution_kind") != "coordinated"
                or calibration.get("direction_units") != 1 or calibration.get("gpus") != 1
                or calibration.get("wallclock_limit_seconds", 1801) > 1800
                or observed["execution"]["kind"] != "coordinated" or observed["trace"]["attempted_nodes"] != 1
                or world.generation.mode != "normal" or world.generation.worker_limit != 1):
            raise KnowledgeError("full-cycle approval requires a real one-direction, one-GPU calibration")
    if settings.get("approval"):
        approval_ref = reference(settings["approval"])
        approval = read_json(approval_ref.path)
        request = json.loads(verify(EvidenceRef.from_dict(approval["request"])))
        if (approval.get("decision") != "approve" or approval.get("scope") != "kernelwiki-rsi-cycle"
                or not isinstance(approval.get("approved_by"), str) or not approval["approved_by"]
                or not valid_limits or request.get("plan_revision") != plan_revision
                or approval.get("limits") != limits or calibration_ref is None
                or request.get("calibration") != asdict(calibration_ref)
                or any(limits[key] < projection[key] for key in ("direction_units", "agent_calls", "gpus"))):
            raise KnowledgeError("approval does not cover this post-calibration frozen cycle and resource envelope")
        return approval_ref
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    request = {"status": "budget_approval_required" if calibration_ref else "calibration_required",
               "plan_revision": plan_revision, "projection": projection, "requested_limits": limits,
               "calibration": asdict(calibration_ref) if calibration_ref else None,
               "calibration_seconds": calibration.get("elapsed_seconds") if calibration else None,
               "published": False, "agent_calls": 0, "direction_units": 0,
               "next_action": "complete the one authorized calibration, then obtain explicit approval of this request; no full-run budget is inferred"}
    write_json(destination / "budget-request.json", request)
    write_json(destination / "result.json", {**request, "artifacts": artifact_hashes(destination)})
    return request


def freeze_inputs(root, config_path, settings, wiki, approval):
    client = KnowledgeClient(wiki)
    knowledge = client.snapshot()
    paths = frozen_sources()
    paths["cycle-config.yaml"] = Path(config_path).resolve()
    pool_manifest = open_buffer(settings["pool"]).manifest
    for name in ("manifest.json", *pool_manifest["artifacts"]):
        paths["initial-pool/" + name] = Path(settings["pool"]) / name
    for group, entries in settings["entries"].items():
        for index, entry in enumerate(entries, 1):
            prefix = f"{group}-{index}/"
            config = entry["config"]
            for label, input_path in fixed_inputs(entry["path"], config).items():
                if label not in {"online-config", "commandment", "codebase_context"}:
                    paths[label] = input_path
            paths[prefix + "config.yaml"] = Path(entry["path"])
            source = snapshot(config["workspace"], config["source_paths"])
            for name in (*source["files"], *config["evaluator"]["contract_files"]):
                paths[prefix + "workspace/" + name] = Path(config["workspace"]) / name
            for key in ("commandment", "codebase_context"):
                if config.get(key):
                    paths[prefix + key] = Path(config[key])
    sources = {name: asdict(reference(path)) for name, path in paths.items()}
    root.mkdir(parents=True, exist_ok=False)
    for name, item in sources.items():
        target = root / "frozen_inputs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(verify(EvidenceRef.from_dict(item)))
    for name, payload in knowledge.files:
        target = root / "frozen_knowledge" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(payload)
    write_json(root / "initial-policy.json", Policy(knowledge.policy, knowledge.records).to_dict())
    manifest = {"schema_version": 1, "settings": settings, "wiki": str(Path(wiki).resolve()),
                "knowledge_revision": knowledge.revision, "started_at": time.time(), "sources": sources,
                "approval": asdict(approval) if approval else None, "artifacts": artifact_hashes(root)}
    write_json(root / "manifest.json", manifest)
    return manifest


def guard_inputs(root, manifest, wiki):
    verify_artifacts(root, manifest)
    for item in manifest["sources"].values():
        verify(EvidenceRef.from_dict(item))
    if KnowledgeClient(wiki).snapshot().revision != manifest["knowledge_revision"]:
        raise KnowledgeError("active KernelWiki changed during the experimental cycle")
    if manifest["approval"]:
        verify(EvidenceRef.from_dict(manifest["approval"]))


def bind_agent(agent, deadline, remaining_calls, timeout, *, roles):
    delegate = agent
    calls = 0

    def request(role, payload, directory):
        nonlocal calls, delegate
        remaining_seconds(deadline, timeout)
        if calls >= remaining_calls:
            raise KnowledgeError("cycle model-call budget exhausted")
        if delegate is None:
            delegate = RuntimeAgent(timeout=timeout, deadline=deadline, roles=roles)
        calls += 1
        return delegate(role, payload, directory)

    return request


def read_completed(root):
    result = read_json(root / "result.json")
    verify_artifacts(root, result)
    if (root / "manifest.json").exists():
        verify_artifacts(root, read_json(root / "manifest.json"))
    for path in (root / "stages").glob("*.done.json"):
        receipt = read_json(path)
        value = json.loads(verify(EvidenceRef.from_dict(receipt["result"])))
        verify_artifacts(Path(receipt["result"]["path"]).parent, value)
    return result
