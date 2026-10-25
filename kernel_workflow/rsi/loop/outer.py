"""Online discoveries and reverse evidence feed the next bounded policy round."""

import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import yaml
from kernelwiki import KnowledgeClient
from kernelwiki.knowledge import KnowledgeError

from ..evidence import artifact_hashes, read_json, reference, write_json
from ..harness import Policy
from ..replay_buffer.build import freeze
from ..replay_buffer.store import open_buffer
from ..reverse_engineering.analyst import ROLE as ANALYST_ROLE
from ..reverse_engineering.investigate import investigate
from ..reverse_engineering.model import subject_for
from ..reverse_engineering.sources.capture import load_online
from .inner import ROLES, evolve
from .online import explore
from .qualification import qualify
from .run import (
    authorize,
    bind_agent,
    execute_stage,
    freeze_inputs,
    guard_inputs,
    load_settings,
    online_budget,
    read_completed,
    spent,
)


def cycle(config_path, wiki, output, *, resume=False, agent=None, explorer=explore, progress=None):
    root = Path(output).resolve()
    if resume and (root / "result.json").exists():
        return read_completed(root)
    config = yaml.safe_load(Path(config_path).read_text())
    pool = (Path(config_path).resolve().parent / config["pool"]).resolve()
    pool_manifest = open_buffer(pool).manifest
    settings = load_settings(config_path, pool_manifest)
    if resume:
        manifest = read_json(root / "manifest.json")
        if settings != manifest["settings"] or str(Path(wiki).resolve()) != manifest["wiki"]:
            raise KnowledgeError("resumed cycle configuration differs from its frozen inputs")
    else:
        approval = authorize(settings, KnowledgeClient(wiki).snapshot().revision, reference(pool / "manifest.json"), root)
        if isinstance(approval, dict):
            return read_json(root / "result.json")
        manifest = freeze_inputs(root, config_path, settings, wiki, approval)
    if settings["kind"] == "test_fixture" and agent is None:
        raise KnowledgeError("fixture cycles require an injected test Agent; no live model call is implied")
    deadline = time.monotonic() + manifest["started_at"] + settings["limits"]["wallclock_seconds"] - time.time()
    records = KnowledgeClient(wiki).snapshot().records
    initial = Policy.from_dict(read_json(root / "initial-policy.json"), records)
    working = initial
    rounds = []
    current_pool = pool

    def stage(name, inputs, operation, **kwargs):
        guard_inputs(root, manifest, wiki)
        result = execute_stage(root, name, inputs, operation, deadline=deadline, limits=settings["limits"], **kwargs)
        guard_inputs(root, manifest, wiki)
        if progress:
            progress({"stage": name, "status": result["status"]})
        return result

    def experimental_agent():
        return bind_agent(agent, deadline, settings["limits"]["agent_calls"] - spent(root)["agent_calls"],
                      settings.get("agent_timeout_seconds", 300), roles={**ROLES, "reverse_analyst": ANALYST_ROLE})

    def discover(name, entry, policy, extra=None):
        config_ref = asdict(reference(entry["path"]))
        stage(name, {"config": config_ref, "policy_revision": policy.revision, **(extra or {})},
                       lambda directory: explorer(entry["path"], wiki, directory, policy=policy, deadline=deadline),
                       reserve=online_budget(entry["config"]))
        return load_online(root / name)

    for round_number in range(1, settings["rounds"] + 1):
        prefix = f"round-{round_number}"
        online, histories = [], []
        before_revision = working.revision
        for index, entry in enumerate(settings["entries"]["discovery"], 1):
            name = prefix + f"-discovery-{index}"
            _, world, attempts = discover(name, entry, working)
            online.append(root / name)
            histories.append((root / name, world, attempts))
        investigations = []
        for directory, world, attempts in histories:
            for attempt in attempts:
                subject = subject_for(attempt, (world,), (reference(directory / "result.json"),))
                name = prefix + f"-reverse-{len(investigations) + 1}"
                stage(name, {"subject": subject.to_dict()},
                      partial(investigate, subject, **settings["investigation"], agent=experimental_agent()),
                      reserve={"direction_units": 0, "agent_calls": settings["investigation"]["model_calls"]})
                investigations.append(root / name)
        frozen = prefix + "-pool"
        stage(frozen, {"pool": asdict(reference(current_pool / "manifest.json")),
                       "online": [asdict(reference(path / "result.json")) for path in online],
                       "investigations": [asdict(reference(path / "result.json")) for path in investigations]},
              partial(freeze, current_pool, online, investigations=investigations), result_name="manifest.json")
        current_pool = root / frozen
        name = prefix + "-evolve"
        evolved = stage(name, {"pool": asdict(reference(current_pool / "manifest.json")), "policy_revision": working.revision},
                        partial(evolve, current_pool, wiki, policy=working, agent=experimental_agent(),
                                max_candidates=settings["max_candidates"]),
                        reserve={"direction_units": 0, "agent_calls": 1 + 2 * settings["max_candidates"]})
        working = Policy.from_dict(read_json(root / name / "selected-policy.json"), records)
        rounds.append({"round": round_number, "input_policy": before_revision, "output_policy": working.revision,
                       "pool": str(current_pool), "status": evolved["status"]})
    status = "insufficient_primary_support" if rounds[-1]["status"] == "insufficient_primary_support" else "no_change"
    qualification = None
    if working.revision != initial.revision:
        def freeze_final(directory):
            write_json(directory / "selected-policy.json", working.to_dict())
            write_json(directory / "result.json", {"status": "final_candidate_frozen", "policy_revision": working.revision,
                                                   "artifacts": artifact_hashes(directory)})
        stage("final", {"policy_revision": working.revision, "rounds": rounds}, freeze_final)
        final_ref = asdict(reference(root / "final/selected-policy.json"))
        for index, entry in enumerate(settings["entries"]["holdout"], 1):
            for arm, policy in (("incumbent", initial), ("candidate", working)):
                discover(f"holdout-{index}-{arm}", entry, policy, {"frozen_candidate": final_ref})
        def qualify_final(directory):
            result = qualify(root, initial, working, pool_manifest, settings)
            write_json(directory / "evidence.json", result)
            write_json(directory / "result.json", {"status": "qualified" if result["passed"] else "rejected",
                                                   "artifacts": artifact_hashes(directory)})
        qualification = stage("qualification", {"frozen_candidate": final_ref}, qualify_final)
        if qualification["status"] == "qualified":
            status = "replay_validated" if settings["kind"] == "coordinated" else "fixture_validated"
        else:
            status = "final_qualification_failed"
    result = {"status": status, "rounds": rounds, "policy_revision": working.revision, "published": False,
              "knowledge_revision": manifest["knowledge_revision"], "execution_kind": settings["kind"],
              "resources": spent(root), "elapsed_seconds": time.time() - manifest["started_at"],
              "test_evaluated": qualification is not None,
              "reason": "active knowledge is unchanged; release is a separate CAS step after real final gates",
              "artifacts": {str(path.relative_to(root)): reference(path).sha256 for path in
                            [root / "manifest.json", *(root / "stages").glob("*.json")]}}
    write_json(root / "result.json", result)
    return result
