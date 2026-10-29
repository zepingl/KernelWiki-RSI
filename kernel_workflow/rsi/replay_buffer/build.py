"""Construct and commit one immutable buffer from verified histories and reports."""

from collections import Counter
from dataclasses import asdict
from pathlib import Path

import yaml
from kernelwiki.knowledge import EvidenceRef, KnowledgeError

from ..evidence import (
    evidence_refs,
    read_json,
    reference,
    verify,
    verify_artifacts,
    write_json,
    write_jsonl,
)
from ..reverse_engineering.model import ReverseReport, subject_for
from ..reverse_engineering.sources.archive import read_archive
from ..reverse_engineering.sources.capture import load_online
from ..reverse_engineering.sources.kernelwiki import kernelwiki_trees, knowledge_index, resolved_commit
from ..reverse_engineering.trace import trace
from ..trace.records import to_wire
from ..trace.tree import fold, knowledge_coverage
from .cohorts import cohorts, split_worlds
from .store import open_buffer


def freeze(pool, online, output, *, investigations=()):
    base = open_buffer(pool)
    manifest, historical = base.manifest, base.worlds
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    worlds = list(historical)
    attempts = list(base.attempts)
    sources = [asdict(reference(Path(pool) / "manifest.json"))]
    traces = {report.subject.attempt.id: report.to_trace() for report in base.reverse_reports()}
    excluded = list(base.excluded)
    subjects = {}
    for directory in online:
        _, world, normalized = load_online(directory)
        sources.append(asdict(reference(Path(directory) / "result.json")))
        attempts.extend(normalized)
        worlds.append(world)
        for attempt in normalized:
            subjects[attempt.id] = subject_for(attempt, (world,))
    split_worlds(worlds, manifest["splits"])
    if len({attempt.id for attempt in attempts}) != len(attempts):
        raise KnowledgeError("duplicate external attempt in combined replay pool")
    original_attempts = {attempt.id: attempt for attempt in attempts}
    for directory in investigations:
        result = read_json(Path(directory) / "result.json")
        verify_artifacts(directory, result)
        if result.get("status") not in {"audited_with_limits", "resolved"}:
            raise KnowledgeError("only completed reverse investigations can enter a buffer")
        if "report_version" not in result:
            verify(EvidenceRef.from_dict(result["pool"]))
        report = ReverseReport.from_dict(result)
        projection = report.to_trace()
        attempt = report.subject.attempt
        if attempt != original_attempts.get(attempt.id):
            raise KnowledgeError("investigation changed the inventoried attempt")
        for item in evidence_refs(result):
            verify(item)
        traces[attempt.id] = projection
        sources.append(asdict(reference(Path(directory) / "result.json")))
    for attempt_id, subject in subjects.items():
        if attempt_id not in traces:
            traces[attempt_id] = trace(subject).to_trace()
    write_jsonl(destination / "attempts.jsonl", attempts)
    write_jsonl(destination / "worlds.jsonl", [world.to_dict() for world in worlds])
    write_jsonl(destination / "reverse.jsonl", [traces[key] for key in sorted(traces)])
    write_jsonl(destination / "excluded.jsonl", excluded)
    artifacts = ["attempts.jsonl", "worlds.jsonl", "reverse.jsonl", "excluded.jsonl"]
    tree_fields = {}
    if base.trace:
        write_jsonl(destination / "trace.jsonl", base.trace)
        artifacts.append("trace.jsonl")
        tree_fields = {"trees": len(base.trees),
                       "tree_cohorts": cohorts(base.trees, manifest["budgets"], traces="trace.jsonl")}
    result = {**manifest, **tree_fields, "schema_version": 2, "status": "frozen", "attempts": len(attempts), "worlds": len(worlds),
              "cohorts": cohorts(worlds, manifest["budgets"]), "reverse_traces": len(traces),
              "replay_ready": sum(item.disposition == "replay_ready" for item in attempts),
              "qualified_nodes": sum(item.measurement.status == "qualified" for item in attempts),
              "sources": sources, "artifacts": {name: reference(destination / name).sha256 for name in artifacts},
              "kernels": {kernel: sum(world.kernel == kernel for world in worlds)
                          for kernel in sorted({world.kernel for world in worlds})}}
    write_json(destination / "manifest.json", result)
    return result


def ingest(config_path, output, *, progress=None):
    config = yaml.safe_load(Path(config_path).read_text())
    config_ref = reference(config_path)
    campaigns = read_json(config["campaigns"])
    campaigns_ref = reference(config["campaigns"])
    tasks = config["tasks"]
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    attempts, worlds, evidence, excluded, records = [], [], [], [], []
    knowledge, wiki = None, None
    if config.get("kernelwiki"):
        # KernelWiki trees come first: a GEAK read can only resolve to an earlier knowledge node.
        wiki = {"repo": str(config["kernelwiki"]),
                "commit": resolved_commit(config["kernelwiki"], config.get("kernelwiki_rev", "HEAD"))}
        records = kernelwiki_trees(wiki["repo"], wiki["commit"])
        knowledge = knowledge_index(records)
    for campaign, campaign_data in campaigns.items():
        history = read_archive(campaign_data["path"], campaign=campaign, tasks=tasks,
                               excluded_tasks=config.get("excluded_tasks", ()), states=config.get("states", {}),
                               knowledge=knowledge)
        attempts.extend(history.attempts)
        worlds.extend(history.worlds)
        evidence.extend(asdict(item) for item in history.evidence)
        excluded.extend(history.excluded)
        records.extend(history.trace)
        if progress:
            progress({"campaign": campaign, "attempts": len(history.attempts), "worlds": len(history.worlds)})
    ids = [attempt.id for attempt in attempts]
    if len(set(ids)) != len(ids):
        raise KnowledgeError("duplicate external invocation in historical inventory")
    write_jsonl(destination / "attempts.jsonl", attempts)
    write_jsonl(destination / "worlds.jsonl", (world.to_dict() for world in worlds))
    write_json(destination / "evidence.json", evidence)
    write_jsonl(destination / "excluded.jsonl", excluded)
    write_jsonl(destination / "trace.jsonl", (to_wire(record) for record in records))
    forest = fold(records)
    trees = forest.code_trees()
    sources = []
    for name, item in (("config.yaml", config_ref), ("campaigns.json", campaigns_ref)):
        target = destination / "inputs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(verify(item))
        sources.append(asdict(reference(target)))
    budgets = config.get("budgets", [1, 2, 4, 6])
    manifest = {"schema_version": 2, "status": "ingested", "attempts": len(attempts), "worlds": len(worlds),
                "excluded_attempts": len(excluded), "excluded_tasks": dict(Counter(item["task"] for item in excluded)),
                "replay_ready": sum(attempt.disposition == "replay_ready" for attempt in attempts),
                "qualified_nodes": sum(attempt.measurement.status == "qualified" for attempt in attempts),
                "gaps": dict(Counter(gap for attempt in attempts for gap in attempt.gaps)),
                "kernels": dict(Counter(world.kernel for world in worlds)),
                "cohorts": cohorts(worlds, budgets), "splits": config["splits"], "budgets": budgets,
                "trees": len(trees), "tree_cohorts": cohorts(trees, budgets, traces="trace.jsonl"),
                "kernelwiki": wiki, "knowledge_coverage": knowledge_coverage(forest),
                "objective": config.get("objective", {"beta_cost": 0.01, "beta_parallel": 0}),
                "sources": sources,
                "artifacts": {name: reference(destination / name).sha256
                              for name in ("attempts.jsonl", "worlds.jsonl", "evidence.json", "excluded.jsonl",
                                           "trace.jsonl", "inputs/config.yaml", "inputs/campaigns.json")},
                "world_scope": "observed source AND input context; explicit continuation chains only",
                "limits": ["All original optimization invocations are inventoried, including interrupted calls.",
                           "Cross-round shared memory and merges remain audit relations or new roots.",
                           "Unknown outcomes remain charged attempts and block primary qualification.",
                           "Old seven-kernel test data is regression evidence, not a fresh holdout."]}
    write_json(destination / "manifest.json", manifest)
    return manifest
