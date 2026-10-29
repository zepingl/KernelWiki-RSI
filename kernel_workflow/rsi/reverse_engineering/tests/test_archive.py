import hashlib
import json
from pathlib import Path

import pytest
import yaml
from kernel_workflow.rsi.replay_buffer.build import ingest
from kernel_workflow.rsi.replay_buffer.store import open_buffer
from kernel_workflow.rsi.reverse_engineering.sources.archive import (
    AuditReader,
    prompt_inputs,
)
from kernelwiki.knowledge import KnowledgeError


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def source(run, content):
    blob_sha = hashlib.sha256(content.encode()).hexdigest()
    blob = run / "objects" / blob_sha[:2] / blob_sha
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_text(content)
    files = {"kernel.py": {"sha256": blob_sha, "size": len(content), "symlink": None}}
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    save(run / "index/candidate_manifests" / f"{fingerprint}.json",
         {"fingerprint": fingerprint, "files": files})
    return fingerprint


def measurement(run, name, fingerprint, workspace, call_id, score, time, base="benchmarks"):
    directory = run / base / name
    report = {"measurement_valid": True, "correctness_pass": True, "performance_scope_complete": True,
              "expected_case_count": 1, "runtime_fingerprint": "runtime-v1",
              "per_case": [{"name": "case", "calls": 2, "baseline_ms": 1,
                            "optimized_ms": 1 / score, "measurement_valid": True, "timer": "graph"}]}
    processes = []
    for index in (1, 2):
        process = {"pid": index, "exit_code": 0, "source_unchanged": True,
                   "contract_unchanged": True, "report_result": report}
        processes.append(process)
        save(directory / f"process_{index}/receipt.json", process)
        save(directory / f"process_{index}/invocation.json", {"candidate": fingerprint})
    summary = {"candidate_fingerprint": fingerprint, "workspace": workspace, "time": time,
               "measurement_valid": True, "full_task_coverage": True, "two_independent_processes": True,
               "default_source_promotion": {"eligible": True}, "processes": processes}
    save(directory / "summary.json", summary)
    save(directory / "before.json", {"candidate": {"fingerprint": fingerprint}, "entry": "unittest.py",
                                      "contracts": {"unittest.py": "frozen"}, "call_id": call_id,
                                      "candidate_runtime": {"overrides": {}}})
    return {**summary, "call_id": call_id, "summary_path": str((directory / "summary.json").relative_to(run))}


@pytest.fixture
def audit(tmp_path):
    return build_audit(tmp_path)


def build_audit(tmp_path, calls_dir="calls"):
    run = tmp_path / "run"
    parent = source(run, "parent")
    candidate = source(run, "candidate")
    canonical = str(run / "experiment/workspace")
    benchmarks = [measurement(run, "baseline", parent, canonical, "baseline", 2.0, "2026-01-01T00:00:00")]
    calls, changes, engineers = [], [], []
    for index in range(2):
        call_id = f"call-{1-index}"
        directory = run / calls_dir / call_id
        output = str(run / f"experiment/round_1/engineer_{index}")
        workspace = output + "/workspace"
        inputs = {"KERNEL_PATH": workspace, "CANONICAL": canonical, "OUTPUT_DIR": output,
                  "DIRECTION": {"id": f"r1_d{index}"}, "SPECIALTY": "compute", "KK_REFS": [str(index)]}
        prompt = "## Inputs\n" + "\n".join(f"- {key}: {json.dumps(value)}" for key, value in inputs.items())
        directory.mkdir(parents=True)
        (directory / "delivered_prompt.txt").write_text(prompt)
        call = {"call_id": call_id, "time": "2026-01-01T01:00:00", "context": inputs,
                "argv": ["claude", "--model", "claude-opus-5.5", "--effort", "high"],
                "delivered_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
        save(directory / "invocation.json", call)
        calls.append({**call, "task": "task-train", "context": {**inputs, "phase": "Optimize"},
                      "path": f"{calls_dir}/{call_id}"})
        save(directory / "pre.json", {"source_states": [{"workspace": canonical, "fingerprint": parent}],
                                       "artifacts": []})
        save(directory / "post.json", {"source_states": [{"workspace": workspace, "fingerprint": parent}],
                                        "artifacts": []})
        save(directory / "final.json", {"source_states": [{"workspace": workspace, "fingerprint": candidate}],
                                         "artifacts": []})
        base = {"call_id": call_id, "workspace": workspace, "kind": "workspace_created", "event_line": 1,
                "before_fingerprint": None, "after_fingerprint": parent,
                "pre_observation": f"{calls_dir}/{call_id}/pre.json", "post_observation": f"{calls_dir}/{call_id}/post.json",
                "interval_start": "2026-01-01T01:01:00"}
        changes.extend([base, {**base, "kind": "source_modified", "event_line": 3,
                                "before_fingerprint": parent, "after_fingerprint": candidate,
                                "pre_observation": f"{calls_dir}/{call_id}/post.json",
                                "post_observation": f"{calls_dir}/{call_id}/final.json",
                                "interval_start": "2026-01-01T01:02:00"}])
        if index == 0:
            calls.append({"call_id": "verify", "context": {"phase": "verify", "PATCH": output + "/best_patch.diff"}})
            benchmarks.append(measurement(run, "verify", candidate, output + "/verify", "verify", 2.2,
                                          "2026-01-01T01:05:00"))
        engineers.append({"call_id": call_id, "invocation": str(directory / "invocation.json")})
    lines(run / "index/agent_calls.jsonl", calls)
    lines(run / "index/source_changes.jsonl", changes)
    lines(run / "index/benchmarks.jsonl", benchmarks)
    config = tmp_path / "config.yaml"
    save(tmp_path / "campaigns.json", {"campaign": {"path": str(run)}})
    save(tmp_path / "chart.json", {"kernels": {"train": {"stages": [{"node": "campaign",
         "points": [{"engineers": engineers}]}]}}})
    config.write_text(yaml.safe_dump({"campaigns": str(tmp_path / "campaigns.json"),
                                     "tasks": {"task-train": "train"},
                                     "splits": {"train": ["train"], "validation": [], "test": []}}))
    return run, config, benchmarks


def test_raw_ingest_preserves_dispatch_order_and_unknown_cost(audit, tmp_path):
    run, config, rows = audit
    # call-1 retains the same source that call-0's verifier measured, so it shares that result.
    manifest = ingest(config, tmp_path / "pool")
    assert manifest["attempts"] == 2 and manifest["replay_ready"] == 2
    assert manifest["worlds"] == 1 and manifest["qualified_nodes"] == 2
    worlds = open_buffer(tmp_path / "pool").worlds
    assert [branch[0].attempt_id for branch in worlds[0].branches] == ["call-1", "call-0"]
    lines(run / "index/benchmarks.jsonl", rows[:1])
    ingest(config, tmp_path / "unmeasured")
    worlds = open_buffer(tmp_path / "unmeasured").worlds
    assert {branch[0].measurement.status for branch in worlds[0].branches} == {"unknown"}


def test_raw_measurement_rejects_wrong_source_and_contract_change(audit):
    run, _, rows = audit
    row = rows[0]
    assert AuditReader(run).measurement(row).status == "qualified"
    path = run / Path(row["summary_path"]).parent / "process_1/invocation.json"
    save(path, {"candidate": "wrong"})
    assert AuditReader(run).measurement(row).status == "unknown"


def failed_measurement(run, name, fingerprint, call_id):
    directory = run / "benchmarks" / name
    process = {"pid": 1, "exit_code": 1, "source_unchanged": True, "contract_unchanged": True,
               "report_result": {"correctness_pass": False, "measurement_valid": False}}
    summary = {"candidate_fingerprint": fingerprint, "measurement_valid": False, "full_task_coverage": False,
               "two_independent_processes": False, "processes": [process]}
    save(directory / "summary.json", summary)
    save(directory / "before.json", {"candidate": {"fingerprint": fingerprint}, "contracts": {"unittest.py": "frozen"}})
    return {**summary, "call_id": call_id, "summary_path": str((directory / "summary.json").relative_to(run))}


def test_measurement_belongs_to_the_source_and_uses_the_lower_median(audit):
    run, _, rows = audit
    parent, candidate = rows[0]["candidate_fingerprint"], rows[1]["candidate_fingerprint"]
    contract = AuditReader(run).measurement(rows[0]).contract
    extra = [measurement(run, f"engineer-{score}", candidate, "engineer", "call-1", score, "2026-01-01T01:03:00")
             for score in (2.6, 2.0, 3.0)]
    lines(run / "index/benchmarks.jsonl", rows + extra)
    reader = AuditReader(run)
    # Runs at 2.0, 2.2 (verifier), 2.6 and 3.0: the lower median is the verifier's run, not the best one.
    assert reader.candidate_measurement(candidate, contract) == reader.measurement(rows[1])
    # An unchanged source inherits the parent's own measurement; nothing is invented.
    assert reader.candidate_measurement(parent, contract) == reader.measurement(rows[0])
    assert reader.candidate_measurement(candidate, "other-contract").status == "unknown"
    # A run whose receipts are missing is skipped instead of voiding the whole attempt.
    (run / extra[0]["summary_path"]).unlink()
    assert AuditReader(run).candidate_measurement(candidate, contract) == reader.measurement(rows[1])


def test_correctness_failure_is_a_known_outcome_unless_contradicted(audit):
    run, _, rows = audit
    rejected = source(run, "incorrect")
    contract = AuditReader(run).measurement(rows[0]).contract
    lines(run / "index/benchmarks.jsonl", rows + [failed_measurement(run, "failed", rejected, "call-1")])
    result = AuditReader(run).candidate_measurement(rejected, contract)
    assert result.status == "failed" and result.source == rejected and result.primary_score is None
    candidate = rows[1]["candidate_fingerprint"]
    lines(run / "index/benchmarks.jsonl", rows + [failed_measurement(run, "flaky", candidate, "call-1")])
    result = AuditReader(run).candidate_measurement(candidate, contract)
    assert result.status == "unknown" and result.reason == "conflicting correctness evidence"


def test_input_tampering_and_missing_parent_stay_trace_only(audit, tmp_path):
    run, config, _ = audit
    (run / "calls/call-1/delivered_prompt.txt").write_text("changed")
    manifest = ingest(config, tmp_path / "pool")
    assert manifest["attempts"] == 2 and manifest["replay_ready"] == 0
    assert manifest["worlds"] == 0
    assert manifest["gaps"] == {"delivered prompt hash mismatch": 1,
                                "round contains unsupported or reused dispatch input": 1}
    (tmp_path / "pool/worlds.jsonl").write_text("[]\n")
    with pytest.raises(KnowledgeError, match="artifact changed"):
        open_buffer(tmp_path / "pool")


def test_hidden_source_context_is_never_reconstructed_from_claims():
    with pytest.raises(KnowledgeError):
        prompt_inputs("The agent claims it started from the winner.")


def test_mixed_campaign_requires_explicit_task_exclusions(audit, tmp_path):
    run, config, _ = audit
    path = run / "index/agent_calls.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows.append({**rows[0], "call_id": "outside", "task": "outside-seven-kernel-scope"})
    lines(path, rows)
    with pytest.raises(KnowledgeError, match="lacks a kernel assignment"):
        ingest(config, tmp_path / "unscoped")
    value = yaml.safe_load(config.read_text())
    value["excluded_tasks"] = ["outside-seven-kernel-scope"]
    config.write_text(yaml.safe_dump(value))
    manifest = ingest(config, tmp_path / "scoped")
    assert manifest["attempts"] == 2 and manifest["excluded_attempts"] == 1
    excluded = json.loads((tmp_path / "scoped/excluded.jsonl").read_text())
    assert excluded["id"] == "outside" and excluded["inventory"]["sha256"]


def test_missing_context_does_not_erase_observed_source_identity(audit, tmp_path):
    run, config, _ = audit
    directory = run / "calls/call-1"
    prompt = (directory / "delivered_prompt.txt").read_text() + '\n- COMMANDMENT: "/missing/context.md"'
    (directory / "delivered_prompt.txt").write_text(prompt)
    invocation = json.loads((directory / "invocation.json").read_text())
    invocation["delivered_prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
    save(directory / "invocation.json", invocation)
    ingest(config, tmp_path / "pool")
    attempts = [json.loads(line) for line in (tmp_path / "pool/attempts.jsonl").read_text().splitlines()]
    attempt = next(item for item in attempts if item["id"] == "call-1")
    assert attempt["disposition"] == "trace_only" and attempt["state"]["fingerprint"]
    assert attempt["parent_source"] and attempt["candidate_source"]
    assert attempt["context"] is None


def test_original_archived_invocations_supplement_an_incomplete_index(audit, tmp_path):
    run, config, _ = audit
    invocation = run / "recording/records/tasks/task-train/launches/original/agents/interrupted/invocation.json"
    save(invocation, {"call_id": "interrupted", "task": "task-train", "context": {"phase": "Optimize", "role": "engineer"}})
    manifest = ingest(config, tmp_path / "pool")
    assert manifest["attempts"] == 3
    attempts = [json.loads(line) for line in (tmp_path / "pool/attempts.jsonl").read_text().splitlines()]
    recovered = next(item for item in attempts if item["id"] == "interrupted")
    assert recovered["invocation"]["path"] == str(invocation)
    assert recovered["disposition"] == "trace_only" and recovered["gaps"]


def test_trace_cli_can_read_a_raw_run_without_building_a_pool(audit, tmp_path, capsys):
    from kernel_workflow.rsi.__main__ import main
    run, _, _ = audit
    output = tmp_path / "standalone.json"
    assert main(["trace", "--run", str(run), "--kernel", "train", "--attempt", "call-1", "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["report_version"] == 1 and "pool" not in result
    assert result["subject"]["attempt"]["id"] == "call-1"
    assert result["trace"]["effect"]["parent_relative_gain"] > 0
    assert not list(tmp_path.rglob("worlds.jsonl"))
    assert "effect" in capsys.readouterr().out
