import json
from dataclasses import asdict
from pathlib import Path

import pytest
from kernel_workflow.rsi.discovery import Attempt, Measurement
from kernel_workflow.rsi.evidence import reference, write_json
from kernel_workflow.rsi.reverse_engineering.investigate import investigate
from kernel_workflow.rsi.reverse_engineering.model import (
    Gap,
    ReverseReport,
    TraceSubject,
)
from kernel_workflow.rsi.reverse_engineering.sources.archive import (
    AuditReader,
    _body_delivery,
)
from kernel_workflow.rsi.reverse_engineering.trace import effect_summary, trace
from kernelwiki.knowledge import KnowledgeError

RAW = Path("/shared_nfs/zepingl/geak_audit/runs")


def raw_subject(run_name, call_id, kernel):
    run = RAW / run_name
    if not (run / "index/agent_calls.jsonl").is_file():
        pytest.skip("local immutable raw GLM7 archive is not mounted")
    return AuditReader(run).subject(call_id, kernel, "raw-golden")


def test_raw_one_trip_delivery_method_source_and_measurement(tmp_path):
    call_id = "20260924T085229_6f2cd0d9f10247188e0ad8dbff911a5d"
    subject = raw_subject("20260923T165210Z_node039_opus55_d6dc79bd", call_id, "dsa-sparse-mla")
    report = trace(subject).to_trace()
    assert sum(item["kind"] == "body_delivered" for item in report["relations"]) == 2
    assert any(item["kind"] == "direction_forwarded" for item in report["relations"])
    assert report["effect"]["method_status"] == "reviewed_direct_gluon_card_adoption_and_scope_qualified_source"
    assert report["effect"]["parent_relative_gain"] > 0
    assert report["threshold"]["value"] == pytest.approx(0.007)
    assert report["effect"]["threshold_met"] is True and report["effect"]["retained"] is True
    assert report["effect"]["independent_knowledge_causal_gain"] is None
    assert all(item["certainty"] == "review" for item in report["relations"] if item["kind"] == "method_adopted")


def test_raw_moe_method_trial_and_exact_source_rollback(tmp_path):
    call_id = "20260922T170237_c4d636f2ae5d4145b4cbf263af278651"
    subject = raw_subject("20260922T144545Z_node056_95047c8c", call_id, "mxfp4-moe")
    report = trace(subject).to_trace()
    assert report["effect"]["method_status"] == "indirect_method_implemented_measured_then_reverted"
    assert report["effect"]["rolled_back"] is True and report["effect"]["retained"] is False
    assert report["effect"]["installed_scope"] == "attempt_restored_parent"
    assert report["attempt"]["state"]["fingerprint"] == report["attempt"]["parent_source"]
    assert report["attempt"]["candidate_source"] != report["attempt"]["state"]["fingerprint"]
    assert len(report["internal_measurements"]) >= 3


def test_raw_absorb_correct_call_recovers_save_but_not_missing_director(tmp_path):
    call_id = "20260930T034949_9d15c81467ab45fab8f1080de98b615a"
    subject = raw_subject("20260929T095127Z_node056_opus55max_4c7eaf92", call_id, "mla-bmm")
    result = investigate(subject, tmp_path / "investigation", max_steps=2)
    report = result["trace"]
    assert result["report_version"] == 1 and "pool" not in result
    assert result["subject"]["attempt"]["id"] == call_id and result["sources"]
    assert result["model_calls"] == 0
    assert report["original_inputs"]["DIRECTION"]["id"] == "r2_d0"
    assert report["effect"]["retained"] is True and report["effect"]["installed_scope"] == "saved_best_at_stop"
    assert report["effect"]["director_status"] == "unknown"
    assert report["effect"]["parent_relative_gain"] is None
    assert report["attempt"]["disposition"] == "trace_only"
    assert report["threshold"]["value"] == pytest.approx(0.008)
    assert report["reviewed_comparisons"] and all(item["threshold_met"] is False for item in report["reviewed_comparisons"])
    assert all(item["certainty"] == "review" for item in report["reviewed_comparisons"])
    assert result["steps"][0]["new_evidence"]


def method_report(tmp_path):
    evidence = tmp_path / "input.json"
    write_json(evidence, {"raw_fixture": "a method was delivered but adoption is inconclusive"})
    ref = reference(evidence)
    attempt = Attempt("one", "train", "online", ref, ref, None, None, Measurement("unknown"), "trace_only", ("fixture",))
    subject = TraceSubject(attempt)
    report = ReverseReport(subject, {"original_inputs": {}, "knowledge": [], "effect": effect_summary(),
                                     "diffs": [{"excerpt": "inconclusive", "evidence": asdict(ref)}]},
                           gaps=(Gap("method_role", "needs_method_review", "Adopted?"),))
    return subject, report, ref


def test_method_investigation_never_repeats_an_unchanged_review(tmp_path, monkeypatch):
    subject, report, ref = method_report(tmp_path)
    monkeypatch.setattr("kernel_workflow.rsi.reverse_engineering.investigate.trace", lambda *args, **kwargs: report)
    calls = []
    def agent(role, payload, directory):
        calls.append((role, payload))
        return {"method_status": "unknown", "rationale": "No observed source diff", "evidence": [asdict(ref)]}
    result = investigate(subject, tmp_path / "review", max_steps=4, model_calls=4, agent=agent)
    assert result["model_calls"] == 1 and len(calls) == 1
    assert result["trace"]["method_reviews"][0]["certainty"] == "review"
    assert result["trace"]["gaps"] == report.to_trace()["gaps"]


def test_method_review_cannot_invent_evidence(tmp_path, monkeypatch):
    subject, report, ref = method_report(tmp_path)
    monkeypatch.setattr("kernel_workflow.rsi.reverse_engineering.investigate.trace", lambda *args, **kwargs: report)
    def agent(*args):
        return {"method_status": "adopted", "rationale": "Unsupported", "evidence": [{**asdict(ref), "sha256": "a" * 64}]}
    with pytest.raises(KnowledgeError, match="invented"):
        investigate(subject, tmp_path / "review", max_steps=1, model_calls=1, agent=agent)


def test_missing_installation_is_unknown_not_no_rollback():
    assert effect_summary(candidate_source="candidate", parent_source="parent")["rolled_back"] is None


def test_exact_body_delivery_requires_matching_raw_result(tmp_path):
    import hashlib
    source = tmp_path / "card.md"
    source.write_text("# Card\nA bounded hint.\n")
    event = tmp_path / "events.jsonl"
    event.write_text(json.dumps({"type": "tool_result", "tool_use_id": "read-1",
                                 "content": "A bounded hint.\n"}) + "\n")
    row = {"card": {"snapshot_path": "card.md", "snapshot_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "ranges": [{"source_start_line": 2, "source_end_line": 2,
                                "result_start_line": 1, "result_end_line": 1}]},
           "evidence": {"event_file": "events.jsonl", "result_event_line": 1, "tool_use_id": "read-1"}}
    assert _body_delivery(tmp_path, row)["certainty"] == "observed"
    row["evidence"]["tool_use_id"] = "different-invocation"
    assert _body_delivery(tmp_path, row)["certainty"] == "unknown"
    row["evidence"]["tool_use_id"] = "read-1"
    row["card"]["ranges"][0] = {"source_start_line": 99, "source_end_line": 100,
                                "result_start_line": 99, "result_end_line": 100}
    assert _body_delivery(tmp_path, row)["certainty"] == "unknown"
