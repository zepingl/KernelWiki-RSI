from dataclasses import asdict, replace

import pytest
from kernel_workflow.rsi.replay_buffer.tests.test_simulator import world
from kernel_workflow.rsi.reverse_engineering.model import (
    Coverage,
    Gap,
    Relation,
    ReverseReport,
    TraceSubject,
)
from kernelwiki.knowledge import KnowledgeError

from .test_trace import method_report


def test_reverse_report_detaches_nested_findings_and_round_trips(tmp_path):
    subject, report, _ = method_report(tmp_path)
    findings = report.findings
    findings["effect"]["retained"] = True
    findings["diffs"].clear()
    assert report.findings["effect"]["retained"] is None and report.findings["diffs"]
    wire = report.to_dict()
    assert wire["report_version"] == 1 and "pool" not in wire
    assert ReverseReport.from_dict(wire).to_dict() == wire
    assert ReverseReport.from_trace(report.to_trace()).to_trace() == report.to_trace()
    wire["subject"]["attempt"]["kernel"] = "forged"
    with pytest.raises(KnowledgeError, match="changed its original attempt"):
        ReverseReport.from_dict(wire)
    with pytest.raises(KnowledgeError, match="authoritative provenance"):
        ReverseReport(subject, {**report.findings, "attempt": asdict(subject.attempt)})


def test_trace_subject_requires_the_actual_direct_parent(tmp_path):
    subject, _, _ = method_report(tmp_path)
    root = world().root
    with pytest.raises(KnowledgeError, match="dispatched source"):
        TraceSubject(subject.attempt, root)
    attempt = replace(subject.attempt, parent_source=root.state.fingerprint, input_context=root.context.revision)
    assert TraceSubject(attempt, root).parent == root
    with pytest.raises(KnowledgeError, match="dispatched context"):
        TraceSubject(replace(attempt, input_context="a" * 64), root)


def test_claims_and_reviews_cannot_establish_observed_relations(tmp_path):
    _, _, ref = method_report(tmp_path)
    with pytest.raises(KnowledgeError, match="cannot prove an observation"):
        Relation("source", "attempt", "body_delivered", (replace(ref, kind="claim"),))
    with pytest.raises(KnowledgeError, match="needs evidence"):
        Relation("source", "attempt", "measured", ())
    relation = Relation("source", "attempt", "method_reviewed", [replace(ref, kind="review")], "review")
    assert isinstance(relation.evidence, tuple)
    assert Relation.from_dict(asdict(relation)) == relation


def test_partial_coverage_cannot_close_history_and_detaches_lists(tmp_path):
    _, _, ref = method_report(tmp_path)
    pending = ["missing.json"]
    coverage = Coverage(["monitor/*.json"], [ref], pending)
    pending.clear()
    assert coverage.status == "partial" and coverage.remaining == ("missing.json",)
    assert Coverage.from_dict(coverage.to_dict()) == coverage
    with pytest.raises(KnowledgeError, match="cannot establish absence"):
        Gap("installation", "unrecoverable_history", "Find saved source", coverage=coverage)
    with pytest.raises(KnowledgeError, match="disagrees"):
        Coverage.from_dict({**coverage.to_dict(), "status": "complete"})
