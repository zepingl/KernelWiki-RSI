from dataclasses import asdict
from importlib import import_module

import pytest
from kernel_workflow.rsi.evidence import reference, write_json
from kernel_workflow.rsi.reverse_engineering.investigate import investigate
from kernel_workflow.rsi.reverse_engineering.model import Coverage, Gap, ReverseReport
from kernel_workflow.rsi.reverse_engineering.sources.archive import AuditReader
from kernel_workflow.rsi.reverse_engineering.trace import trace

from .test_archive import audit
from .test_trace import method_report

__all__ = ["audit"]


def test_each_new_gap_gets_its_own_evidence_action(tmp_path, monkeypatch):
    subject, initial, original = method_report(tmp_path)
    refs = {}
    for name in ("installation", "parent_measurement"):
        path = tmp_path / (name + ".json")
        write_json(path, {"attempt": subject.attempt.id, "resolved": name})
        refs[name] = reference(path)
    calls = []

    class Reader:
        def search(self, subject, gap, inputs, *, previous, limit):
            calls.append(gap.id)
            assert previous is None
            return ({"document": {"resolved": gap.id}, "evidence": asdict(refs[gap.id])},), Coverage(
                (gap.id,), (refs[gap.id],))

    def retrace(subject, *, documents=(), **kwargs):
        resolved = {item["document"]["resolved"] for item in documents}
        target = "installation" if "installation" not in resolved else "parent_measurement"
        gaps = (() if target in resolved else (Gap(target, "resolvable_local", "Find " + target),))
        return ReverseReport(subject, {**initial.findings, "local_evidence": [item["evidence"] for item in documents]},
                             gaps=(*gaps, Gap("knowledge_causality", "needs_online_control", "Isolate knowledge.")))

    module = import_module("kernel_workflow.rsi.reverse_engineering.investigate")
    monkeypatch.setattr(module, "trace", retrace)
    result = investigate(subject, tmp_path / "investigation", max_steps=4, reader=Reader())
    assert calls == ["installation", "parent_measurement"]
    assert [item["question"]["id"] for item in result["steps"]] == calls
    assert all(item["new_evidence"] and item["coverage"]["status"] == "complete" for item in result["steps"])
    assert set(result["coverage"]) == set(calls)
    assert [item["id"] for item in result["trace"]["gaps"]] == ["knowledge_causality"]
    assert result["model_calls"] == 0 and original in subject.sources


def test_partial_read_budget_preserves_the_gap_and_unread_scope(audit, tmp_path):
    run, _, _ = audit
    write_json(run / "monitor/task-train-save.json", {"call_id": "call-1", "note": "no verified retention manifest"})
    reader = AuditReader(run)
    subject = reader.subject("call-1", "train", "campaign")
    result = investigate(subject, tmp_path / "partial", max_steps=1, read_limit=1, reader=reader)
    step = result["steps"][0]
    assert step["question"]["id"] == "installation"
    assert step["coverage"]["status"] == "partial" and step["coverage"]["remaining"]
    gap = next(item for item in result["trace"]["gaps"] if item["id"] == "installation")
    assert gap["classification"] == "resolvable_local"
    assert gap["coverage"]["status"] == "partial"
    assert result["stop_reason"] == "step_budget"
    assert result["trace"]["effect"]["retained"] is None


def test_complete_search_is_only_absence_within_its_named_scope(audit, tmp_path):
    run, _, _ = audit
    for name in ("artifact_catalog", "artifact_versions", "artifact_transfers"):
        (run / "index" / (name + ".jsonl")).write_text("")
    reader = AuditReader(run)
    subject = reader.subject("call-1", "train", "campaign")
    initial = trace(subject, reader=reader)
    gap = next(item for item in initial.gaps if item.id == "installation")
    _, coverage = reader.search(subject, gap, initial.findings["original_inputs"])
    result = trace(subject, reader=reader, coverages={gap.id: coverage})
    unresolved = next(item for item in result.gaps if item.id == gap.id)
    assert coverage.status == "complete"
    assert unresolved.classification == "unrecoverable_history" and unresolved.coverage.scope
    assert result.findings["effect"]["retained"] is None


@pytest.mark.parametrize("relevant", [False, True])
def test_unknown_review_is_reconsidered_only_after_new_method_evidence(tmp_path, monkeypatch, relevant):
    subject, initial, original = method_report(tmp_path)
    path = tmp_path / "new-diff.json"
    write_json(path, {"attempt": subject.attempt.id, "source_diff": "new implementation evidence"})
    added = reference(path)
    calls = []

    class Reader:
        def search(self, subject, gap, inputs, *, previous, limit):
            return ({"document": {"new_method": relevant}, "evidence": asdict(added)},), Coverage(("monitor",), (added,))

    def retrace(subject, *, documents=(), coverages=None, **kwargs):
        findings = initial.findings
        if documents and relevant:
            findings["diffs"].append({"excerpt": "new verified diff", "evidence": asdict(added)})
        gaps = list(initial.gaps)
        if not coverages:
            gaps.append(Gap("director", "resolvable_local", "Find source-bound Director."))
        return ReverseReport(subject, findings, gaps=gaps)

    def agent(role, payload, directory):
        calls.append(payload)
        return {"method_status": "unknown", "rationale": "Still inconclusive.", "evidence": [asdict(original)]}

    module = import_module("kernel_workflow.rsi.reverse_engineering.investigate")
    monkeypatch.setattr(module, "trace", retrace)
    result = investigate(subject, tmp_path / "review", max_steps=4, model_calls=3, agent=agent, reader=Reader())
    assert result["model_calls"] == len(calls) == (2 if relevant else 1)
    assert result["steps"][0]["action"] == "evidence_bounded_method_review"
    assert result["steps"][1]["question"]["id"] == "director"
    assert all(item["certainty"] == "review" for item in result["trace"]["method_reviews"])
    assert result["trace"]["effect"]["independent_knowledge_causal_gain"] is None
