import json
from dataclasses import asdict

import pytest
from kernel_workflow.rsi.evidence import read_json, reference, write_json
from kernel_workflow.rsi.replay_buffer.build import freeze, ingest
from kernel_workflow.rsi.replay_buffer.store import open_buffer
from kernel_workflow.rsi.reverse_engineering.investigate import investigate
from kernel_workflow.rsi.reverse_engineering.sources.archive import AuditReader
from kernel_workflow.rsi.reverse_engineering.tests.test_archive import audit
from kernelwiki.knowledge import KnowledgeError

__all__ = ["audit"]


def test_snapshot_owns_read_only_selection_and_lookup(audit, tmp_path):
    _, config, _ = audit
    pool = tmp_path / "pool"
    ingest(config, pool)
    buffer = open_buffer(pool)
    metadata = buffer.manifest
    metadata["budgets"].append(99)
    metadata["cohorts"]["primary"].append(buffer.worlds[0].id)
    assert buffer.budgets == (1, 2, 4, 6) and buffer.select() == ()
    assert buffer.select("train", cohort="diagnostic") == buffer.worlds
    assert buffer.attempt("call-1").id == "call-1"
    assert buffer.parent("call-1").measurement.status == "qualified"
    assert buffer.subject("call-1").attempt == buffer.attempt("call-1")
    assert buffer.reverse_reports() == ()
    with pytest.raises(KnowledgeError, match="unknown attempt"):
        buffer.attempt("not-inventoried")


def test_standalone_reverse_report_freezes_once_without_an_intermediate_pool(audit, tmp_path):
    run, config, _ = audit
    pool, output = tmp_path / "pool", tmp_path / "buffer"
    ingest(config, pool)
    subject = AuditReader(run).subject("call-1", "train", "campaign")
    report = investigate(subject, tmp_path / "investigation", max_steps=1)
    assert "pool" not in report and report["report_version"] == 1
    freeze(pool, (), output, investigations=(tmp_path / "investigation",))
    buffer = open_buffer(output)
    assert len(buffer.worlds) == 1 and len(buffer.attempts) == 2
    assert buffer.reverse_reports()[0].to_trace() == report["trace"]
    before = reference(output / "manifest.json")
    with pytest.raises(FileExistsError):
        freeze(pool, (), output, investigations=(tmp_path / "investigation",))
    assert reference(output / "manifest.json") == before


def test_schema_two_pool_bound_investigations_remain_readable(audit, tmp_path):
    _, config, _ = audit
    pool = tmp_path / "pool"
    ingest(config, pool)
    current = investigate(open_buffer(pool).subject("call-1"), tmp_path / "current", max_steps=1)
    legacy = tmp_path / "legacy"
    write_json(legacy / "result.json", {"status": current["status"], "trace": current["trace"],
                                      "pool": asdict(reference(pool / "manifest.json")), "model_calls": 0,
                                      "steps": [], "artifacts": {}})
    freeze(pool, (), tmp_path / "combined", investigations=(legacy,))
    assert open_buffer(tmp_path / "combined").reverse_reports()[0].to_trace() == current["trace"]


def test_investigation_cannot_relabel_an_inventoried_attempt(audit, tmp_path):
    _, config, _ = audit
    pool, directory = tmp_path / "pool", tmp_path / "investigation"
    ingest(config, pool)
    investigate(open_buffer(pool).subject("call-1"), directory, max_steps=1)
    result = read_json(directory / "result.json")
    result["subject"]["attempt"]["kernel"] = "forged"
    result["trace"]["attempt"]["kernel"] = "forged"
    (directory / "result.json").write_text(json.dumps(result))
    with pytest.raises(KnowledgeError, match="changed the inventoried attempt"):
        freeze(pool, (), tmp_path / "rejected", investigations=(directory,))
    assert not (tmp_path / "rejected/manifest.json").exists()
    assert open_buffer(pool).attempt("call-1").kernel == "train"


def test_schema_one_buffers_are_not_silently_requalified(tmp_path):
    write_json(tmp_path / "manifest.json", {"schema_version": 1})
    with pytest.raises(KnowledgeError, match="requires reconstruction"):
        open_buffer(tmp_path)
