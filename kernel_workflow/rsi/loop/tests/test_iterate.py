import json
import subprocess

import pytest
from kernel_workflow.rsi.evidence import reference, write_json, write_jsonl
from kernel_workflow.rsi.loop import knowledge_change, migration
from kernel_workflow.rsi.loop.iterate import AUTO_SIGNER, iterate
from kernel_workflow.rsi.replay_buffer.cohorts import cohorts
from kernel_workflow.rsi.replay_buffer.tests.test_episode_replay import knowledge_records, lane_records
from kernel_workflow.rsi.trace.records import to_wire
from kernel_workflow.rsi.trace.tree import document_ledger, fold
from kernelwiki import KnowledgeClient
from kernelwiki.knowledge import KnowledgeError

PAGE = "knowledge/general/tuning.md"
ORIGINAL = "# Tuning\nTry the tile sweep first.\n"


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def loop(tmp_path):
    """Three kernels whose first round dispatched three directions; only the first pointed to the tuning page."""
    wiki = tmp_path / "wiki"
    (wiki / "knowledge/general").mkdir(parents=True)
    (wiki / "knowledge/policy.yaml").write_text("schema_version: 1\nrule_ids: []\ntop_k: 3\nworkers: 2\n")
    (wiki / PAGE).write_text(ORIGINAL)
    (wiki / ".gitignore").write_text(".kernelwiki-*/\n")
    git(wiki, "init", "-q", "-b", "rsi-exp")
    git(wiki, "add", "-A")
    git(wiki, "commit", "-q", "-m", "seed")
    pool = tmp_path / "pool"
    pool.mkdir()
    records = knowledge_records()
    for kernel in ("train", "validation", "test-secret"):
        receipt = tmp_path / f"{kernel}.json"
        receipt.write_text(json.dumps({"kernel": kernel}) + "\n")
        records += lane_records(kernel, kernel, ((2.6, 2.2, 2.1), (2.8, 2.5), (3.0,)),
                                evidence=(reference(receipt),), pointed=(1, 0))
    trees = fold(records).code_trees()
    write_jsonl(pool / "trace.jsonl", [to_wire(record) for record in records])
    write_jsonl(pool / "worlds.jsonl", [])
    write_jsonl(pool / "attempts.jsonl", [])
    write_json(pool / "manifest.json", {
        "schema_version": 2, "budgets": [1, 2, 4, 6], "objective": {"beta_cost": 0.01, "beta_parallel": 0},
        "splits": {"train": ["train"], "validation": ["validation"], "test": ["test-secret"]},
        "cohorts": cohorts(()), "tree_cohorts": cohorts(trees, (1, 2, 4, 6), traces="trace.jsonl"),
        "artifacts": {name: reference(pool / name).sha256 for name in ("worlds.jsonl", "attempts.jsonl", "trace.jsonl")}})
    return pool, wiki


class LoopAgent:
    """Round 1: a page-conditioned select rule, and field evidence for that page. Afterwards: nothing new."""

    def __init__(self):
        self.proposals = self.curations = 0

    def __call__(self, role, payload, directory):
        if role == "preflight":
            return {"ok": True}
        if role == "reviewer":
            return {"decision": "approve", "reasons": ["fixture"]}
        if role == "curator":
            self.curations += 1
            row = next(row for row in payload["knowledge_evidence"] if row["path"] == PAGE)
            pages = [{"path": PAGE, "section": f"In {row['directions']} recorded first rounds the direction that "
                                               "followed this page beat its siblings (association, 3 runs).",
                      "citations": [row["citable"][0]["node"]]}] if self.curations == 1 else []
            return {"rationale": "fixture", "pages": pages}
        self.proposals += 1
        if self.proposals > 1:
            return {"kind": "evidence_update", "rationale": "no new evidence", "policy": {}, "records": []}
        return {"kind": "policy_update", "rationale": "the tuning-guided direction was the round's best",
                "policy": {"schema_version": 2, "rule_ids": ["select-tuning"], "top_k": 3, "workers": 2,
                           "max_width": 8},
                "records": [{"id": "select-tuning", "title": "Dispatch the tuning-guided direction", "body": "fixture",
                             "applies_to": [], "evidence": payload["training"]["examples"][0]["evidence"],
                             "policy_rule": {"when": [{"field": "knowledge", "op": "contains", "value": PAGE}],
                                             "effect": "select"}}]}


def test_iterate_publishes_gated_changes_until_a_round_adds_nothing(loop, tmp_path):
    pool, wiki = loop
    result = iterate(pool, wiki, tmp_path / "run", rounds=3, agent=LoopAgent(), max_candidates=1)
    assert result["status"] == "converged" and [row["round"] for row in result["rounds"]] == [1, 2]
    assert [item["kind"] for item in result["rounds"][0]["published"]] == ["policy", "amend_pages"]
    assert [item["cause"] for item in result["chain"]] == ["start", "policy", "amend_pages"]
    assert "schema_version: 2" in (wiki / "knowledge/policy.yaml").read_text()
    page = (wiki / PAGE).read_text()
    assert page.startswith(ORIGINAL) and knowledge_change.SECTION_BEGIN in page and "association" in page
    log = git(wiki, "log", "--format=%B", "-2")
    assert log.count(f"Approved-by: {AUTO_SIGNER}") == 2 and not git(wiki, "status", "--porcelain")
    # Round 2 started from the published policy and saw the amended page.
    round_two = tmp_path / "run/round-02/evolve"
    assert json.loads((round_two / "incumbent.json").read_text())["policy"]["schema_version"] == 2
    assert knowledge_change.SECTION_BEGIN in (round_two / "frozen_knowledge" / PAGE).read_text()
    # Any round can be undone, newest publication first.
    migration.switch(tmp_path / "run/round-01/pages", wiki, "previous")
    assert (wiki / PAGE).read_text() == ORIGINAL
    migration.switch(tmp_path / "run/round-01/policy", wiki, "previous")
    assert KnowledgeClient(wiki).snapshot().revision == result["chain"][0]["revision"]


def test_iterate_refuses_the_main_branch(loop, tmp_path):
    pool, wiki = loop
    git(wiki, "branch", "-m", "main")
    with pytest.raises(KnowledgeError, match="experiment branch"):
        iterate(pool, wiki, tmp_path / "run", rounds=1, agent=LoopAgent())


@pytest.mark.parametrize("page, reason", [
    ({"path": "knowledge/general/other.md", "section": "x", "citations": ["train/e1/0"]}, "recorded runs used"),
    ({"path": PAGE, "section": "x", "citations": ["train/e1/1"]}, "cite directions that used it"),
    ({"path": PAGE, "section": "x" * 1501, "citations": ["train/e1/0"]}, "plain text"),
    ({"path": PAGE, "section": "---\nkernelwiki:\n  id: x\n---", "citations": ["train/e1/0"]}, "plain text"),
])
def test_field_evidence_is_confined_to_used_pages_and_their_directions(loop, page, reason):
    pool, _ = loop
    from kernel_workflow.rsi.replay_buffer.store import open_buffer
    ledger = document_ledger(fold(open_buffer(pool).trace))
    with pytest.raises(KnowledgeError, match=reason):
        knowledge_change.check_pages([page], ledger, {PAGE, "knowledge/general/other.md"})


def test_amending_twice_equals_amending_once_with_the_last_evidence():
    once = knowledge_change.amend(ORIGINAL.encode(), "evidence A")
    assert once.decode().startswith(ORIGINAL) and once.decode().count(knowledge_change.SECTION_BEGIN) == 1
    assert knowledge_change.amend(once, "evidence B") == knowledge_change.amend(ORIGINAL.encode(), "evidence B")
