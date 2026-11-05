import json
import subprocess
from dataclasses import replace

import pytest
from kernel_workflow.rsi.evidence import reference, write_json, write_jsonl
from kernel_workflow.rsi.loop import knowledge_iteration, migration
from kernel_workflow.rsi.loop.iterate import AUTO_SIGNER, iterate
from kernel_workflow.rsi.replay_buffer.cohorts import cohorts
from kernel_workflow.rsi.replay_buffer.tests.test_episode_replay import CARD, knowledge_records, lane_records, state
from kernel_workflow.rsi.trace.records import Expansion, Node, Tree, Use, to_wire
from kernel_workflow.rsi.trace.tree import fold
from kernelwiki import KnowledgeClient
from kernelwiki.knowledge import KnowledgeError

PAGE = "knowledge/general/tuning.md"               # three kernels' winning directions pointed to it
CAUTIONED = "knowledge/general/caution.md"         # three kernels' losing directions pointed to it
CAUTION_CARD = "kernelwiki:knowledge/general/caution.md@abc123456789"
FRONT = "---\ntitle: Tuning\n---\n"
ORIGINAL = FRONT + "# Tuning\nTry the tile sweep first. Then tune the launch grid for the shapes that matter.\n"
CAUTION_PAGE = "# Caution page\nAlways rewrite the epilogue first.\n"
WIN = "a/e2/0"                                     # kernel a's continued, knowledge-free second-round win


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def loop(tmp_path):
    wiki = tmp_path / "wiki"
    for path, text in ((PAGE, ORIGINAL), (CAUTIONED, CAUTION_PAGE),
                       ("knowledge/operator/gemm/tuning.md", "# GEMM tuning\n"),
                       ("knowledge/policy.yaml", "schema_version: 1\nrule_ids: []\ntop_k: 3\nworkers: 2\n"),
                       (".gitignore", ".kernelwiki-*/\n")):
        (wiki / path).parent.mkdir(parents=True, exist_ok=True)
        (wiki / path).write_text(text)
    git(wiki, "init", "-q", "-b", "rsi-exp")
    git(wiki, "add", "-A")
    git(wiki, "commit", "-q", "-m", "seed")
    pool, run = tmp_path / "pool", tmp_path / "run"
    pool.mkdir()
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}\n")
    evidence = (reference(receipt),)
    records = [*knowledge_records(), Tree("kw2", "knowledge:general_page", "kw-gate"),
               Expansion("kw2#author", "kw2", "author", ()), Node(CAUTION_CARD, "kw2", "kw2#author", 0, state("c"))]
    for kernel in "abc":
        records += lane_records(kernel, kernel, ((2.6, 2.2, 2.1), (2.8, 2.5), (3.0,)), evidence=evidence, pointed=(1, 0))
    for kernel in "def":
        records += [replace(record, uses=(Use(CAUTION_CARD, "pointed", evidence),))
                    if isinstance(record, Node) and record.uses else record
                    for record in lane_records(kernel, kernel, ((2.6, 2.2, 2.1), (2.8, 2.5), (3.0,)),
                                               evidence=evidence, pointed=(1, 2))]
    nodes = {record.id: record for record in records if isinstance(record, Node)}
    diff = run / "index/source_diffs" / f"{nodes['a/e1/0'].state}__{nodes[WIN].state}.diff"
    diff.parent.mkdir(parents=True)
    diff.write_text("-    BLOCK_K = 32\n+    BLOCK_K = 64  # halve the k-loop trips\n")
    trees = fold(records).code_trees()
    write_jsonl(pool / "trace.jsonl", [to_wire(record) for record in records])
    write_jsonl(pool / "worlds.jsonl", [])
    write_jsonl(pool / "attempts.jsonl", [])
    write_json(pool / "inputs/campaigns.json", {"fixture": {"path": str(run)}})
    write_json(pool / "manifest.json", {
        "schema_version": 2, "budgets": [1, 2, 4, 6], "objective": {"beta_cost": 0.01, "beta_parallel": 0},
        "splits": {"train": list("abcdef"), "validation": [], "test": []},
        "cohorts": cohorts(()), "tree_cohorts": cohorts(trees, (1, 2, 4, 6), traces="trace.jsonl"),
        "artifacts": {name: reference(pool / name).sha256 for name in ("worlds.jsonl", "attempts.jsonl", "trace.jsonl")}})
    return pool, wiki


CARD_DRAFT = {"title": "Double BLOCK_K to halve k-loop trips", "slug": "double-block-k-to-halve-k-loop-trips",
              "operator": "gemm", "key": "bf16 gemm on gfx950, compute-bound", "description": "A larger K tile.",
              "type": "lever", "kernel_class": "gemm", "regime": "compute-bound", "keywords": ["block-k", "tiling"],
              "levers": ["tile.block_k"], "lever": "Raise BLOCK_K from 32 to 64.", "apply": "Change the constexpr.",
              "verify": "Count k-loop trips in the ISA.", "pitfall": "LDS can overflow at large tiles."}


class Writer:
    """Revises the page by one sentence citing its best direction, distills the diff, approves everything."""

    def __init__(self, review=lambda payload, calls: "approve"):
        self.calls, self.review = [], review

    def __call__(self, role, payload, directory):
        self.calls.append(role)
        if role == "knowledge_reviewer":
            decision = self.review(payload, self.calls)
            return {"decision": decision, "reasons": [f"fixture {decision}"]}
        if payload["op"] == "revise":
            best = max(payload["directions"], key=lambda row: row["gain"])
            return {"rationale": "fixture", "citations": [best["node"]],
                    "body": payload["page"] + "In the recorded runs the tile sweep was the round's best direction.\n"}
        return {"rationale": "fixture", "card": CARD_DRAFT}


def test_iterate_publishes_reviewed_knowledge_and_then_has_nothing_left(loop, tmp_path):
    pool, wiki = loop
    before = KnowledgeClient(wiki).snapshot().revision
    result = iterate(pool, wiki, tmp_path / "out", agent=Writer())
    assert result["status"] == "published"
    assert [(item["op"], item["status"]) for item in result["items"]] == [
        ("caution", "accepted"), ("revise", "accepted"), ("distill", "accepted")]
    # The revised page keeps its front matter; code renders the evidence with the ledger's numbers.
    page = (wiki / PAGE).read_text()
    assert page.startswith(ORIGINAL) and "the round's best direction" in page
    assert "3 paired contrasts" in page and "`a/e1/0`: a, round 1, compute; +30.0% over its parent" in page
    caution = (wiki / CAUTIONED).read_text()
    assert caution.startswith(knowledge_iteration.CAUTION[0]) and caution.endswith(CAUTION_PAGE)
    card = (wiki / f"knowledge/operator/gemm/learned_knowledge/kernel-{CARD_DRAFT['slug']}.md").read_text()
    assert f"`{WIN}`" in card and "effect: +7.7% over its parent on a" in card and "origin: kernelwiki-rsi" in card
    assert f"Approved-by: {AUTO_SIGNER}" in git(wiki, "log", "-1", "--format=%B") and not git(wiki, "status", "--porcelain")
    # Everything this buffer supports is now in KernelWiki: the next run has no work and calls no agent.
    again = Writer()
    assert iterate(pool, wiki, tmp_path / "again", agent=again)["status"] == "converged" and again.calls == []
    # The whole publication can be undone.
    migration.switch(tmp_path / "out/request", wiki, "previous")
    assert KnowledgeClient(wiki).snapshot().revision == before and (wiki / PAGE).read_text() == ORIGINAL


def test_iterate_refuses_the_main_branch(loop, tmp_path):
    pool, wiki = loop
    git(wiki, "branch", "-m", "main")
    with pytest.raises(KnowledgeError, match="experiment branch"):
        iterate(pool, wiki, tmp_path / "out", agent=Writer())


def test_a_rejected_draft_is_redrafted_once_and_then_dropped(loop, tmp_path):
    pool, wiki = loop
    # Every first draft is rejected; the revision's redraft is approved, the card's is rejected again.
    reviews = {}

    def review(payload, calls):
        path = payload["change"][0]["path"]
        reviews[path] = reviews.get(path, 0) + 1
        return "approve" if path == PAGE and reviews[path] == 2 else "reject"

    agent = Writer(review)
    result = iterate(pool, wiki, tmp_path / "out", agent=agent)
    assert {item["op"]: item["status"] for item in result["items"]} == {
        "caution": "accepted", "revise": "accepted", "distill": "rejected"}
    assert agent.calls.count("writer") == 4 and result["agent_calls"] == 8


@pytest.mark.parametrize("op, draft, reason", [
    ("revise", {"rationale": "x", "body": "---\nkernelwiki: x\n---\nbody", "citations": ["a/e1/0"]}, "front matter"),
    ("revise", {"rationale": "x", "body": "Something else entirely.", "citations": ["a/e1/0"]}, "keeps too little"),
    ("revise", {"rationale": "x", "body": ORIGINAL.removeprefix(FRONT) + "x" * 2001, "citations": ["a/e1/0"]},
     "grows by"),
    ("revise", {"rationale": "x", "body": "# Tuning\nTry the tile sweep first. Then tune the launch grid.\n",
                "citations": ["a/e1/1"]}, "directions that used"),
    ("revise", {"rationale": "x", "body": "# Tuning\nTry the tile sweep first. Then tune the launch grid.\n"
                                          + knowledge_iteration.EVIDENCE[0], "citations": ["a/e1/0"]}, "plain text"),
    ("distill", {"rationale": "x", "card": {**CARD_DRAFT, "operator": "nowhere"}}, "not a KernelWiki operator"),
    ("distill", {"rationale": "x", "card": {**CARD_DRAFT, "slug": "Bad Slug"}}, "slug"),
    ("distill", {"rationale": "x", "card": {**CARD_DRAFT, "type": "anti-pattern"}}, "type"),
    ("distill", {"rationale": "x", "card": {key: value for key, value in CARD_DRAFT.items() if key != "verify"}},
     "missing fields"),
])
def test_drafts_that_break_a_rule_are_refused(loop, op, draft, reason):
    pool, wiki = loop
    evidence, files = knowledge_iteration.Evidence(pool), dict(KnowledgeClient(wiki).snapshot().files)
    subject = PAGE if op == "revise" else WIN
    with pytest.raises(KnowledgeError, match=reason):
        knowledge_iteration.render(evidence, {"op": op, "subject": subject}, files, draft, "2026-11-04")


def test_an_existing_card_path_is_not_overwritten(loop):
    pool, wiki = loop
    evidence, files = knowledge_iteration.Evidence(pool), dict(KnowledgeClient(wiki).snapshot().files)
    path = f"knowledge/operator/gemm/learned_knowledge/kernel-{CARD_DRAFT['slug']}.md"
    with pytest.raises(KnowledgeError, match="already exists"):
        knowledge_iteration.render(evidence, {"op": "distill", "subject": WIN}, {**files, path: b"x"},
                                   {"rationale": "x", "card": CARD_DRAFT}, "2026-11-04")


def test_a_caution_is_replaced_not_stacked():
    page = FRONT + "# Body\n"
    once = FRONT + "<!-- kernelwiki-rsi:caution -->\nold\n<!-- /kernelwiki-rsi:caution -->\n\n# Body\n"
    assert knowledge_iteration._unmarked(once) == page
    revised = page.rstrip("\n") + "\n\n<!-- kernelwiki-rsi:evidence -->\nx\n<!-- /kernelwiki-rsi:evidence -->\n"
    assert knowledge_iteration._unmarked(revised) == page


def test_a_writer_may_decline(loop, tmp_path):
    pool, wiki = loop

    def decline(role, payload, directory):
        return {"rationale": "nothing new", "body": None} if payload["op"] == "revise" else \
            {"rationale": "a re-measurement", "card": None}

    result = iterate(pool, wiki, tmp_path / "out", agent=decline)
    assert [item["status"] for item in result["items"]] == ["accepted", "declined", "declined"]
    assert json.loads((tmp_path / "out/proposal/proposal.json").read_text())["items"][0]["op"] == "caution"


def test_an_anti_pattern_card_is_revised_not_cautioned(loop):
    pool, wiki = loop
    files = dict(KnowledgeClient(wiki).snapshot().files)
    files[CAUTIONED] = ("---\ntype: anti-pattern\n---\n" + CAUTION_PAGE).encode()
    items = knowledge_iteration.worklist(knowledge_iteration.Evidence(pool), files)
    assert {"op": "revise", "subject": CAUTIONED} in items and all(item["op"] != "caution" for item in items)
