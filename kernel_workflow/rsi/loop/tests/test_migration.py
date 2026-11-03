import json
import subprocess

import pytest
from kernel_workflow.rsi.harness import Observation, Policy
from kernel_workflow.rsi.loop import knowledge_change, migration
from kernel_workflow.rsi.loop.inner import evolve
from kernel_workflow.rsi.loop.tests.test_inner_episodes import WidthDeveloper, episode_pool
from kernel_workflow.rsi.reverse_engineering.sources.kernelwiki import kernelwiki_trees
from kernel_workflow.rsi.trace.tree import fold
from kernelwiki import KnowledgeClient
from kernelwiki.knowledge import KernelState, KnowledgeError


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def round_and_wiki(tmp_path):
    pool, wiki = episode_pool(tmp_path)
    (wiki / ".gitignore").write_text(".kernelwiki-*/\n")  # as in KernelWiki: version state stays out of git
    git(wiki, "init", "-q", "-b", "main")
    git(wiki, "add", "-A")
    git(wiki, "commit", "-q", "-m", "seed")
    evolve(pool, wiki, tmp_path / "evolve", agent=WidthDeveloper(), max_candidates=1, episodes="step")
    return tmp_path / "evolve", wiki


def test_signed_migration_publishes_rolls_back_and_restores(round_and_wiki, tmp_path):
    evolve_dir, wiki = round_and_wiki
    original = (wiki / "knowledge/policy.yaml").read_bytes()
    request = tmp_path / "request"
    value = migration.request(evolve_dir, wiki, request)
    assert [change["id"] for change in value["changes"]] == ["narrow"]
    assert value["replay_gate"]["candidate"]["objective"] > value["replay_gate"]["incumbent"]["objective"]
    with pytest.raises(KnowledgeError, match="signed approval"):
        migration.publish(request, wiki)
    migration.sign(request, "reviewer@example")
    with pytest.raises(KnowledgeError, match="already decided"):
        migration.sign(request, "someone-else")
    text = (request / "request.json").read_text()
    (request / "request.json").write_text(text.replace("pooled", "Pooled", 1))
    with pytest.raises(KnowledgeError, match="different request"):
        migration.publish(request, wiki)
    (request / "request.json").write_text(text)

    published = migration.publish(request, wiki)
    assert published["revision"] == KnowledgeClient(wiki).snapshot().revision != published["previous_revision"]
    assert "schema_version: 2" in (wiki / "knowledge/policy.yaml").read_text()
    assert (wiki / "knowledge/general/rsi/narrow.md").exists()
    assert git(wiki, "rev-parse", "HEAD") == published["commit"] and not git(wiki, "status", "--porcelain")
    assert "Approved-by: reviewer@example" in git(wiki, "log", "-1", "--format=%B")
    with pytest.raises(KnowledgeError, match="parent scheduler runs schema 1"):
        Policy(KnowledgeClient(wiki).snapshot().policy, KnowledgeClient(wiki).snapshot().records).select(
            Observation(KernelState(), (), 1, 0, 1, 1))

    back = migration.switch(request, wiki, "previous")
    assert (wiki / "knowledge/policy.yaml").read_bytes() == original
    assert not (wiki / "knowledge/general/rsi/narrow.md").exists()
    assert KnowledgeClient(wiki).snapshot().revision == published["previous_revision"]
    forward = migration.switch(request, wiki, "published")
    assert KnowledgeClient(wiki).snapshot().revision == published["revision"]
    assert back["status"] == "switched_to_previous" and forward["status"] == "switched_to_published"
    assert int(git(wiki, "rev-list", "--count", "HEAD")) == 4
    with pytest.raises(KnowledgeError, match="already published"):
        migration.publish(request, wiki)


def test_request_refuses_a_round_that_kept_the_incumbent(tmp_path):
    pool, wiki = episode_pool(tmp_path, kernels=())
    evolve(pool, wiki, tmp_path / "evolve", agent=WidthDeveloper(), max_candidates=1, episodes="step")
    with pytest.raises(KnowledgeError, match="selected a working policy"):
        migration.request(tmp_path / "evolve", wiki, tmp_path / "request")


CARD = "---\nname: {slug}\nkernel_class: {cls}\n---\n# {slug}\n- lever: keep the text\n- source: run fixture, 2026-08-01\n"


@pytest.fixture
def geak_and_wiki(tmp_path):
    geak, wiki = tmp_path / "geak", tmp_path / "wiki"
    learned = geak / "kernel_workflow/knowledge/learned"
    learned.mkdir(parents=True)
    (learned / "README.md").write_text("# not a card\n")
    for slug, cls in (("imported.md", "gemm"), ("dropped.md", "quantize_cast")):
        (learned / slug).write_text(CARD.format(slug=slug, cls=cls))
    (wiki / "knowledge/operator/gemm/learned_knowledge").mkdir(parents=True)
    (wiki / "knowledge/operator/gemm/learned_knowledge/kernel-imported.md").write_text("imported earlier\n")
    (wiki / "knowledge/policy.yaml").write_text("schema_version: 1\nrule_ids: []\ntop_k: 3\nworkers: 2\n")
    (wiki / ".gitignore").write_text(".kernelwiki-*/\n")
    for repo in (geak, wiki):
        git(repo, "init", "-q", "-b", "main")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "seed")
    return geak, wiki


def test_missing_geak_cards_are_ingested_verbatim_signed_and_reversible(geak_and_wiki, tmp_path):
    geak, wiki = geak_and_wiki
    commit = git(geak, "rev-parse", "HEAD")
    proposal = knowledge_change.propose(geak, commit, wiki, tmp_path / "proposal", date="2026-11-03")
    target = "knowledge/general/learned_knowledge/kernel-dropped.md"
    assert [item["path"] for item in proposal["files"]] == [target]
    request = tmp_path / "request"
    value = migration.request(tmp_path / "proposal", wiki, request)
    assert value["kind"] == "ingest_learned" and value["changes"][0]["kernel_class"] == "quantize_cast"
    with pytest.raises(KnowledgeError, match="signed approval"):
        migration.publish(request, wiki)
    migration.sign(request, "reviewer@example")
    page = tmp_path / "proposal/files" / target
    original = page.read_bytes()
    page.write_bytes(original.replace(b"keep the text", b"edited text"))
    with pytest.raises(KnowledgeError, match="not its rendered GEAK source"):
        migration.publish(request, wiki)
    page.write_bytes(original)

    published = migration.publish(request, wiki)
    text = (wiki / target).read_text()
    assert text.startswith("---\norigin: geak\nevidence_scope: kernel\nname: dropped.md\n")
    assert f"https://github.com/AMD-AGI/GEAK/blob/{commit}/kernel_workflow/knowledge/learned/dropped.md" in text
    assert git(wiki, "rev-parse", "HEAD") == published["commit"] and not git(wiki, "status", "--porcelain")
    # The projection sees a new learned-card version whose source is the pinned GEAK commit.
    forest = fold(kernelwiki_trees(wiki, published["commit"]))
    node = published["versions"][target]
    view = forest.trees["kernelwiki/learned_card"]
    assert {use.target for use in view.expansion(view.nodes[node]).uses} == {f"external/AMD-AGI/GEAK@{commit}"}

    migration.switch(request, wiki, "previous")
    assert not (wiki / target).exists()
    assert KnowledgeClient(wiki).snapshot().revision == published["previous_revision"]
    migration.switch(request, wiki, "published")
    assert (wiki / target).read_text() == text
    assert KnowledgeClient(wiki).snapshot().revision == published["revision"]


def test_nothing_to_ingest_is_an_explicit_answer(geak_and_wiki, tmp_path):
    geak, wiki = geak_and_wiki
    (wiki / "knowledge/general/learned_knowledge").mkdir(parents=True)
    (wiki / "knowledge/general/learned_knowledge/kernel-dropped.md").write_text("already here\n")
    git(wiki, "add", "-A")
    git(wiki, "commit", "-q", "-m", "add")
    with pytest.raises(KnowledgeError, match="already in KernelWiki"):
        knowledge_change.propose(geak, "HEAD", wiki, tmp_path / "proposal")
