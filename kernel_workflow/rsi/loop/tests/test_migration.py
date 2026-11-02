import json
import subprocess

import pytest
from kernel_workflow.rsi.harness import Observation, Policy
from kernel_workflow.rsi.loop import migration
from kernel_workflow.rsi.loop.inner import evolve
from kernel_workflow.rsi.loop.tests.test_inner_episodes import WidthDeveloper, episode_pool
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
