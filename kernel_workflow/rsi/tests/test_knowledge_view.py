import json
import subprocess

import pytest
from kernel_workflow.rsi.knowledge_view import materialize, resolve, to_kernelwiki, to_view
from kernelwiki.knowledge import KnowledgeError


@pytest.mark.parametrize("view, kernelwiki", [
    ("operators/dense_gemm/backends/triton.md", "knowledge/operator/dense_gemm/backend/triton.md"),
    ("operators/dense_gemm/overview.md", "knowledge/operator/dense_gemm/README.md"),
    ("operators/dense_gemm/tuning.md", "knowledge/operator/dense_gemm/tuning.md"),
    ("index/taxonomy.md", "knowledge/general/reference/geak_index/taxonomy.md"),
    ("reference/rocm.md", "knowledge/general/reference/rocm.md"),
    ("languages/gluon/programming_model.md", "knowledge/general/backend/gluon/programming_model.md"),
    ("expert_skills/index.yaml", "knowledge/general/expert_recipes/index.yaml"),
    ("profiling/rocprof.md", "knowledge/general/profile_bottleneck/rocprof.md"),
    ("kernelwiki/general/learned_knowledge/a.md", "knowledge/general/learned_knowledge/a.md"),
])
def test_one_rule_table_maps_both_ways(view, kernelwiki):
    assert to_kernelwiki(view) == kernelwiki and to_view(kernelwiki) == view


def test_history_resolves_only_live_documents_and_unique_learned_slugs():
    documents = {"knowledge/operator/dense_gemm/tuning.md", "knowledge/operator/x/learned_knowledge/fast.md",
                 "knowledge/general/learned_knowledge/dup.md", "knowledge/operator/y/learned_knowledge/dup.md"}
    assert resolve("operators/dense_gemm/tuning.md", documents) == "knowledge/operator/dense_gemm/tuning.md"
    assert resolve("operators/softmax/tuning.md", documents) is None          # GEAK-only operator
    assert resolve("languages/gluon/", documents) is None                      # a directory is not a document
    assert resolve("fast.md", documents) == "knowledge/operator/x/learned_knowledge/fast.md"
    assert resolve("dup.md", documents) is None                                # ambiguous slug stays unresolved


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


def test_materialized_view_is_a_pinned_commit_with_a_complete_manifest(tmp_path):
    repo = tmp_path / "wiki"
    files = {"knowledge/operator/gemm/backend/triton.md": "triton v1", "knowledge/operator/gemm/README.md": "gemm",
             "knowledge/general/expert_recipes/index.yaml": "skills: []", "knowledge/policy.yaml": "rule_ids: []",
             "knowledge/general/rsi/rule.md": "policy rule card"}
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)
    git(repo, "init", "-q", "-b", "main")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "v1")
    pinned = git(repo, "rev-parse", "HEAD")
    (repo / "knowledge/operator/gemm/backend/triton.md").write_text("triton v2 (uncommitted)")
    manifest = materialize(repo, "HEAD", tmp_path / "view")
    assert manifest["commit"] == pinned
    assert manifest["files"] == {"expert_skills/index.yaml": "knowledge/general/expert_recipes/index.yaml",
                                 "operators/gemm/backends/triton.md": "knowledge/operator/gemm/backend/triton.md",
                                 "operators/gemm/overview.md": "knowledge/operator/gemm/README.md"}
    assert (tmp_path / "view/operators/gemm/backends/triton.md").read_text() == "triton v1"
    assert json.loads((tmp_path / "view/manifest.json").read_text()) == manifest
    assert git(repo, "status", "--porcelain") == "M knowledge/operator/gemm/backend/triton.md"
    with pytest.raises(FileExistsError):
        materialize(repo, "HEAD", tmp_path / "view")


def test_a_view_needs_a_committed_kernelwiki(tmp_path):
    (tmp_path / "wiki/knowledge").mkdir(parents=True)
    with pytest.raises(KnowledgeError, match="git checkout"):
        materialize(tmp_path / "wiki", "HEAD", tmp_path / "view")


def test_two_documents_on_one_view_path_are_refused(tmp_path):
    repo = tmp_path / "wiki"
    for name in ("knowledge/operator/gemm/README.md", "knowledge/operator/gemm/overview.md"):
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(name)
    git(repo, "init", "-q", "-b", "main")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "v1")
    with pytest.raises(KnowledgeError, match="one view path"):
        materialize(repo, "HEAD", tmp_path / "view")
