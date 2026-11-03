import json
import subprocess

import pytest
from kernel_workflow.rsi.knowledge_view import lane_view, materialize, resolve, to_kernelwiki, to_view
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
                 "knowledge/general/learned_knowledge/dup.md", "knowledge/operator/y/learned_knowledge/dup.md",
                 "knowledge/operator/z/learned_knowledge/kernel-imported.md",
                 "knowledge/general/learned_knowledge/e2e-system.md"}
    assert resolve("operators/dense_gemm/tuning.md", documents) == "knowledge/operator/dense_gemm/tuning.md"
    assert resolve("operators/softmax/tuning.md", documents) is None          # GEAK-only operator
    assert resolve("languages/gluon/", documents) is None                      # a directory is not a document
    assert resolve("fast.md", documents) == "knowledge/operator/x/learned_knowledge/fast.md"
    assert resolve("dup.md", documents) is None                                # ambiguous slug stays unresolved
    # KernelWiki imported GEAK's learned cards as kernel-<slug> and e2e-<slug>.
    assert resolve("imported.md", documents) == "knowledge/operator/z/learned_knowledge/kernel-imported.md"
    assert resolve("system.md", documents) == "knowledge/general/learned_knowledge/e2e-system.md"


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


def test_materialized_view_is_a_pinned_commit_with_a_complete_manifest(tmp_path):
    repo = tmp_path / "wiki"
    files = {"knowledge/operator/gemm/backend/triton.md": "triton v1", "knowledge/operator/gemm/README.md": "gemm",
             "knowledge/operator/gemm/backend/gluon_aiter.md": "gluon from aiter",
             "knowledge/operator/gemm/backend/triton_aiter.md": "triton from aiter",
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
    assert manifest["files"] == {
        "expert_skills/index.yaml": "knowledge/general/expert_recipes/index.yaml",
        "operators/gemm/backends/gluon.md": "knowledge/operator/gemm/backend/gluon_aiter.md",
        "operators/gemm/backends/gluon_aiter.md": "knowledge/operator/gemm/backend/gluon_aiter.md",
        "operators/gemm/backends/triton.md": "knowledge/operator/gemm/backend/triton.md",
        "operators/gemm/backends/triton_aiter.md": "knowledge/operator/gemm/backend/triton_aiter.md",
        "operators/gemm/overview.md": "knowledge/operator/gemm/README.md"}
    # The conventional <lang>.md shows the AITER page only where KernelWiki has no <lang>.md of its own.
    assert manifest["aliases"] == {"operators/gemm/backends/gluon.md": "knowledge/operator/gemm/backend/gluon_aiter.md"}
    assert (tmp_path / "view/operators/gemm/backends/gluon.md").read_text() == "gluon from aiter"
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


def test_lane_view_lays_out_learned_cards_and_prints_opt_in_lane_args(tmp_path):
    repo = tmp_path / "wiki"
    files = {"knowledge/operator/gemm/learned_knowledge/kernel-fast.md": "---\nname: fast.md\n---\n# fast\n",
             "knowledge/operator/gemm/learned_knowledge/README.md": "# about cards\n",
             "knowledge/general/learned_knowledge/e2e-system.md": "---\nname: system.md\n---\n# e2e\n",
             "knowledge/operator/gemm/backend/triton.md": "triton"}
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)
    git(repo, "init", "-q", "-b", "main")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "v1")
    result = lane_view(repo, "HEAD", tmp_path / "out", run_id="run-1")
    manifest = json.loads((tmp_path / "out/view/manifest.json").read_text())
    # Kernel cards appear under GEAK's learned layout; e2e cards and READMEs do not.
    assert {view for view in manifest["files"] if view.startswith("learned/")} == {"learned/fast.md"}
    assert manifest["generated"] == {"learned/INDEX.md": "kernel_workflow/scripts/kb.py index"}
    assert "fast.md" in (tmp_path / "out/view/learned/INDEX.md").read_text()
    assert result["lane_args"] == {"perf_knowledge_dir": str(tmp_path / "out/view"),
                                   "learned_kb_dir": str(tmp_path / "out/view/learned"),
                                   "learned_inbox_dir": str(tmp_path / "out/learned-inbox"), "run_id": "run-1"}
    assert not any((tmp_path / "out/learned-inbox").iterdir())
