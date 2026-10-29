import hashlib
import subprocess

from kernel_workflow.rsi.reverse_engineering.sources.kernelwiki import kernelwiki_trees, knowledge_index
from kernel_workflow.rsi.trace.records import Decide
from kernel_workflow.rsi.trace.tree import fold, knowledge_ledger

AITER = "a" * 40
PAGE = "knowledge/operator/gemm/backend/triton_aiter.md"
LEARNED = "knowledge/operator/gemm/learned_knowledge/kernel-lesson.md"
EVIDENCE = "corpus/evidence/gemm_source.yaml"
DECISIONS = "corpus/decisions/gemm.yaml"


def commit(repo, files, message, delete=()):
    for path, text in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text)
    for path in delete:
        (repo / path).unlink()
    subprocess.run(["git", "-C", repo, "add", "-A"], check=True)
    subprocess.run(["git", "-C", repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", message], check=True)
    return subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()[:12]


def wiki(tmp_path):
    repo = tmp_path / "KernelWiki"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", repo], check=True)
    page = f"# Triton GEMM\nSee https://github.com/ROCm/aiter/blob/{AITER}/aiter/ops/gemm.py\n"
    first = commit(repo, {
        PAGE: page,
        LEARNED: "---\norigin: geak\neffect: speedup: 1.2x\n---\n# lesson\n",
        EVIDENCE: f'provenance:\n  aiter_commit: "{AITER}"\n  aiter_dirty_sources: []\nrecords:\n'
                  '  - evidence_id: "src_0123456789abcdef"\n'}, "import")
    second = commit(repo, {
        PAGE: page + "More.\n",
        DECISIONS: "cards:\n  - id: tile\n    source_evidence: [src_0123456789abcdef, src_missing00000000]\n"
                   f"    references: [\"{PAGE}:1-2\", \"knowledge/general/absent.md\"]\n"}, "decide")
    third = commit(repo, {}, "retire lesson", delete=(LEARNED,))
    return repo, page, (first, second, third)


def test_history_becomes_knowledge_trees_with_sources_gaps_and_retirements(tmp_path):
    repo, page, (first, second, third) = wiki(tmp_path)
    records = kernelwiki_trees(repo)
    forest = fold(records)
    assert {view.tree.subject for view in forest.trees.values()} == {
        "external", "knowledge:backend_page", "knowledge:learned_card", "knowledge:evidence_set",
        "knowledge:decision_card"}
    aiter = f"external/ROCm/aiter@{AITER}"
    pages = forest.trees["kernelwiki/backend_page"]
    v1, v2 = f"kernelwiki:{PAGE}@{first}", f"kernelwiki:{PAGE}@{second}"
    assert pages.expansion(pages.nodes[v1]).op == "ingest"
    assert [use.target for use in pages.expansion(pages.nodes[v1]).uses] == [aiter]
    assert pages.expansion(pages.nodes[v2]).op == "revise" and pages.expansion(pages.nodes[v2]).parents == (v1,)
    assert pages.nodes[v1].state == hashlib.sha256(page.encode()).hexdigest()
    assert pages.grade(v2) == ("A", ("no gate defined for this knowledge kind",))

    evidence = forest.trees["kernelwiki/evidence_set"]
    evidence_node = f"kernelwiki:{EVIDENCE}@{first}"
    assert evidence.grade(evidence_node) == (
        "A", ("gate result not recorded (rerunnable: corpus re-extraction and test_corpus)",))

    cards = forest.trees["kernelwiki/decision_card"]
    card = f"kernelwiki:{DECISIONS}#tile@{second}"
    assert {use.target for use in cards.expansion(cards.nodes[card]).uses} == {evidence_node, v2}
    assert {item.gap.question: item.gap.reasons for item in cards.gaps} == {
        "cited evidence id not found": ("src_missing00000000",),
        "cited page not found at this commit": ("knowledge/general/absent.md",)}

    learned = forest.trees["kernelwiki/learned_card"]
    lesson = f"kernelwiki:{LEARNED}@{first}"
    assert learned.expansion(learned.nodes[lesson]).op == "distill"
    assert learned.grade(lesson) == ("T", ("no machine-readable source",))
    assert {item.gap.question for item in learned.gaps} == {"front matter is not valid YAML",
                                                           "no structured run reference"}
    assert learned.decisions == [Decide("kernelwiki/learned_card", lesson, "retired", f"deleted in {third}")]

    index = knowledge_index(records)
    assert index[pages.nodes[v2].state] == v2 and card not in index.values()
    assert knowledge_ledger(forest)[v1]["after"] == [v2]
