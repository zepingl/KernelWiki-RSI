import pytest
from kernel_workflow.rsi.trace.gaps import Gap
from kernel_workflow.rsi.trace.records import (
    CODE,
    ContextRef,
    Decide,
    Expansion,
    GapRecord,
    Generation,
    Measure,
    Node,
    Plan,
    Slot,
    Tree,
    Use,
    from_wire,
    to_wire,
)
from kernel_workflow.rsi.trace.tree import UNREGISTERED, fold, gap_worklist, knowledge_coverage, knowledge_ledger
from kernelwiki.knowledge import EvidenceRef, KernelState, KnowledgeError

EVIDENCE = (EvidenceRef("fixture.json", "a" * 64),)
CLAIM = (EvidenceRef("claim.json", "b" * 64, kind="claim"),)
GEN = Generation("fixed-agent", "high")
CARD = "kw/tuning@1"
CARD_USE = Use(CARD, "pointed", EVIDENCE)


def state(digit):
    return str(digit) * 64


def expansion(name, parent, *, round=1, informed=(), slots=1, op="generate", tree="t", uses=()):
    return Expansion(f"{tree}/{name}", tree, op, (parent,),
                     Plan(round, ContextRef(informed, EVIDENCE), GEN, tuple(Slot(f"d{index}") for index in range(slots)),
                          slots), uses)


def measure(node, score, contract="contract-v1"):
    return Measure(node, contract, "verifier", True, score, EVIDENCE)


def wiki():
    """One AITER commit and one KernelWiki card ingested from it."""
    return [
        Tree("ext", "external", "external"), Node("ext/ROCm/aiter@abc1234", "ext", None, 0, "abc1234"),
        Tree("kw", "knowledge:learned_card", "kernelwiki:learned_card"),
        Expansion("kw/tuning@1#ingest", "kw", "ingest", (), uses=(Use("ext/ROCm/aiter@abc1234", "source", EVIDENCE),)),
        Node(CARD, "kw", "kw/tuning@1#ingest", 0, state(9)),
    ]


def lane(informed=(), card=None):
    """root -> e1{a, b}; a committed -> e2{c}; then stop."""
    return [
        Tree("t", CODE, "contract-v1", KernelState(operator="attention"),
             facets={"geak": {"campaign": "c", "kernel": "train", "launch": "l"}}),
        Node("t/root", "t", None, 0, state(0)), measure("t/root", 2.0),
        expansion("e1", "t/root", slots=2),
        Node("t/e1/0", "t", "t/e1", 0, state(1), (card,) if card else ()), measure("t/e1/0", 2.4),
        Node("t/e1/1", "t", "t/e1", 1, state(2)), measure("t/e1/1", 2.2),
        Decide("t", "t/e1/0", "committed", "inferred"),
        expansion("e2", "t/e1/0", round=2, informed=informed),
        Node("t/e2/0", "t", "t/e2", 0, state(3)), measure("t/e2/0", 2.6),
        Decide("t", None, "stop", "done"),
    ]


def test_every_record_round_trips_through_the_wire():
    records = [*wiki(), *lane(informed=("t/e1/0", "t/e1/1"), card=CARD_USE)]
    assert [from_wire(to_wire(record)) for record in records] == records
    with pytest.raises(KnowledgeError, match="schema version"):
        from_wire({**to_wire(records[0]), "schema_version": 3})


def test_claims_never_become_observed_use_or_independent_measure():
    with pytest.raises(KnowledgeError, match="claim cannot establish"):
        Use(CARD, "read", CLAIM)
    assert Use(CARD, "pointed", CLAIM).role == "pointed"
    with pytest.raises(KnowledgeError, match="observed receipts"):
        Measure("t/root", "contract-v1", "engineer", True, 2.0, CLAIM)


def test_fold_rejects_forward_references_contract_changes_and_broken_commit_chains():
    records = lane()
    with pytest.raises(KnowledgeError, match="earlier nodes"):
        fold([records[0], expansion("e1", "t/root")])
    with pytest.raises(KnowledgeError, match="tree contract"):
        fold([*records[:2], measure("t/root", 2.0, contract="other")])
    sideways = [*records[:9], Decide("t", "t/e1/1", "committed", "second sibling")]
    with pytest.raises(KnowledgeError, match="previous commit"):
        fold(sideways)
    with pytest.raises(KnowledgeError, match="slot of an earlier expansion"):
        fold([*records[:4], Node("t/e1/2", "t", "t/e1", 2, state(4))])


def test_delegate_mounts_child_trees_only_on_dispatched_slots():
    parent = [Tree("e2e", CODE, "throughput-v1"), Node("e2e/root", "e2e", None, 0, state(0)),
              expansion("milestone", "e2e/root", op="delegate", slots=2, tree="e2e")]
    lane_tree = Tree("lane", CODE, "latency-v1", parent={"tree": "e2e", "expansion": "e2e/milestone", "slot": 1})
    assert set(fold([*parent, lane_tree]).trees) == {"e2e", "lane"}
    with pytest.raises(KnowledgeError, match="delegate slot"):
        fold([*parent, Tree("lane", CODE, "latency-v1",
                            parent={"tree": "e2e", "expansion": "e2e/milestone", "slot": 2})])
    generate = [*parent[:2], expansion("round", "e2e/root", tree="e2e")]
    with pytest.raises(KnowledgeError, match="delegate slot"):
        fold([*generate, Tree("lane", CODE, "latency-v1",
                              parent={"tree": "e2e", "expansion": "e2e/round", "slot": 0})])
    with pytest.raises(KnowledgeError, match="only a code tree"):
        Tree("kw", "knowledge:card", "gate", parent={"tree": "e2e", "expansion": "e2e/milestone", "slot": 0})


def test_grades_follow_ancestry_of_information_and_independent_measurement():
    ancestor_only = fold(lane(informed=("t/root", "t/e1/0"))).trees["t"]
    assert {node: ancestor_only.grade(node)[0] for node in ("t/e1/0", "t/e1/1", "t/e2/0")} == \
        {"t/e1/0": "E", "t/e1/1": "E", "t/e2/0": "E"}
    sibling_memory = fold(lane(informed=("t/e1/0", "t/e1/1"))).trees["t"]
    assert sibling_memory.grade("t/e2/0") == ("A", ("context informed by non-ancestor nodes",))
    unmeasured = fold([record for record in lane() if record != measure("t/e1/1", 2.2)]).trees["t"]
    assert unmeasured.grade("t/e1/1") == ("A", ("no independent measure under the tree contract",))
    unobserved = fold([*lane()[:4], Node("t/e1/0", "t", "t/e1", 0, None)]).trees["t"]
    assert unobserved.grade("t/e1/0")[0] == "T"


def test_outcome_is_the_lower_median_and_a_failure_verdict_wins():
    records = [*lane(), measure("t/e2/0", 3.0), measure("t/e2/0", 2.2), measure("t/e2/0", 2.4)]
    view = fold(records).trees["t"]
    assert view.outcome("t/e2/0").primary_score == 2.4
    failed = fold([*records, Decide("t", "t/e2/0", "incorrect", "reference mismatch")]).trees["t"]
    assert failed.outcome("t/e2/0").status == "failed"
    assert view.children("t/root") and [node.id for node in view.children("t/root")] == ["t/e1/0", "t/e1/1"]


def test_each_subject_owns_its_ops_measurers_and_verdicts():
    with pytest.raises(KnowledgeError, match="carries a plan"):
        Expansion("kw/x", "kw", "revise", ("a",), Plan(1, ContextRef((), EVIDENCE), GEN, (), 1))
    with pytest.raises(KnowledgeError, match="carries a plan"):
        Expansion("t/x", "t", "generate", ("t/root",))
    with pytest.raises(KnowledgeError, match="at least two parents"):
        Expansion("kw/x", "kw", "merge", (CARD,))
    with pytest.raises(KnowledgeError, match="not belong to a kernel_source tree"):
        fold([*lane()[:2], Expansion("t/x", "t", "revise", ("t/root",))])
    with pytest.raises(KnowledgeError, match="not belong to a knowledge"):
        fold([*wiki(), expansion("x", CARD, tree="kw")])
    with pytest.raises(KnowledgeError, match="produced by an expansion"):
        fold([*wiki(), Node("kw/loose", "kw", None, 0, state(8))])
    with pytest.raises(KnowledgeError, match="commit it pins"):
        fold([Tree("ext", "external", "external"), Node("ext/bad", "ext", None, 0, "not-a-commit")])
    with pytest.raises(KnowledgeError, match="passes or fails through a decision"):
        Measure(CARD, "kernelwiki:learned_card", "schema_gate", True, 1.0, EVIDENCE)
    with pytest.raises(KnowledgeError, match="does not belong to a knowledge"):
        fold([*wiki(), Measure(CARD, "kernelwiki:learned_card", "verifier", True, 2.0, EVIDENCE)])
    with pytest.raises(KnowledgeError, match="does not belong to a knowledge"):
        fold([*wiki(), Decide("kw", CARD, "committed", "not a code tree")])
    assert fold([*wiki(), Decide("kw", CARD, "retired", "superseded")]).trees["kw"].decisions[0].verdict == "retired"


def test_code_reads_only_kernelwiki_nodes_and_knowledge_cites_any_earlier_node():
    with pytest.raises(KnowledgeError, match="earlier node"):
        fold([*lane(card=CARD_USE)])
    with pytest.raises(KnowledgeError, match="only reads KernelWiki"):
        fold([*wiki(), *lane(card=Use("ext/ROCm/aiter@abc1234", "read", EVIDENCE))])
    with pytest.raises(KnowledgeError, match="only reads KernelWiki"):
        fold([*wiki(), *lane(card=Use(CARD, "source", EVIDENCE))])
    with pytest.raises(KnowledgeError, match="only cites its sources"):
        fold([*wiki(), Expansion("kw/x", "kw", "revise", (CARD,), uses=(Use(CARD, "read", EVIDENCE),))])
    distilled = [*wiki(), *lane(card=CARD_USE),
                 Expansion("kw/lesson@1#distill", "kw", "distill", (), uses=(Use("t/e1/0", "source", EVIDENCE),)),
                 Node("kw/lesson@1", "kw", "kw/lesson@1#distill", 0, state(7)),
                 Expansion("kw/merged@1#merge", "kw", "merge", (CARD, "kw/lesson@1")),
                 Node("kw/merged@1", "kw", "kw/merged@1#merge", 0, state(6))]
    view = fold(distilled).trees["kw"]
    assert view.children(CARD) == view.children("kw/lesson@1") == [view.nodes["kw/merged@1"]]


def test_knowledge_grades_need_sources_and_a_recorded_gate():
    view = fold(wiki()).trees["kw"]
    assert view.grade(CARD) == ("A", ("no gate defined for this knowledge kind",))
    gated = [Tree("kw", "knowledge:decision_card", "render-check", params={"gate": "render --check"}),
             Expansion("kw/c#author", "kw", "author", ()), Node("kw/c", "kw", "kw/c#author", 0, state(5))]
    assert fold(gated).trees["kw"].grade("kw/c") == ("T", ("no machine-readable source",))
    cited = [*wiki()[:2], gated[0],
             Expansion("kw/c#author", "kw", "author", (), uses=(Use("ext/ROCm/aiter@abc1234", "source", CLAIM),)),
             gated[2]]
    assert fold(cited).trees["kw"].grade("kw/c") == ("A", ("a source is only claimed",
                                                          "gate result not recorded (rerunnable: render --check)"))
    measured = [*wiki(), Measure(CARD, "kernelwiki:learned_card", "schema_gate", True, None, EVIDENCE),
                Decide("kw", CARD, "accepted", "gate passed")]
    assert fold(measured).trees["kw"].grade(CARD) == ("E", ())


def test_ledger_centres_on_kernelwiki_nodes_and_worklist_explains_non_e_nodes():
    forest = fold([*wiki(), *lane(informed=("t/e1/0", "t/e1/1"), card=CARD_USE)])
    row = knowledge_ledger(forest)[CARD]
    assert row["op"] == "ingest" and row["sources"] == ["ext/ROCm/aiter@abc1234"]
    assert row["before"] == [] and row["after"] == [] and row["roles"] == {"pointed": 1}
    assert row["contrasts"] == [{"expansion": "t/e1", "with": ["t/e1/0"], "without": ["t/e1/1"],
                                 "gain": pytest.approx(2.4 / 2.2 - 1)}]
    assert gap_worklist(forest) == [
        {"node": CARD, "grade": "A", "blocked_by": ["no gate defined for this knowledge kind"], "gaps": []},
        {"node": "t/e2/0", "grade": "A", "blocked_by": ["context informed by non-ancestor nodes"], "gaps": []}]
    assert forest.code_trees() == (forest.trees["t"],)


def test_coverage_counts_resolved_reads_against_unregistered_ones():
    unregistered = GapRecord("t/e1/1", Gap("t/e1/1#read", "needs_online_control", UNREGISTERED, ("read", "x.md")))
    forest = fold([*wiki(), *lane(card=CARD_USE), unregistered])
    assert knowledge_coverage(forest) == {"resolved": 1, "unregistered": 1, "share": 0.5}
    assert knowledge_coverage(fold(lane()))["share"] is None


def test_facets_never_change_invariants_or_grades():
    plain = lane()
    decorated = [Node(record.id, record.tree, record.expansion, record.slot, record.state, record.uses,
                      {"other": {"anything": 1}}) if isinstance(record, Node) else record for record in plain]
    left, right = fold(plain).trees["t"], fold(decorated).trees["t"]
    assert [left.grade(node) for node in left.nodes] == [right.grade(node) for node in right.nodes]
