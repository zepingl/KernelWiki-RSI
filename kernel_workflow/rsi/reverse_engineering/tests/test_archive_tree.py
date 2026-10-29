import hashlib
import json

from kernel_workflow.rsi.reverse_engineering.sources.archive import read_archive
from kernel_workflow.rsi.reverse_engineering.tests.test_archive import build_audit, lines, measurement, save, source
from kernel_workflow.rsi.trace.records import Decide, Expansion, Measure, Node, Tree, Use
from kernel_workflow.rsi.trace.tree import UNREGISTERED, fold, knowledge_coverage
from kernel_workflow.rsi.trace.tests.test_tree import EVIDENCE

LAUNCH = "tasks/task-train/launches/L1"
CARD = "perf_knowledge/operators/gemm/tuning.md"


def lane_run(tmp_path):
    run, _, benchmarks = build_audit(tmp_path, f"{LAUNCH}/agents")
    candidate = benchmarks[1]["candidate_fingerprint"]
    for call_id, result in (("a-plan", {"stop": False, "directions": [
                                {"id": "r1_d0", "kk_refs": [CARD]},
                                {"id": "r1_d1", "learned_refs": ["learned/card.md"]}]}),
                            ("z-plan", {"stop": True, "reasoning": "converged", "directions": []})):
        (run / LAUNCH / "agents" / call_id).mkdir(parents=True)
        (run / LAUNCH / "agents" / call_id / "delivered_prompt.txt").write_text("## Inputs\n- HISTORY: {}")
        (run / LAUNCH / "agents" / call_id / "delivered_result.txt").write_text("```json\n" + json.dumps(result) + "\n```")
    calls = [json.loads(line) for line in (run / "index/agent_calls.jsonl").read_text().splitlines()]
    calls += [{"call_id": call_id, "task": "task-train", "path": f"{LAUNCH}/agents/{call_id}",
               "context": {"phase": "plan_round", "role": "tech_lead"}} for call_id in ("a-plan", "z-plan")]
    lines(run / "index/agent_calls.jsonl", calls)
    snapshot = run / "snapshots/GEAK" / CARD
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("tuning card")
    lines(run / "index/knowledge_lineage.jsonl", [
        *({"kind": "direction_transfer", "call_id": call_id, "direction_transfer": {
            "origin_candidates": [{"call_id": "a-plan", "direction_index": index}]}}
          for index, call_id in ((0, "call-1"), (1, "call-0"))),
        {"kind": "direct_read", "call_id": "call-1", "card": {"ref": CARD, "snapshot_sha256": "d" * 64}}])
    inside = measurement(run, "verify", candidate, "verify", "verify", 2.2, "2026-01-01T01:05:00", base=f"{LAUNCH}/benchmarks")
    lines(run / "index/benchmarks.jsonl", [*benchmarks, inside])
    return run, benchmarks, hashlib.sha256(b"tuning card").hexdigest()


def test_a_lane_launch_becomes_one_expansion_with_labelled_measured_slots(tmp_path):
    run, benchmarks, revision = lane_run(tmp_path)
    history = read_archive(run, campaign="campaign", tasks={"task-train": "train"})
    view = fold(history.trace).trees["campaign/train"]
    assert view.root.state == benchmarks[0]["candidate_fingerprint"]
    assert [item.by for item in view.measures[view.root.id]] == ["director"]
    [expansion] = view.expansions
    assert isinstance(expansion, Expansion) and expansion.parents == (view.root.id,) and expansion.dispatched == 2
    assert [slot.id for slot in expansion.plan.slots] == ["r1_d0", "r1_d1"]
    first, second = (view.nodes[f"{expansion.id}/{slot}"] for slot in (0, 1))
    assert isinstance(first, Node) and first.facets["geak"]["attempt"] == "call-1"
    # Without a KernelWiki index no read resolves: each one is an unregistered-knowledge gap, not a use.
    assert first.uses == second.uses == ()
    assert {(item.subject, item.gap.reasons) for item in view.gaps if item.gap.question == UNREGISTERED} == {
        (first.id, ("pointed", "operators/gemm/tuning.md", revision)),
        (first.id, ("read", "operators/gemm/tuning.md", "d" * 64)),
        (second.id, ("cited", "learned/card.md", "no revision"))}
    # Only the run inside this launch measures these occurrences; it is shared by the identical source.
    assert all(isinstance(item, Measure) and item.by == "verifier" for item in view.measures[first.id])
    assert view.outcome(first.id) == view.outcome(second.id) and view.outcome(first.id).status == "qualified"
    assert [view.grade(node.id)[0] for node in (first, second)] == ["E", "E"]
    assert [item.verdict for item in view.decisions] == ["stop"] and isinstance(view.decisions[0], Decide)


def add_engineer(run, call_id, launch, parent, candidate, canonical):
    """One more engineer dispatch, recorded the same way build_audit records its two."""
    directory = run / launch / "agents" / call_id
    output = str(run / f"experiment/{call_id}/engineer_0")
    workspace = output + "/workspace"
    inputs = {"KERNEL_PATH": workspace, "CANONICAL": canonical, "OUTPUT_DIR": output,
              "DIRECTION": {"id": "r1_d0"}, "SPECIALTY": "compute", "KK_REFS": []}
    prompt = "## Inputs\n" + "\n".join(f"- {key}: {json.dumps(value)}" for key, value in inputs.items())
    directory.mkdir(parents=True)
    (directory / "delivered_prompt.txt").write_text(prompt)
    call = {"call_id": call_id, "time": "2026-01-02T01:00:00", "context": inputs,
            "argv": ["claude", "--model", "claude-opus-5.5", "--effort", "high"],
            "delivered_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    save(directory / "invocation.json", call)
    for name, location, fingerprint in (("pre", canonical, parent), ("post", workspace, parent),
                                        ("final", workspace, candidate)):
        save(directory / f"{name}.json", {"source_states": [{"workspace": location, "fingerprint": fingerprint}],
                                         "artifacts": []})
    relative = f"{launch}/agents/{call_id}"
    base = {"call_id": call_id, "workspace": workspace, "kind": "workspace_created", "event_line": 1,
            "before_fingerprint": None, "after_fingerprint": parent, "pre_observation": f"{relative}/pre.json",
            "post_observation": f"{relative}/post.json", "interval_start": "2026-01-02T01:01:00"}
    changes = [base, {**base, "kind": "source_modified", "event_line": 3, "before_fingerprint": parent,
                      "after_fingerprint": candidate, "pre_observation": f"{relative}/post.json",
                      "post_observation": f"{relative}/final.json", "interval_start": "2026-01-02T01:02:00"}]
    return {**call, "task": "task-train", "context": {**inputs, "phase": "Optimize"}, "path": relative}, changes


def test_a_launch_starting_from_an_unreconstructed_source_opens_a_new_segment(tmp_path):
    run, benchmarks, _ = lane_run(tmp_path)
    second = "tasks/task-train/launches/L2"
    integrated, improved = source(run, "integrated"), source(run, "improved")
    canonical = benchmarks[0]["workspace"]
    engineer, changes = add_engineer(run, "call-9", second, integrated, improved, canonical)
    plan = {"call_id": "b-plan", "task": "task-train", "path": f"{second}/agents/b-plan",
            "context": {"phase": "plan_round", "role": "tech_lead"}}
    (run / second / "agents/b-plan").mkdir(parents=True)
    (run / second / "agents/b-plan/delivered_prompt.txt").write_text("inputs")
    (run / second / "agents/b-plan/delivered_result.txt").write_text(json.dumps({"directions": [{"id": "r1_d0"}]}))
    for name, rows in (("agent_calls", [plan, engineer]), ("source_changes", changes),
                       ("knowledge_lineage", [{"kind": "direction_transfer", "call_id": "call-9", "direction_transfer": {
                           "origin_candidates": [{"call_id": "b-plan", "direction_index": 0}]}}]),
                       ("benchmarks", [measurement(run, "integrated", integrated, canonical, "setup", 2.3,
                                                   "2026-01-02T00:00:00")])):
        path = run / f"index/{name}.jsonl"
        lines(path, [*(json.loads(line) for line in path.read_text().splitlines()), *rows])
    forest = fold(read_archive(run, campaign="campaign", tasks={"task-train": "train"}).trace)
    assert set(forest.trees) == {"campaign/train", "campaign/train/L2"}
    segment = forest.trees["campaign/train/L2"]
    assert segment.root.state == integrated
    # The new segment's plan still saw the earlier launch through HISTORY: information, not ancestry.
    [expansion] = segment.expansions
    assert set(expansion.plan.context.informed_by) == set(forest.trees["campaign/train"].nodes) - {"campaign/train/root"}
    assert segment.grade(f"{expansion.id}/0")[1] == ("no independent measure under the tree contract",
                                                     "context informed by non-ancestor nodes")


def test_reads_resolve_to_kernelwiki_nodes_by_content_and_coverage_counts_the_rest(tmp_path):
    run, _, revision = lane_run(tmp_path)
    card = "kernelwiki/knowledge/operator/gemm/tuning.md@abc1234"
    wiki = [Tree("kernelwiki/page", "knowledge:page", "kernelwiki:page"),
            Expansion(card + "#author", "kernelwiki/page", "author", ()),
            Node(card, "kernelwiki/page", card + "#author", 0, revision)]
    history = read_archive(run, campaign="campaign", tasks={"task-train": "train"}, knowledge={revision: card})
    forest = fold([*wiki, *history.trace])
    first = forest.trees["campaign/train"].nodes["campaign/train/a-plan/0"]
    assert [(use.target, use.role) for use in first.uses] == [(card, "pointed")]
    assert isinstance(first.uses[0], Use) and first.uses[0].evidence != EVIDENCE
    # The engineer read a different snapshot and the learned card names no revision: both stay unregistered.
    assert knowledge_coverage(forest) == {"resolved": 1, "unregistered": 2, "share": 1 / 3}
