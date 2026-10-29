import json

from kernel_workflow.rsi.evidence import reference, write_json, write_jsonl
from kernel_workflow.rsi.loop.inner import evolve
from kernel_workflow.rsi.replay_buffer.cohorts import cohorts
from kernel_workflow.rsi.replay_buffer.tests.test_episode_replay import knowledge_records, lane_records
from kernel_workflow.rsi.trace.records import to_wire
from kernel_workflow.rsi.trace.tree import fold
from kernelwiki import KnowledgeClient


def episode_pool(tmp_path, kernels=("train", "validation", "test-secret")):
    wiki = tmp_path / "wiki"
    (wiki / "knowledge").mkdir(parents=True)
    (wiki / "knowledge/policy.yaml").write_text("schema_version: 1\nrule_ids: []\ntop_k: 3\nworkers: 2\n")
    pool = tmp_path / "pool"
    pool.mkdir()
    records = knowledge_records()
    for kernel in kernels:
        receipt = tmp_path / f"{kernel}.json"
        receipt.write_text(json.dumps({"kernel": kernel, "synthetic_fixture": True}) + "\n")
        records += lane_records(kernel, kernel, ((2.6, 2.2, 2.1), (2.8, 2.5), (3.0,)), evidence=(reference(receipt),))
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


class WidthDeveloper:
    def __init__(self):
        self.calls = []

    def __call__(self, role, payload, directory):
        self.calls.append(role)
        assert "test-secret" not in json.dumps(payload) and '"kernel": "validation"' not in json.dumps(payload)
        if role == "preflight":
            return {"ok": True}
        if role == "reviewer":
            return {"decision": "approve", "reasons": ["fixture"]}
        assert payload["training"]["examples"][0]["steps"]
        return {"kind": "policy_update", "rationale": "Dispatching only the first planned direction keeps quality",
                "policy": {"schema_version": 2, "rule_ids": ["narrow"], "top_k": 3, "workers": 2, "max_width": 8},
                "records": [{"id": "narrow", "title": "Narrow", "body": "fixture", "applies_to": [],
                             "evidence": payload["training"]["examples"][0]["evidence"],
                             "policy_rule": {"when": [], "effect": "width", "value": 1}}]}


def test_episode_evolution_selects_a_cheaper_width_without_publishing(tmp_path):
    pool, wiki = episode_pool(tmp_path)
    before = KnowledgeClient(wiki).snapshot().revision
    agent = WidthDeveloper()
    result = evolve(pool, wiki, tmp_path / "out", agent=agent, max_candidates=1, episodes="step")
    assert result["status"] == "working_policy_selected" and result["scope"] == "complete"
    selected = json.loads((tmp_path / "out/selected-policy.json").read_text())
    assert selected["policy"]["schema_version"] == 2 and not result["published"]
    assert KnowledgeClient(wiki).snapshot().revision == before
    assert agent.calls == ["preflight", "developer", "reviewer"]


def test_missing_kernels_block_unless_explicitly_diagnostic(tmp_path):
    pool, wiki = episode_pool(tmp_path, kernels=("validation", "test-secret"))
    blocked = evolve(pool, wiki, tmp_path / "blocked", agent=WidthDeveloper(), max_candidates=1, episodes="step")
    assert blocked["status"] == "insufficient_primary_support" and blocked["agent_calls"] == 0
