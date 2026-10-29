from dataclasses import replace

import pytest
from kernelwiki.knowledge import (
    Condition,
    KernelState,
    KnowledgeError,
    KnowledgePolicy,
    KnowledgeRecord,
    PolicyRule,
    query,
    read_markdown,
    render_markdown,
    validate_policy,
)


def record(name="prefer-leaf", effect="prefer"):
    return KnowledgeRecord(name, "Prefer measured continuations", "Evidence-bounded scheduling.\n",
                           (Condition("backend", "eq", "triton"),), (),
                           PolicyRule((Condition("node_kind", "eq", "leaf"),), effect))


def test_roundtrip_and_existing_markdown_normalization():
    expected = record()
    assert KnowledgeRecord.from_dict(expected.to_dict()) == expected
    assert read_markdown("general/policy.md", render_markdown(expected).decode()) == expected
    ordinary = read_markdown("operator/moe/backend/triton/notes.md", "# Notes\nSome reference text.")
    assert ordinary.policy_rule is None
    assert ordinary.matches(KernelState(operator="moe", backend="triton"))
    assert not ordinary.matches(KernelState(operator="attention", backend="triton"))


def test_retrieval_is_deterministic_and_unknown_does_not_match():
    records = (record("zeta"), record("alpha"), KnowledgeRecord("general", "General", ""))
    assert [item.id for item in query(records, KernelState(backend="triton"), 2)] == ["alpha", "zeta"]
    assert [item.id for item in query(records, KernelState(), 2)] == ["general"]


def test_legacy_front_matter_is_plain_text_but_explicit_namespace_is_strict():
    legacy = "---\nlegacy: text: not YAML\n---\n# Legacy\nReference text."
    assert read_markdown("general/legacy.md", legacy).body == legacy
    with pytest.raises(KnowledgeError, match="invalid KernelWiki front matter"):
        read_markdown("general/invalid.md", "---\nkernelwiki: text: invalid\n---\n# Invalid")


def test_backend_file_taxonomy_matches_operator_and_backend():
    reference = read_markdown("operator/dense_gemm/backend/triton_aiter.md", "# Triton reference")
    assert reference.matches(KernelState(operator="dense_gemm", backend="triton"))
    assert not reference.matches(KernelState(operator="dense_gemm", backend="flydsl"))
    assert not reference.matches(KernelState(operator="batched_gemm", backend="triton"))
    overview = read_markdown("general/backend/README.md", "# Backend overview")
    assert overview.matches(KernelState(backend="triton"))
    with pytest.raises(KnowledgeError, match="top_k"):
        query((reference,), KernelState(), 4)


@pytest.mark.parametrize("condition", [
    {"field": "source_id", "op": "eq", "value": "secret"},
    {"field": "parent_gain", "op": "gte", "value": float("nan")},
    {"field": "parent_gain", "op": "gte", "value": True},
    {"field": "node_kind", "op": "eval", "value": "open('/tmp')"},
    {"field": "node_kind", "op": "in", "value": []},
    {"field": "node_kind", "op": "eq", "value": "hidden"},
    {"field": "measurement_status", "op": "in", "value": ["qualified", "invented"]},
])
def test_policy_rejects_hidden_fields_code_and_invalid_values(condition):
    with pytest.raises(KnowledgeError):
        Condition.from_dict(condition)


def test_policy_resolves_rules_and_rejects_conflicts():
    with pytest.raises(KnowledgeError, match="missing executable"):
        validate_policy(KnowledgePolicy(rule_ids=("missing",)), ())
    with pytest.raises(KnowledgeError, match="conflicting"):
        validate_policy(KnowledgePolicy(rule_ids=("alpha", "beta")),
                        (record("alpha"), record("beta", "defer")))
    with pytest.raises(KnowledgeError, match="stop"):
        replace(record().policy_rule, effect="stop")
    with pytest.raises(KnowledgeError):
        KnowledgePolicy.from_dict({"schema_version": 1, "rule_ids": [], "python": "pass"})
