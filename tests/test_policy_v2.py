import pytest
from kernelwiki.knowledge import (
    Condition,
    KnowledgeError,
    KnowledgePolicy,
    KnowledgeRecord,
    PolicyRule,
    read_markdown,
    render_markdown,
    validate_policy,
)

CARD = "knowledge:backend_page/operator/gemm/backend/triton.md@" + "c" * 64


def rule(name, effect, when=(), value=None):
    return KnowledgeRecord(name, name, "Evidence-bounded scheduling.\n", (), (), PolicyRule(tuple(when), effect, value))


def test_v2_rules_round_trip_through_markdown():
    for expected in (rule("narrow-when-stuck", "width", (Condition("stagnation", "gte", 2),), 1),
                     rule("skip-deep-late", "skip", (Condition("mode", "eq", "deep"), Condition("round", "gte", 3))),
                     rule("prefer-tuning-card", "select", (Condition("knowledge", "contains", CARD),)),
                     rule("stop-when-done", "stop", (Condition("best_gain", "gte", 0.5),))):
        assert KnowledgeRecord.from_dict(expected.to_dict()) == expected
        assert read_markdown("general/rsi/policy.md", render_markdown(expected).decode()) == expected


def test_v1_cards_keep_their_exact_serialized_form():
    v1 = rule("prefer-leaf", "prefer", (Condition("node_kind", "eq", "leaf"),))
    assert "value" not in v1.to_dict()["policy_rule"]
    assert "max_width" not in KnowledgePolicy(rule_ids=("prefer-leaf",)).to_dict()


def test_scopes_values_and_set_operator_are_checked():
    with pytest.raises(KnowledgeError, match="conditions for width"):
        PolicyRule((Condition("specialty", "eq", "compute"),), "width", 2)
    with pytest.raises(KnowledgeError, match="conditions for prefer"):
        PolicyRule((Condition("knowledge", "contains", CARD),), "prefer")
    with pytest.raises(KnowledgeError, match="only a width rule"):
        PolicyRule((), "select", 2)
    with pytest.raises(KnowledgeError, match="width must be"):
        PolicyRule((), "width", 9)
    with pytest.raises(KnowledgeError, match="contains"):
        Condition("knowledge", "eq", CARD)
    assert Condition("knowledge", "contains", CARD).matches({"knowledge": frozenset({CARD})})
    assert not Condition("knowledge", "contains", CARD).matches({"knowledge": frozenset()})


def test_policy_schema_and_rule_effects_must_agree():
    width = rule("narrow", "width", (), 2)
    leaf = rule("prefer-leaf", "prefer", (Condition("node_kind", "eq", "leaf"),))
    validate_policy(KnowledgePolicy(schema_version=2, rule_ids=("narrow",), max_width=4), (width,))
    with pytest.raises(KnowledgeError, match="schema 1"):
        validate_policy(KnowledgePolicy(rule_ids=("narrow",)), (width,))
    with pytest.raises(KnowledgeError, match="schema 2"):
        validate_policy(KnowledgePolicy(schema_version=2, rule_ids=("prefer-leaf",), max_width=4), (leaf,))
    with pytest.raises(KnowledgeError, match="max_width"):
        KnowledgePolicy(schema_version=2)
    conflicting = rule("narrower", "width", (), 1)
    with pytest.raises(KnowledgeError, match="conflicting effects"):
        validate_policy(KnowledgePolicy(schema_version=2, rule_ids=("narrow", "narrower"), max_width=4),
                        (width, conflicting))
