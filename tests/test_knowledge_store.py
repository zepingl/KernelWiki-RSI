import fcntl
import hashlib
from dataclasses import replace

import kernelwiki.knowledge_store as storage
import pytest
from kernelwiki import (
    EvidenceRef,
    KernelState,
    KnowledgeClient,
    KnowledgeError,
    KnowledgePolicy,
    KnowledgeRecord,
    KnowledgeUpdate,
)
from kernelwiki.knowledge import Condition, PolicyRule


@pytest.fixture
def client(tmp_path):
    root = tmp_path / "wiki"
    (root / "knowledge/general").mkdir(parents=True)
    (root / "knowledge/general/reference.md").write_text("# Reference\nKeep this human-authored page.\n")
    (root / "knowledge/policy.yaml").write_text("schema_version: 1\nrule_ids: []\ntop_k: 3\nworkers: 2\n")
    return KnowledgeClient(root)


def candidate(tmp_path):
    evidence = tmp_path / "replay.json"
    evidence.write_text('{"fixture":true,"validation":"passed"}\n')
    ref = EvidenceRef(str(evidence), hashlib.sha256(evidence.read_bytes()).hexdigest())
    rule = KnowledgeRecord("continue", "Continue qualified branches", "Synthetic publication fixture.\n",
                           (), (ref,), PolicyRule((Condition("node_kind", "eq", "leaf"),), "prefer"))
    return KnowledgeUpdate(KnowledgePolicy(rule_ids=(rule.id,)), (rule,),
                           {"status": "replay_validated"}, (ref,))


def test_publish_pinned_read_and_rollback(client, tmp_path):
    original = client.snapshot()
    update = candidate(tmp_path)
    revision = client.publish(update, original.revision)
    assert revision != original.revision
    assert client.snapshot().policy.rule_ids == ("continue",)
    assert client.snapshot(original.revision) == original
    assert client.query(KernelState(), revision)[0].id == "continue"
    assert (client.root / "knowledge/general/reference.md").read_text().startswith("# Reference")
    assert client.rollback(original.revision, revision) == original.revision
    assert client.snapshot() == original
    assert not (client.root / "knowledge/general/rsi/continue.md").exists()
    assert client.snapshot(revision).policy.rule_ids == ("continue",)


def test_stale_revision_and_user_edits_abort_without_overwrite(client, tmp_path):
    original = client.snapshot()
    update = candidate(tmp_path)
    revision = client.publish(update, original.revision)
    with pytest.raises(KnowledgeError, match="stale"):
        client.publish(update, original.revision)
    path = client.root / "knowledge/general/reference.md"
    path.write_text("User edit\n")
    with pytest.raises(KnowledgeError, match="user modifications"):
        client.rollback(original.revision, revision)
    assert path.read_text() == "User edit\n"


def test_lock_failure_does_not_fall_back_to_unserialized_writes(client, tmp_path):
    client.state.mkdir()
    with (client.state / "write.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(KnowledgeError, match="another knowledge writer"):
            client.publish(candidate(tmp_path), client.snapshot().revision)


def test_interrupted_sync_pins_old_read_and_recovers_forward(client, tmp_path, monkeypatch):
    original = client.snapshot()
    update = candidate(tmp_path)
    original_write = storage.atomic_bytes

    def interrupted(path, payload, **kwargs):
        if str(path).endswith("active.json"):
            raise OSError("injected disk failure at active-pointer update")
        return original_write(path, payload, **kwargs)

    monkeypatch.setattr(storage, "atomic_bytes", interrupted)
    with pytest.raises(OSError, match="injected"):
        client.publish(update, original.revision)
    assert client.snapshot() == original
    monkeypatch.setattr(storage, "atomic_bytes", original_write)
    recovered = client.recover()
    assert recovered != original.revision
    assert client.snapshot().policy.rule_ids == ("continue",)
    assert not (client.state / "pending.json").exists()


def test_recovery_preserves_intervening_user_edit(client, tmp_path, monkeypatch):
    original = client.snapshot()
    original_sync = client._sync
    monkeypatch.setattr(client, "_sync", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("interrupted")))
    with pytest.raises(OSError):
        client.publish(candidate(tmp_path), original.revision)
    (client.root / "knowledge/general/reference.md").write_text("User edit\n")
    monkeypatch.setattr(client, "_sync", original_sync)
    with pytest.raises(KnowledgeError, match="user edits conflict"):
        client.recover()


def test_tampered_snapshot_and_evidence_are_rejected(client, tmp_path):
    original = client.snapshot()
    update = candidate(tmp_path)
    revision = client.publish(update, original.revision)
    path = client.state / "versions" / revision / "knowledge/policy.yaml"
    path.write_text("modified")
    with pytest.raises(KnowledgeError, match="snapshot was modified"):
        client.snapshot(revision)


def test_symlinks_and_unqualified_promotion_are_rejected(client, tmp_path):
    original = client.snapshot()
    update = candidate(tmp_path)
    with pytest.raises(KnowledgeError, match="publication requires"):
        client.publish(replace(update, validation={"status": "online_validated"}), original.revision)
    (client.root / "knowledge/link.md").symlink_to(tmp_path / "outside")
    with pytest.raises(KnowledgeError, match="symlinks"):
        client.snapshot()


def test_evidence_update_cannot_bypass_replay_qualification(client, tmp_path):
    original = client.snapshot()
    update = candidate(tmp_path)
    with pytest.raises(KnowledgeError, match="cannot change policy behavior"):
        client.publish(replace(update, validation={"status": "evidence_update"}), original.revision)
    reference_record = KnowledgeRecord("new-reference", "Reference", "Evidence-only note.")
    revision = client.publish(replace(update, policy=original.policy, records=(reference_record,),
                                      validation={"status": "evidence_update"}), original.revision)
    assert client.snapshot(revision).policy == original.policy


def test_explicit_revision_rejects_symlinked_state_root(client, tmp_path):
    revision = client.publish(candidate(tmp_path), client.snapshot().revision)
    moved = tmp_path / "moved-state"
    client.state.rename(moved)
    client.state.symlink_to(moved, target_is_directory=True)
    with pytest.raises(KnowledgeError, match="symlinks"):
        client.snapshot(revision)


def test_changed_record_evidence_cannot_hide_behind_different_update_evidence(client, tmp_path):
    update = candidate(tmp_path)
    original = client.snapshot().revision
    changed = tmp_path / "changed.json"
    changed.write_text("changed")
    record = replace(update.records[0], evidence=(replace(update.evidence[0], path=str(changed)),))
    with pytest.raises(KnowledgeError, match="record evidence changed"):
        client.publish(replace(update, records=(record,)), original)
    assert client.snapshot().revision == original


def test_verbatim_pages_are_added_published_and_rolled_back_without_touching_policy(client, tmp_path):
    original = client.snapshot()
    evidence = tmp_path / "provenance.json"
    evidence.write_text('{"source": "GEAK@02bfbb0b"}\n')
    ref = EvidenceRef(str(evidence), hashlib.sha256(evidence.read_bytes()).hexdigest())
    card = b"---\nname: imported\norigin: geak\n---\n# Imported card\n- lever: keep it verbatim\n"
    name = "knowledge/general/learned_knowledge/kernel-imported.md"
    update = KnowledgeUpdate(original.policy, (), {"status": "evidence_update"}, (ref,), ((name, card),))
    revision = client.publish(update, expected_revision=original.revision)
    assert (client.root / name).read_bytes() == card
    assert client.snapshot().policy == original.policy and dict(client.snapshot().files)[name] == card
    assert (client.root / "knowledge/policy.yaml").read_text().startswith("schema_version: 1\n")   # not reformatted
    client.rollback(original.revision, expected_revision=revision)
    assert not (client.root / name).exists() and client.snapshot().revision == original.revision


@pytest.mark.parametrize("name, payload, reason", [
    ("knowledge/general/notes.txt", b"text\n", "knowledge/\\*.md"),
    ("corpus/outside.md", b"# outside\n", "knowledge/\\*.md"),
    ("knowledge/general/rule.md", b"---\nkernelwiki:\n  id: rule\n  title: Rule\n  policy_rule:\n    when: []\n"
                                  b"    effect: prefer\n---\nbody\n", "policy rule"),
])
def test_verbatim_pages_cannot_leave_knowledge_or_carry_rules(client, tmp_path, name, payload, reason):
    original = client.snapshot()
    evidence = tmp_path / "provenance.json"
    evidence.write_text("{}\n")
    ref = EvidenceRef(str(evidence), hashlib.sha256(evidence.read_bytes()).hexdigest())
    update = KnowledgeUpdate(original.policy, (), {"status": "evidence_update"}, (ref,), ((name, payload),))
    with pytest.raises(KnowledgeError, match=reason):
        client.publish(update, expected_revision=original.revision)
    assert client.snapshot().revision == original.revision


def test_a_reference_page_can_be_replaced_verbatim_and_rolled_back(client, tmp_path):
    original = client.snapshot()
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}\n")
    ref = EvidenceRef(str(evidence), hashlib.sha256(evidence.read_bytes()).hexdigest())
    page = "knowledge/general/reference.md"
    before = (client.root / page).read_bytes()
    amended = before + b"\n<!-- field evidence -->\nMeasured in 12 runs.\n"
    update = KnowledgeUpdate(original.policy, (), {"status": "evidence_update"}, (ref,), ((page, amended),))
    revision = client.publish(update, expected_revision=original.revision)
    assert (client.root / page).read_bytes() == amended
    client.rollback(original.revision, expected_revision=revision)
    assert (client.root / page).read_bytes() == before
