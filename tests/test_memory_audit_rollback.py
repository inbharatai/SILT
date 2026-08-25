"""Memory separation, rollback snapshots and audit-chain integrity."""

from __future__ import annotations

import json

import pytest

from asea.audit.logger import AuditLog
from asea.core.errors import AuditIntegrityError, RollbackError
from asea.core.protocol import PacketType, PromotionStatus
from asea.memory.store import APPROVED, CANDIDATE, REJECTED, MemoryStore, RollbackLayer


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "memory")


def _packet(capability, provenance, factory, **kw):
    return factory(
        capability, provenance,
        packet_type=PacketType.GLOSSARY,
        distilled_skill={"entries": [{"source": "ভাত", "target": "rice"}]},
        **kw,
    )


# -- separation -------------------------------------------------------------


def test_buckets_are_physically_separate(store, capability, clean_provenance, packet_factory):
    candidate = _packet(capability, clean_provenance, packet_factory)
    store.put_candidate(candidate)
    assert store.count(CANDIDATE) == 1
    assert store.count(APPROVED) == 0
    assert (store.root / CANDIDATE).exists() and (store.root / APPROVED).exists()


def test_receiver_only_ever_reads_approved(store, capability, clean_provenance, packet_factory):
    """The containment property: candidates are invisible to the receiver."""
    candidate = _packet(capability, clean_provenance, packet_factory)
    store.put_candidate(candidate)
    assert store.approved_skills(candidate.target_module) == []

    candidate.rollback_token = "snap"
    candidate.promotion_status = PromotionStatus.PROMOTED
    store.approve(candidate)
    visible = store.approved_skills(candidate.target_module)
    assert len(visible) == 1
    assert "sender_output" not in visible[0]


def test_store_refuses_to_approve_unpromoted_packet(store, capability, clean_provenance, packet_factory):
    packet = _packet(capability, clean_provenance, packet_factory)
    with pytest.raises(RollbackError, match="refusing to write"):
        store.approve(packet)


def test_approval_removes_the_candidate_copy(store, capability, clean_provenance, packet_factory):
    packet = _packet(capability, clean_provenance, packet_factory)
    store.put_candidate(packet)
    packet.rollback_token = "s"
    packet.promotion_status = PromotionStatus.PROMOTED
    store.approve(packet)
    assert store.count(CANDIDATE) == 0
    assert store.count(APPROVED) == 1


def test_approved_skills_filter_by_target_and_capability(store, capability, clean_provenance, packet_factory):
    packet = _packet(capability, clean_provenance, packet_factory)
    packet.rollback_token = "s"
    packet.promotion_status = PromotionStatus.PROMOTED
    store.approve(packet)
    assert store.approved_skills("someone-else") == []
    assert len(store.approved_skills(packet.target_module, capability.as_str())) == 1
    assert store.approved_skills(packet.target_module, "other/text/general/-") == []


def test_round_trip_preserves_unicode(store, capability, clean_provenance, packet_factory):
    packet = _packet(capability, clean_provenance, packet_factory)
    store.put_candidate(packet)
    loaded = store.get(CANDIDATE, packet.packet_id)
    assert loaded.distilled_skill["entries"][0]["source"] == "ভাত"


def test_rejected_packets_are_retained_for_audit(store, capability, clean_provenance, packet_factory):
    packet = _packet(
        capability, clean_provenance, packet_factory,
        promotion_status=PromotionStatus.REJECTED, rejection_reason="unsafe",
    )
    store.put_rejected(packet)
    assert store.count(REJECTED) == 1
    assert store.stats() == {"candidate": 0, "approved": 0, "rejected": 1}


# -- rollback ---------------------------------------------------------------


def test_snapshot_and_rollback_restores_prior_state(store, capability, clean_provenance, packet_factory):
    rollback = RollbackLayer(store)

    first = _packet(capability, clean_provenance, packet_factory)
    first.rollback_token = "s0"
    first.promotion_status = PromotionStatus.PROMOTED
    store.approve(first)

    token = rollback.snapshot(label="before second promotion")

    second = _packet(capability, clean_provenance, packet_factory)
    # Distinct content: the duplicate-content guard (adversarial audit A2)
    # correctly refuses identical payloads for the same receiver.
    second.distilled_skill = {"entries": [{"source": "পানী", "target": "water"}]}
    second.rollback_token = token
    second.promotion_status = PromotionStatus.PROMOTED
    store.approve(second)
    assert store.count(APPROVED) == 2

    result = rollback.rollback(token)
    assert result["restored"] == 1
    assert store.count(APPROVED) == 1
    assert store.list(APPROVED)[0].packet_id == first.packet_id


def test_snapshot_metadata_is_listed(store):
    rollback = RollbackLayer(store)
    token = rollback.snapshot(label="empty state")
    entries = rollback.list_snapshots()
    assert len(entries) == 1
    assert entries[0]["token"] == token
    assert entries[0]["label"] == "empty state"
    assert entries[0]["packet_count"] == 0


def test_rollback_to_unknown_token_fails_loudly(store):
    with pytest.raises(RollbackError, match="unknown snapshot"):
        RollbackLayer(store).rollback("no-such-token")


# -- audit ------------------------------------------------------------------


@pytest.fixture
def log(tmp_path):
    return AuditLog(tmp_path / "audit" / "audit.jsonl")


def test_audit_chain_verifies(log):
    for i in range(5):
        log.append("event_{}".format(i), actor="tester", detail={"i": i})
    assert log.verify() == {"ok": True, "entries": 5}
    log.assert_intact()


def test_audit_entries_are_linked(log):
    first = log.append("a", actor="t")
    second = log.append("b", actor="t")
    assert second["prev_hash"] == first["hash"]
    assert first["prev_hash"] == "0" * 64


def test_tampering_with_content_is_detected(log):
    log.append("a", actor="t", detail={"amount": 1})
    log.append("b", actor="t", detail={"amount": 2})

    entries = log.entries()
    entries[0]["detail"]["amount"] = 999  # forge history
    with open(log.path, "w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")

    result = log.verify()
    assert result["ok"] is False
    assert result["broken_at"] == 0
    with pytest.raises(AuditIntegrityError):
        log.assert_intact()


def test_deleting_a_line_is_detected(log):
    for i in range(3):
        log.append("e{}".format(i), actor="t")
    entries = log.entries()
    del entries[1]
    with open(log.path, "w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")
    assert log.verify()["ok"] is False


def test_verify_reports_missing_field_instead_of_keyerror(log):
    """A tamper (or torn write) that deletes a tracked field must surface as a
    structured {ok: False, broken_at, reason}, not a raw KeyError that crashes
    assert_intact() and its callers (adversarial audit 2026-08-13 #9/#34)."""
    log.append("a", actor="t", detail={"amount": 1})
    entries = log.entries()
    # Delete a tracked payload field from the first (and only) entry. The line
    # stays valid JSON, so entries() parses it fine; verify() must flag it.
    del entries[0]["session_id"]
    with open(log.path, "w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")

    result = log.verify()
    assert result["ok"] is False
    assert result["broken_at"] == 0
    assert "missing field" in result["reason"]
    assert "session_id" in result["reason"]
    # assert_intact() must raise the structured error, not KeyError.
    with pytest.raises(AuditIntegrityError):
        log.assert_intact()


def test_audit_filters_by_packet_and_session(log):
    log.append("x", actor="t", packet_id="p1", session_id="s1")
    log.append("y", actor="t", packet_id="p2", session_id="s1")
    log.append("z", actor="t", packet_id="p1", session_id="s2")
    assert len(log.for_packet("p1")) == 2
    assert len(log.for_session("s1")) == 2
