"""End-to-end runs over the shipped sample data."""

from __future__ import annotations

import pytest

from asea.core.pipeline import Pipeline
from asea.core.protocol import Domain, LearningLevel, PromotionStatus
from asea.modules.mock.zoo import (
    make_generic_receiver,
    make_generic_sender,
    make_qwen,
    rule_cap,
    text_cap,
)
from asea.promotion.gate import PromotionGate, PromotionPolicy


def knowledge(suites, splits=("extraction",), overrides=None):
    table = {}
    for s in suites:
        bucket = table.setdefault(s.capability().as_str(), {})
        for case in s.cases:
            if case.split in splits:
                bucket[str(case.prompt).strip()] = case.expected
        for k, v in (overrides or {}).items():
            if k in bucket:
                bucket[k] = v
    return table


@pytest.fixture
def assamese_setup(tmp_path, as_en_suite, hi_en_suite):
    sender = make_generic_sender(
        module_id="assamese-corpus-mock",
        capabilities=[text_cap("translate", "as->en", Domain.TRANSLATION)],
        knowledge=knowledge([as_en_suite], overrides={"চাহ": "coffee"}),
    )
    receiver_knowledge = knowledge([hi_en_suite], splits=("regression",))
    receiver_knowledge.setdefault(
        as_en_suite.capability().as_str(), {}
    )["ভাল"] = "good"
    receiver = make_qwen(knowledge=receiver_knowledge, fallback="echo")

    pipeline = Pipeline(
        workspace=tmp_path / "ws",
        gate=PromotionGate(PromotionPolicy(strict_no_mock=False)),
    )
    pipeline.register_module(sender)
    pipeline.register_module(receiver)
    pipeline.bind_adapter("as-to-qwen", sender.module_id, receiver.module_id)
    return pipeline, [as_en_suite, hi_en_suite]


def test_assamese_flow_promotes_with_measured_gain(assamese_setup):
    pipeline, suites = assamese_setup
    report = pipeline.run("as-to-qwen", suites=suites)

    assert report.negotiation["actionable"] == 1
    assert len(report.promoted) == 1
    evaluation = report.evaluations[0]
    assert evaluation["improvement"] > 0.1
    assert evaluation["regressions"][0]["regressed"] is False
    assert pipeline.store.stats()["approved"] == 1
    assert pipeline.audit.verify()["ok"] is True


def test_assamese_flow_exercises_both_drop_reasons(assamese_setup):
    """The shipped data is seeded to trigger these; if it stops, the data drifted."""
    pipeline, suites = assamese_setup
    report = pipeline.run("as-to-qwen", suites=suites)
    reasons = " ".join(d["reason"] for d in report.dropped_relevance)
    assert "receiver_competent" in reasons
    assert "sender_incorrect" in reasons


def test_raw_sender_output_never_reaches_the_approved_store(assamese_setup):
    pipeline, suites = assamese_setup
    pipeline.run("as-to-qwen", suites=suites)
    for packet in pipeline.store.list("approved"):
        assert packet.sender_output is None
    for view in pipeline.store.approved_skills("qwen-mock"):
        assert "sender_output" not in view


def test_second_run_finds_no_gap_after_learning(assamese_setup):
    """Idempotence: once promoted, the same transfer must not repeat.

    Distillation is deterministic, so the second run produces content-identical
    packets -- and the duplicate-content guard (adversarial audit A2) must
    refuse to approve them a second time, recording the refusal in the audit.
    """
    pipeline, suites = assamese_setup
    first = pipeline.run("as-to-qwen", suites=suites)
    second = pipeline.run("as-to-qwen", suites=suites)
    hashes_first = {p.content_hash() for p in first.distilled}
    hashes_second = {p.content_hash() for p in second.distilled}
    assert hashes_first == hashes_second, "distillation must be deterministic"

    assert first.promoted, "first run promotes"
    assert not second.promoted, "second run must not double-promote"
    assert second.rejected, "duplicate is refused, not silently dropped"
    assert pipeline.store.stats()["approved"] == 1
    assert any(e["event"] == "duplicate_refused" for e in pipeline.audit.entries())


def test_no_gap_means_no_transfer(tmp_path, as_en_suite):
    """A receiver that already performs well must not be taught anything."""
    cap = as_en_suite.capability()
    full = knowledge([as_en_suite], splits=("extraction", "heldout"))
    sender = make_generic_sender(module_id="src", capabilities=[cap], knowledge=full)
    receiver = make_generic_receiver(
        module_id="already-good", capabilities=[cap], knowledge=full
    )
    pipeline = Pipeline(workspace=tmp_path / "ws2")
    pipeline.register_module(sender)
    pipeline.register_module(receiver)
    pipeline.bind_adapter("a", "src", "already-good")
    report = pipeline.run("a", suites=[as_en_suite])

    assert report.negotiation["actionable"] == 0
    assert report.extracted == 0
    assert report.promoted == []


def test_medical_flow_parks_for_human_and_promotes_only_after_approval(
    tmp_path, medical_suite
):
    cap = rule_cap(Domain.MEDICAL, "triage")
    sender = make_generic_sender(
        module_id="triage-corpus-mock", capabilities=[cap],
        knowledge=knowledge([medical_suite]),
    )
    receiver = make_generic_receiver(
        module_id="small-medical-mock", capabilities=[cap], fallback="english"
    )
    pipeline = Pipeline(
        workspace=tmp_path / "med",
        gate=PromotionGate(PromotionPolicy(strict_no_mock=False)),
    )
    pipeline.register_module(sender)
    pipeline.register_module(receiver)
    pipeline.bind_adapter("med", sender.module_id, receiver.module_id)

    report = pipeline.run("med", suites=[medical_suite])
    assert report.promoted == []
    assert len(report.pending_human) == 1
    assert pipeline.store.approved_skills(receiver.module_id) == []

    packet_id = report.pending_human[0]
    decision = pipeline.approve_pending(packet_id, approver="dr.reviewer@example.org")
    assert decision["status"] == PromotionStatus.PROMOTED.value
    assert len(pipeline.store.approved_skills(receiver.module_id)) == 1

    events = [e["event"] for e in pipeline.audit.for_packet(packet_id)]
    assert "pending_human_approval" in events
    assert "human_decision" in events
    approver_events = [
        e for e in pipeline.audit.entries() if e["actor"] == "dr.reviewer@example.org"
    ]
    assert approver_events, "the approver must be named in the audit trail"


def test_rollback_undoes_a_promotion(assamese_setup):
    pipeline, suites = assamese_setup
    report = pipeline.run("as-to-qwen", suites=suites)
    assert pipeline.store.stats()["approved"] == 1

    packet = pipeline.store.list("approved")[0]
    result = pipeline.rollback_to(packet.rollback_token)

    assert result["removed"] == 1
    assert pipeline.store.stats()["approved"] == 0
    assert pipeline.store.approved_skills("qwen-mock") == []
    assert any(e["event"] == "rollback" for e in pipeline.audit.entries())
    assert pipeline.audit.verify()["ok"] is True


def test_full_run_under_default_strict_policy_rejects_mock_packets(
    tmp_path, as_en_suite, hi_en_suite
):
    """The demos relax this. Verify the shipped default actually protects."""
    sender = make_generic_sender(
        module_id="src", capabilities=[as_en_suite.capability()],
        knowledge=knowledge([as_en_suite]),
    )
    receiver = make_qwen(knowledge=None, fallback="echo")
    pipeline = Pipeline(workspace=tmp_path / "strict")  # default policy
    pipeline.register_module(sender)
    pipeline.register_module(receiver)
    pipeline.bind_adapter("a", "src", "qwen-mock")

    report = pipeline.run("a", suites=[as_en_suite, hi_en_suite])
    assert report.promoted == []
    assert report.rejected, "a mock-derived packet must be rejected by default"
    assert "mock" in report.decisions[0]["reason"]
    failed = [c["name"] for c in report.decisions[0]["checks"] if not c["passed"]]
    assert "no_mock_provenance" in failed


def test_learning_level_is_capped_by_the_weaker_party(assamese_setup):
    pipeline, suites = assamese_setup
    report = pipeline.run(
        "as-to-qwen", suites=suites, requested_level=LearningLevel.L5_DISTILL_DATASET
    )
    assert report.session.negotiated_level == LearningLevel.L3_SKILL_PACKET
    for packet in report.distilled:
        assert packet.learning_level == LearningLevel.L3_SKILL_PACKET
