"""Promotion gate. Every rule gets its own test, including the ones that must
never be relaxed."""

from __future__ import annotations

import pytest

from asea.core.errors import PromotionBlocked
from asea.core.protocol import (
    Domain,
    EvaluationScores,
    LearningLevel,
    OriginKind,
    PacketType,
    PromotionStatus,
    Provenance,
)
from asea.promotion.gate import PromotionGate, PromotionPolicy


def good_scores(**overrides) -> EvaluationScores:
    payload = dict(
        schema_compliance=1.0,
        semantic_similarity=0.9,
        task_success=0.9,
        language_preservation=1.0,
        hallucination_risk=0.05,
        aggregate=0.9,
        baseline_score=0.30,
        candidate_score=0.80,
        regression_detected=False,
    )
    payload.update(overrides)
    return EvaluationScores(**payload)


@pytest.fixture
def promotable(capability, clean_provenance, packet_factory):
    """A packet that passes every check. Tests below break one thing at a time."""
    packet = packet_factory(
        capability, clean_provenance,
        packet_type=PacketType.GLOSSARY,
        distilled_skill={"entries": [{"source": "ভাত", "target": "rice"}]},
        confidence_score=0.9,
        safety_score=1.0,
        evaluator_score=0.9,
        scores=good_scores(),
        learning_level=LearningLevel.L3_SKILL_PACKET,
        promotion_status=PromotionStatus.EVALUATED,
    )
    return packet


def test_clean_packet_is_promoted(promotable):
    decision = PromotionGate().apply(promotable, rollback_token="snap-1")
    assert decision.approved is True
    assert promotable.promotion_status == PromotionStatus.PROMOTED
    assert decision.reason() == "all checks passed"


def test_missing_rollback_token_blocks(promotable):
    decision = PromotionGate().apply(promotable)
    assert not decision.approved
    assert any(c.name == "rollback_metadata" for c in decision.failures)


def test_low_evaluator_score_blocks(promotable):
    promotable.evaluator_score = 0.2
    decision = PromotionGate().apply(promotable, rollback_token="s")
    assert any(c.name == "evaluator_threshold" for c in decision.failures)


def test_low_safety_score_blocks(promotable):
    promotable.safety_score = 0.1
    decision = PromotionGate().apply(promotable, rollback_token="s")
    assert any(c.name == "safety_threshold" for c in decision.failures)


def test_schema_failure_blocks(promotable):
    promotable.scores = good_scores(schema_compliance=0.5)
    decision = PromotionGate().apply(promotable, rollback_token="s")
    assert any(c.name == "schema_validation" for c in decision.failures)


def test_no_improvement_blocks(promotable):
    promotable.scores = good_scores(baseline_score=0.80, candidate_score=0.80)
    decision = PromotionGate().apply(promotable, rollback_token="s")
    assert any(c.name == "benchmark_improvement" for c in decision.failures)


def test_regression_blocks_even_with_improvement(promotable):
    """A packet that helps here and hurts there is not an improvement."""
    promotable.scores = good_scores(
        regression_detected=True, regression_detail="hindi: 0.9 -> 0.6"
    )
    decision = PromotionGate().apply(promotable, rollback_token="s")
    assert not decision.approved
    assert any(c.name == "no_regression" for c in decision.failures)


def test_control_movement_blocks_even_when_it_improves_a_control(promotable):
    """The sibling of test_regression_blocks_even_with_improvement (audit
    2026-08-17). ``no_regression`` only sees a control suite that DROPPED; this
    proves ``no_control_movement`` catches the symmetric case -- a control suite
    that IMPROVED (|delta| > bound), i.e. the packet bled into a capability it
    was not targeting. The scores carry no regression (the control did not
    drop), so ``no_regression`` must PASS while ``no_control_movement`` FAILS --
    the two checks are distinct and the bound fires on movement, not on drops."""
    promotable.scores = good_scores(
        control_movement_detected=True,
        control_movement_detail="hindi: 0.50 -> 0.70 (|delta| +0.20 > 0.05)",
    )
    decision = PromotionGate().apply(promotable, rollback_token="s")
    assert not decision.approved
    failed_names = {c.name for c in decision.failures}
    assert "no_control_movement" in failed_names
    assert "no_regression" not in failed_names


def test_missing_provenance_blocks(promotable):
    promotable.provenance = Provenance(origin_kind=OriginKind.CURATED_CORPUS, chain=[])
    decision = PromotionGate().apply(promotable, rollback_token="s")
    assert any(c.name == "provenance_present" for c in decision.failures)


def test_excessive_synthetic_depth_blocks(promotable):
    """The model-collapse brake."""
    promotable.provenance = Provenance(
        origin_kind=OriginKind.MODEL_GENERATED, chain=["a", "b", "c"], synthetic_depth=5
    )
    decision = PromotionGate().apply(promotable, rollback_token="s")
    assert any(c.name == "synthetic_depth" for c in decision.failures)


def test_self_transfer_via_provenance_blocks(promotable):
    """Receiver appears in its own teaching chain: refuse."""
    promotable.provenance = Provenance(
        origin_kind=OriginKind.MODEL_GENERATED,
        chain=["source", promotable.target_module],
        synthetic_depth=1,
    )
    decision = PromotionGate().apply(promotable, rollback_token="s")
    assert any(c.name == "no_self_transfer" for c in decision.failures)


def test_mock_provenance_blocked_under_default_strict_policy(promotable):
    """The demo scripts disable this. The DEFAULT must enforce it."""
    promotable.provenance = Provenance(
        origin_kind=OriginKind.MODEL_GENERATED, chain=["qwen-mock"], is_mock=True
    )
    decision = PromotionGate().apply(promotable, rollback_token="s")
    assert not decision.approved
    assert any(c.name == "no_mock_provenance" for c in decision.failures)

    relaxed = PromotionGate(PromotionPolicy(strict_no_mock=False))
    assert relaxed.apply(promotable, rollback_token="s").approved is True


@pytest.mark.parametrize(
    "level", [LearningLevel.L4_PEFT_CANDIDATE, LearningLevel.L5_DISTILL_DATASET]
)
def test_training_levels_cannot_be_promoted_to_a_live_receiver(promotable, level):
    promotable.learning_level = level
    decision = PromotionGate().apply(promotable, rollback_token="s")
    assert not decision.approved
    assert any(c.name == "applicable_learning_level" for c in decision.failures)


# -- the rules that are not configurable -----------------------------------


@pytest.mark.parametrize("domain", [Domain.MEDICAL, Domain.LEGAL, Domain.FINANCE])
def test_high_risk_domains_never_auto_promote(promotable, domain):
    promotable.domain = domain
    decision = PromotionGate().apply(promotable, rollback_token="s")
    assert decision.status == PromotionStatus.PENDING_HUMAN
    assert decision.approved is False
    assert decision.needs_human is True


def test_high_risk_promotes_only_with_named_approver(promotable):
    promotable.domain = Domain.MEDICAL
    decision = PromotionGate().apply(
        promotable, rollback_token="s", human_approver="dr.reviewer@example.org"
    )
    assert decision.approved is True
    assert promotable.human_approved_by == "dr.reviewer@example.org"


def test_human_approval_does_not_waive_other_checks(promotable):
    """An approver can satisfy the approval check and nothing else."""
    promotable.domain = Domain.MEDICAL
    promotable.safety_score = 0.0
    decision = PromotionGate().apply(
        promotable, rollback_token="s", human_approver="dr.reviewer@example.org"
    )
    assert not decision.approved
    assert any(c.name == "safety_threshold" for c in decision.failures)


def test_no_configuration_can_disable_human_approval(promotable):
    """Even a maximally permissive policy cannot auto-promote medical."""
    promotable.domain = Domain.MEDICAL
    reckless = PromotionGate(
        PromotionPolicy(
            min_evaluator_score=0.0, min_safety_score=0.0, min_schema_compliance=0.0,
            min_improvement=-1.0, max_synthetic_depth=99, strict_no_mock=False,
            require_rollback_token=False,
        )
    )
    decision = reckless.apply(promotable)
    assert decision.status == PromotionStatus.PENDING_HUMAN


def test_enforce_raises_on_blocked_packet(promotable):
    promotable.domain = Domain.MEDICAL
    decision = PromotionGate().apply(promotable, rollback_token="s")
    with pytest.raises(PromotionBlocked):
        PromotionGate.enforce(decision)


def test_rejected_packet_records_a_reason(promotable):
    promotable.evaluator_score = 0.1
    PromotionGate().apply(promotable, rollback_token="s")
    assert promotable.promotion_status == PromotionStatus.REJECTED
    assert "evaluator_threshold" in promotable.rejection_reason


def test_decision_serialises_every_check(promotable):
    """Equality over the FULL check set (adversarial audit 2026-08-13 #18: the
    old ``<=`` subset assertion omitted ``distilled_payload_present`` and
    ``case_regression_limit``, so deleting either check from the gate would
    leave the suite green -- a false-green completeness claim). The promotable
    fixture is a LOW-risk translation packet, so ``human_approval`` does not
    fire; the gate therefore emits exactly these 15 checks (the control-movement
    bound was added 2026-08-17 as a sibling hard check to no_regression; the
    SPRT early-reject bound was added 2026-08-17 as the statistical sibling of
    no_control_movement -- passes here because the fixture has no SPRT record)."""
    decision = PromotionGate().apply(promotable, rollback_token="s")
    payload = decision.to_dict()
    names = {c["name"] for c in payload["checks"]}
    assert names == {
        "schema_validation", "distilled_payload_present", "evaluator_threshold",
        "safety_threshold", "benchmark_improvement", "no_regression",
        "case_regression_limit", "no_control_movement",
        "no_statistical_early_reject",
        "provenance_present",
        "synthetic_depth", "no_self_transfer", "rollback_metadata",
        "applicable_learning_level", "no_mock_provenance",
    }


def test_distilled_payload_present_blocks_a_payload_less_packet(promotable):
    """A packet with good scores but no distilled payload is blocked ONLY by
    ``distilled_payload_present`` (audit #18: this failing path had no test, so
    removing the check would silently let empty-content packets promote)."""
    promotable.distilled_skill = None
    decision = PromotionGate().apply(promotable, rollback_token="s")
    assert decision.approved is False
    failed = {c["name"] for c in decision.to_dict()["checks"] if not c["passed"]}
    assert "distilled_payload_present" in failed
    assert promotable.promotion_status == PromotionStatus.REJECTED

    # An empty dict is equally payload-less.
    promotable.distilled_skill = {}
    promotable.promotion_status = PromotionStatus.EVALUATED
    promotable.rejection_reason = None
    decision2 = PromotionGate().apply(promotable, rollback_token="s")
    assert decision2.approved is False
    assert "distilled_payload_present" in {
        c["name"] for c in decision2.to_dict()["checks"] if not c["passed"]
    }


# -- per-case regression (added after a real-model run exposed the gap) -------


def test_case_regression_is_permitted_by_default(promotable):
    """Default policy is aggregate-only, preserving prior behaviour."""
    promotable.scores = good_scores(case_count=6, case_regression_count=2)
    assert PromotionGate().apply(promotable, rollback_token="s").approved is True


def test_case_regression_limit_blocks_when_configured(promotable):
    """An aggregate gain must not hide a case that used to work.

    Observed with real weights: overall +0.053 while one held-out case went from
    a correct answer to a wrong one. See docs/real_run_findings.md.
    """
    promotable.scores = good_scores(case_count=6, case_regression_count=2)
    strict = PromotionGate(PromotionPolicy(max_case_regression_ratio=0.2))
    decision = strict.apply(promotable, rollback_token="s")
    assert not decision.approved
    assert any(c.name == "case_regression_limit" for c in decision.failures)


def test_case_regression_limit_passes_when_clean(promotable):
    promotable.scores = good_scores(case_count=6, case_regression_count=0)
    strict = PromotionGate(PromotionPolicy(max_case_regression_ratio=0.2))
    assert strict.apply(promotable, rollback_token="s").approved is True


def test_case_regression_ratio_handles_no_case_data(promotable):
    promotable.scores = good_scores(case_count=0, case_regression_count=0)
    assert promotable.scores.case_regression_ratio == 0.0
    strict = PromotionGate(PromotionPolicy(max_case_regression_ratio=0.0))
    assert strict.apply(promotable, rollback_token="s").approved is True
