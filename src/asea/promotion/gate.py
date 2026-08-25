"""Promotion gate.

Every rule is an explicit, individually-reported check. A packet is promoted
only if *all* of them pass; there is no aggregate score that can drown out a
single hard failure, and no code path that promotes without going through
:meth:`PromotionGate.decide`.

Hard rules (cannot be relaxed by configuration):
  * HIGH-risk domains (medical, legal, finance) require a named human approver.
  * A packet whose learning level exceeds what this codebase can apply (L4/L5)
    is never promoted to a live receiver -- it is exported instead.
  * Provenance chain and rollback token must exist.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core.errors import PromotionBlocked
from ..core.protocol import (
    APPLICABLE_LEVELS,
    PromotionStatus,
    RiskTier,
    SkillPacket,
)


class PromotionPolicy:
    def __init__(
        self,
        min_evaluator_score: float = 0.6,
        min_safety_score: float = 0.7,
        min_schema_compliance: float = 1.0,
        min_improvement: float = 0.01,
        max_case_regression_ratio: float = 1.0,
        max_synthetic_depth: int = 2,
        strict_no_mock: bool = True,
        require_rollback_token: bool = True,
    ) -> None:
        self.min_evaluator_score = min_evaluator_score
        self.min_safety_score = min_safety_score
        self.min_schema_compliance = min_schema_compliance
        self.min_improvement = min_improvement
        #: Fraction of held-out cases allowed to get WORSE even when the average
        #: improves. Default 1.0 keeps the aggregate-only behaviour; set it lower
        #: (e.g. 0.2) to refuse packets that break cases which already worked.
        #: Observed with real models: an aggregate +0.05 hid a single case going
        #: from correct to wrong. See docs/real_run_findings.md.
        self.max_case_regression_ratio = max_case_regression_ratio
        #: Beyond this many model-generated generations from verified data, the
        #: packet is refused. This is the model-collapse brake.
        self.max_synthetic_depth = max_synthetic_depth
        #: When True, any packet touched by a mock module cannot be promoted.
        self.strict_no_mock = strict_no_mock
        self.require_rollback_token = require_rollback_token


class Check:
    def __init__(self, name: str, passed: bool, detail: str, hard: bool = False):
        self.name = name
        self.passed = passed
        self.detail = detail
        self.hard = hard

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "hard": self.hard,
        }


class GateDecision:
    def __init__(
        self,
        packet_id: str,
        checks: List[Check],
        needs_human: bool,
        status: PromotionStatus,
    ) -> None:
        self.packet_id = packet_id
        self.checks = checks
        self.needs_human = needs_human
        self.status = status

    @property
    def approved(self) -> bool:
        return self.status == PromotionStatus.PROMOTED

    @property
    def failures(self) -> List[Check]:
        return [c for c in self.checks if not c.passed]

    def reason(self) -> str:
        if self.approved:
            return "all checks passed"
        if self.needs_human and not self.failures:
            return "awaiting human approval (high-risk domain)"
        return "; ".join("{}: {}".format(c.name, c.detail) for c in self.failures)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "packet_id": self.packet_id,
            "status": self.status.value,
            "approved": self.approved,
            "needs_human": self.needs_human,
            "reason": self.reason(),
            "checks": [c.to_dict() for c in self.checks],
        }


class PromotionGate:
    def __init__(self, policy: Optional[PromotionPolicy] = None) -> None:
        self.policy = policy or PromotionPolicy()

    def decide(
        self, packet: SkillPacket, human_approver: Optional[str] = None
    ) -> GateDecision:
        p = self.policy
        checks: List[Check] = []

        # 1. structural validity
        scores = packet.scores
        schema = scores.schema_compliance if scores else 0.0
        checks.append(
            Check(
                "schema_validation",
                schema >= p.min_schema_compliance,
                "schema_compliance {:.2f} (need >= {:.2f})".format(
                    schema, p.min_schema_compliance
                ),
                hard=True,
            )
        )
        checks.append(
            Check(
                "distilled_payload_present",
                bool(packet.distilled_skill),
                "distilled_skill is {}".format(
                    "present" if packet.distilled_skill else "missing"
                ),
                hard=True,
            )
        )

        # 2. evaluator threshold
        ev = packet.evaluator_score
        checks.append(
            Check(
                "evaluator_threshold",
                ev is not None and ev >= p.min_evaluator_score,
                "evaluator_score {} (need >= {:.2f})".format(
                    "unset" if ev is None else "{:.3f}".format(ev),
                    p.min_evaluator_score,
                ),
            )
        )

        # 3. safety threshold
        sa = packet.safety_score
        checks.append(
            Check(
                "safety_threshold",
                sa is not None and sa >= p.min_safety_score,
                "safety_score {} (need >= {:.2f})".format(
                    "unset" if sa is None else "{:.3f}".format(sa), p.min_safety_score
                ),
                hard=True,
            )
        )

        # 4. measured benefit, and no collateral damage
        improvement = scores.improvement if scores else None
        checks.append(
            Check(
                "benchmark_improvement",
                improvement is not None and improvement >= p.min_improvement,
                "improvement {} (need >= {:.3f})".format(
                    "unmeasured" if improvement is None else "{:+.4f}".format(improvement),
                    p.min_improvement,
                ),
            )
        )
        checks.append(
            Check(
                "no_regression",
                bool(scores and not scores.regression_detected),
                (scores.regression_detail if scores and scores.regression_detail
                 else "no regression detected"),
                hard=True,
            )
        )

        ratio = scores.case_regression_ratio if scores else 0.0
        checks.append(
            Check(
                "case_regression_limit",
                ratio <= p.max_case_regression_ratio,
                "{}/{} held-out cases regressed (ratio {:.2f}, max {:.2f})".format(
                    scores.case_regression_count if scores else 0,
                    scores.case_count if scores else 0,
                    ratio,
                    p.max_case_regression_ratio,
                ),
                hard=True,
            )
        )
        # Control-movement bound (audit 2026-08-17): the sibling hard check of
        # no_regression. ``no_regression`` only fails on a control suite that
        # DROPPED; this fails on a control suite that MOVED in either direction
        # (|delta| > evaluator.max_control_movement, default 0.05). A packet
        # that lifts a non-targeted capability -- bleeding into it via prompt
        # conditioning -- is a quiet capability change that the drop-only check
        # could not see. Symmetric to the Gate 2 bound; closes the half of the
        # gate that was still drop-only. The bound is the evaluator's knob (like
        # regression_tolerance), not a policy knob -- the gate reads the bool.
        checks.append(
            Check(
                "no_control_movement",
                bool(scores and not scores.control_movement_detected),
                (scores.control_movement_detail if scores and scores.control_movement_detail
                 else "no control-suite movement detected"),
                hard=True,
            )
        )
        # SPRT early-stop bound (B2, audit 2026-08-17): the statistical sibling
        # of no_control_movement. ``scores.sprt`` is populated ONLY when the
        # candidate held-out run was stopped early by the SPRT -- which, by the
        # SPRT's asymmetry, can ONLY ever be a REJECT (early-PROMOTE is
        # forbidden; ``SPRT.should_stop`` is True only on REJECT). A stopped
        # candidate run is PARTIAL, so its aggregate is over a short, optimistic
        # sample -- it must NEVER be trusted to promote. This HARD check fails
        # the packet the moment a statistical early-reject is on record,
        # regardless of the partial aggregate, so a clearly-failing packet is
        # rejected after a handful of cases instead of grinding through the full
        # held-out set. ``None`` (SPRT disabled OR run completed without early
        # stop) passes -- byte-identical to pre-B2.
        #
        # The DECISION is fail-closed: any non-None ``sprt`` (including a
        # malformed/partial dict deserialised from on-disk JSON) fails the
        # check. The DETAIL string is formatted defensively so a malformed record
        # cannot raise out of ``decide()`` before the packet is stamped REJECTED
        # -- a fail-closed check must never become fail-open by crashing
        # (adversarial audit 2026-08-17, finding 1).
        sprt_record = scores.sprt if scores else None
        sprt_failed = sprt_record is not None
        if sprt_failed:
            try:
                sprt_detail = "SPRT early-REJECT after {}/{} cases (verdict={}, llr={:.4f})".format(
                    sprt_record.get("cases_evaluated", "?"),
                    # The held-out set size is not stored on the record; the gate
                    # reports cases actually scored vs the candidate count the
                    # evaluator did record, which is the partial run length on an
                    # early stop.
                    scores.case_count if scores else 0,
                    sprt_record.get("verdict", "?"),
                    float(sprt_record.get("llr", 0.0) or 0.0),
                )
            except Exception:
                sprt_detail = "SPRT early-reject record present but malformed: {}".format(
                    sprt_record
                )
        else:
            sprt_detail = "no SPRT early-reject (SPRT disabled or run completed)"
        checks.append(
            Check("no_statistical_early_reject", not sprt_failed, sprt_detail, hard=True)
        )

        # 5. provenance
        prov = packet.provenance
        checks.append(
            Check(
                "provenance_present",
                bool(prov and prov.chain),
                "chain={}".format(prov.chain if prov else None),
                hard=True,
            )
        )
        checks.append(
            Check(
                "synthetic_depth",
                prov.synthetic_depth <= p.max_synthetic_depth,
                "synthetic_depth {} (max {})".format(
                    prov.synthetic_depth, p.max_synthetic_depth
                ),
                hard=True,
            )
        )
        checks.append(
            Check(
                "no_self_transfer",
                packet.target_module not in prov.chain,
                "receiver '{}' {} in provenance chain".format(
                    packet.target_module,
                    "appears" if packet.target_module in prov.chain else "absent",
                ),
                hard=True,
            )
        )

        # 6. rollback metadata
        if p.require_rollback_token:
            checks.append(
                Check(
                    "rollback_metadata",
                    bool(packet.rollback_token),
                    "rollback_token {}".format(
                        "present" if packet.rollback_token else "missing"
                    ),
                    hard=True,
                )
            )

        # 7. learning level applicability
        checks.append(
            Check(
                "applicable_learning_level",
                packet.learning_level in APPLICABLE_LEVELS,
                "level L{} {}".format(
                    int(packet.learning_level),
                    "is applicable"
                    if packet.learning_level in APPLICABLE_LEVELS
                    else "is export-only (no training performed by this system)",
                ),
                hard=True,
            )
        )

        # 8. mock containment
        if p.strict_no_mock:
            checks.append(
                Check(
                    "no_mock_provenance",
                    not prov.is_mock,
                    "provenance {} a mock module".format(
                        "includes" if prov.is_mock else "excludes"
                    ),
                    hard=True,
                )
            )

        # 9. human approval for high-risk domains -- not configurable
        needs_human = packet.risk_tier == RiskTier.HIGH
        if needs_human:
            checks.append(
                Check(
                    "human_approval",
                    bool(human_approver),
                    "domain '{}' is {} risk; approver={}".format(
                        packet.domain.value, RiskTier.HIGH.value, human_approver or "none"
                    ),
                    hard=True,
                )
            )

        all_passed = all(c.passed for c in checks)
        if all_passed:
            status = PromotionStatus.PROMOTED
        elif needs_human and all(c.passed for c in checks if c.name != "human_approval"):
            status = PromotionStatus.PENDING_HUMAN
        else:
            status = PromotionStatus.REJECTED

        return GateDecision(packet.packet_id, checks, needs_human, status)

    def apply(
        self,
        packet: SkillPacket,
        rollback_token: Optional[str] = None,
        human_approver: Optional[str] = None,
    ) -> GateDecision:
        """Stamp the packet with the gate's decision.

        Raises PromotionBlocked if a caller tries to force a promotion that the
        gate refused -- there is intentionally no bypass argument.
        """
        if rollback_token:
            packet.rollback_token = rollback_token
        if human_approver:
            packet.human_approved_by = human_approver

        decision = self.decide(packet, human_approver=human_approver)

        if decision.status == PromotionStatus.REJECTED:
            packet.rejection_reason = decision.reason()
        packet.promotion_status = decision.status
        return decision

    @staticmethod
    def enforce(decision: GateDecision) -> None:
        if not decision.approved:
            raise PromotionBlocked(
                "packet {} not promotable: {}".format(
                    decision.packet_id, decision.reason()
                )
            )
