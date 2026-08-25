"""Gate 2 -- the second gate, applied to a TRAINED adapter.

Reuses the gate *machinery* (``Check``, ``GateDecision``, ``PromotionPolicy``,
the all-or-nothing ``all(c.passed)`` semantics, and the non-disableable
human-approval logic) from :mod:`asea.promotion.gate`. It does NOT reuse
``PromotionGate`` itself: an adapter is not a SkillPacket, and several checks
have adapter-specific analogues (depth propagates from sources; risk is the
max-severity source domain; ``no_self_lineage`` checks the receiver is not the
teacher of its own training data). See ``docs/deep_apply_design.md`` §7 for the
full check mapping.

Binding rules (same as the packet gate):

* All-or-nothing: ``PROMOTED`` only if every check passes.
* HIGH-risk domains (medical/legal/finance) require a NAMED human approver. If
  ANY source packet is high-risk, the adapter is HIGH risk and parks at
  ``PENDING_HUMAN`` regardless of scores. No policy knob can disable this.
* A packet gate check is never weakened here; Gate 2 is additive and
  independent of Gate 1 (it uses its OWN policy thresholds).
"""

from __future__ import annotations

import math
from typing import List, Optional

from ..core.protocol import (
    LearningLevel,
    PromotionStatus,
    RiskTier,
)
from ..promotion.gate import Check, GateDecision, PromotionPolicy
from .adapter_packet import AdapterPacket


class DeepApplyPolicy(PromotionPolicy):
    """Gate 2 policy. Same thresholds as ``PromotionPolicy`` plus an
    adapter-size sanity floor (``min_trainable_params``) so a degenerate
    zero-parameter adapter cannot be admitted. Defaults preserve the existing
    bar; the floor defaults to 1 (any real LoRA has >0 trainable params)."""

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
        min_trainable_params: int = 1,
        max_control_movement: float = 0.05,
    ) -> None:
        super().__init__(
            min_evaluator_score=min_evaluator_score,
            min_safety_score=min_safety_score,
            min_schema_compliance=min_schema_compliance,
            min_improvement=min_improvement,
            max_case_regression_ratio=max_case_regression_ratio,
            max_synthetic_depth=max_synthetic_depth,
            strict_no_mock=strict_no_mock,
            require_rollback_token=require_rollback_token,
        )
        #: Adapter-size sanity floor. A 0-trainable-param "adapter" trained
        #: nothing and must not be admitted however good the held-out looks.
        self.min_trainable_params = min_trainable_params
        #: Control-movement bound (audit 2026-08-17). The maximum |delta| a
        #: CONTROL (non-target) suite may move in EITHER direction before the
        #: adapter is blocked for touching a capability it was not targeting.
        #: A favorable control shift (training bleeding into a non-target
        #: capability) is NOT benign -- it signals untargeted drift and must
        #: block just as a regression does. ``regression_tolerance`` only
        #: catches drops; this bound catches movement. Gate-strengthening.
        self.max_control_movement = max_control_movement


def _is_finite(x: float) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


class DeepApplyGate:
    def __init__(self, policy: Optional[DeepApplyPolicy] = None) -> None:
        self.policy = policy or DeepApplyPolicy()

    def decide(
        self, adapter: AdapterPacket, human_approver: Optional[str] = None
    ) -> GateDecision:
        p = self.policy
        checks: List[Check] = []
        scores = adapter.scores

        # 1. structural validity (adapter schema + finite training loss)
        schema_ok = bool(adapter.adapter_id) and bool(adapter.base_model) and bool(adapter.target_module)
        checks.append(
            Check(
                "schema_validation",
                schema_ok,
                "adapter identity {}".format("present" if schema_ok else "incomplete"),
                hard=True,
            )
        )
        # NEW check: training-loss divergence guard. This is a degenerate-artifact
        # sanity check (did training numerically blow up?), NOT a quality
        # endorsement -- it does not trust the trainer's claim that training
        # *worked*, only that it did not produce NaN/Inf.
        checks.append(
            Check(
                "training_loss_finite",
                _is_finite(adapter.training_loss),
                "training_loss {}".format(
                    "finite" if _is_finite(adapter.training_loss) else "NaN/Inf (diverged)"
                ),
                hard=True,
            )
        )

        # 2. adapter artifact present + non-degenerate size
        artifact_ok = (
            adapter.adapter_artifact_ref is not None
            and adapter.trainable_param_count >= p.min_trainable_params
        )
        checks.append(
            Check(
                "adapter_artifact_present",
                artifact_ok,
                "artifact_ref={}, trainable_params={} (need >= {})".format(
                    "present" if adapter.adapter_artifact_ref else "missing",
                    adapter.trainable_param_count,
                    p.min_trainable_params,
                ),
                hard=True,
            )
        )

        # 3. evaluator threshold
        ev = adapter.evaluator_score
        checks.append(
            Check(
                "evaluator_threshold",
                ev is not None and ev >= p.min_evaluator_score,
                "evaluator_score {} (need >= {:.2f})".format(
                    "unset" if ev is None else "{:.3f}".format(ev), p.min_evaluator_score
                ),
            )
        )

        # 4. safety threshold (inherited min over source packets)
        sa = adapter.safety_score
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

        # 5. measured benefit + no collateral damage
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
        # Control-movement bound (audit 2026-08-17): a CONTROL (non-target)
        # suite that moved |delta| > max_control_movement in EITHER direction
        # blocks the adapter. ``no_regression`` only sees score drops; a
        # control suite that IMPROVED meant training bled into a capability it
        # was not targeting, and that was invisible to the gate until now.
        checks.append(
            Check(
                "no_control_movement",
                bool(scores and not scores.control_movement_detected),
                (scores.control_movement_detail if scores and scores.control_movement_detail
                 else "no control-suite movement detected"),
                hard=True,
            )
        )

        # 6. provenance: merged source chain + source packet ids
        prov = adapter.provenance
        checks.append(
            Check(
                "provenance_present",
                bool(prov and prov.chain) and bool(adapter.source_packet_ids),
                "source_packets={}, chain={}".format(
                    len(adapter.source_packet_ids), prov.chain if prov else None
                ),
                hard=True,
            )
        )
        checks.append(
            Check(
                "synthetic_depth",
                adapter.synthetic_depth <= p.max_synthetic_depth,
                "synthetic_depth {} (max {}) [propagated max over {} source packets]".format(
                    adapter.synthetic_depth, p.max_synthetic_depth, len(adapter.source_packet_ids)
                ),
                hard=True,
            )
        )
        # no_self_lineage: the receiver must not be the teacher of its own
        # training data -- its module_id must not appear in any source chain.
        receiver_in_chain = adapter.target_module in (prov.chain if prov else [])
        checks.append(
            Check(
                "no_self_lineage",
                not receiver_in_chain,
                "receiver '{}' {} in source provenance chain".format(
                    adapter.target_module, "appears" if receiver_in_chain else "absent"
                ),
                hard=True,
            )
        )

        # 7. rollback metadata
        if p.require_rollback_token:
            checks.append(
                Check(
                    "rollback_metadata",
                    bool(adapter.rollback_token),
                    "rollback_token {}".format(
                        "present" if adapter.rollback_token else "missing"
                    ),
                    hard=True,
                )
            )

        # 8. learning level applicability -- deep-apply admits L4 only; L5 stays export-only
        checks.append(
            Check(
                "applicable_learning_level",
                adapter.learning_level == LearningLevel.L4_PEFT_CANDIDATE,
                "level L{} {}".format(
                    int(adapter.learning_level),
                    "is applicable (deep-apply)"
                    if adapter.learning_level == LearningLevel.L4_PEFT_CANDIDATE
                    else "is export-only (deep-apply trains L4 PEFT only)",
                ),
                hard=True,
            )
        )

        # 9. mock containment -- no source packet may be mock-derived
        if p.strict_no_mock:
            checks.append(
                Check(
                    "no_mock_provenance",
                    not prov.is_mock,
                    "source provenance {} a mock module".format(
                        "includes" if prov.is_mock else "excludes"
                    ),
                    hard=True,
                )
            )

        # 10. human approval for high-risk adapters -- not configurable
        # Any high-risk source packet => adapter is HIGH risk => PENDING_HUMAN.
        needs_human = adapter.risk_tier == RiskTier.HIGH
        if needs_human:
            checks.append(
                Check(
                    "human_approval",
                    bool(human_approver),
                    "adapter risk_tier={} (max over source domains {}); approver={}".format(
                        RiskTier.HIGH.value,
                        [d.value for d in adapter.source_domains],
                        human_approver or "none",
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

        return GateDecision(adapter.adapter_id, checks, needs_human, status)

    def apply(
        self,
        adapter: AdapterPacket,
        rollback_token: Optional[str] = None,
        human_approver: Optional[str] = None,
    ) -> GateDecision:
        """Stamp the adapter with Gate 2's decision. No bypass argument."""
        if rollback_token:
            adapter.rollback_token = rollback_token
        if human_approver:
            adapter.human_approved_by = human_approver

        decision = self.decide(adapter, human_approver=human_approver)
        if decision.status == PromotionStatus.REJECTED:
            adapter.rejection_reason = decision.reason()
        adapter.promotion_status = decision.status
        return decision

    @staticmethod
    def enforce(decision: GateDecision) -> None:
        from ..core.errors import PromotionBlocked
        if not decision.approved:
            raise PromotionBlocked(
                "adapter {} not admittable: {}".format(
                    decision.packet_id, decision.reason()
                )
            )