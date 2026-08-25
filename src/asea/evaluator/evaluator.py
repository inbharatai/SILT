"""Evaluator: does this packet actually make the receiver better?

The method is a before/after A/B on the **held-out** split:

    baseline  = receiver alone
    candidate = receiver + this packet's distilled_skill

plus a regression sweep over suites the transfer is *not* targeting. A packet
that lifts its own capability while damaging another is rejected -- that
trade-off is exactly how quiet capability loss creeps into a system.

What this measures honestly: whether conditioning the receiver on the packet
changes its held-out outputs for the better, under a lexical proxy metric. What
it does not measure: real-world adequacy, native-speaker acceptability, or
correctness beyond the reference strings supplied in the suite.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..benchmarks.harness import BenchmarkHarness, BenchmarkSuite, SuiteResult
from ..core.errors import EvaluationError
from ..core.interfaces import ModuleAdapter
from ..core.protocol import EvaluationScores, PromotionStatus, SkillPacket
from ..sprt import SPRT, SprtConfig
from . import metrics


class EvaluationReport:
    def __init__(
        self,
        packet: SkillPacket,
        baseline: SuiteResult,
        candidate: SuiteResult,
        regressions: List[Dict[str, Any]],
        scores: EvaluationScores,
    ) -> None:
        self.packet = packet
        self.baseline = baseline
        self.candidate = candidate
        self.regressions = regressions
        self.scores = scores

    @property
    def improvement(self) -> float:
        return self.candidate.score - self.baseline.score

    def case_diff(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Per-case before/after rows.

        This is the most useful artefact the evaluator produces: an aggregate
        delta tells you something moved, but only the per-case diff tells you
        whether a packet fixed the cases you cared about or quietly broke ones
        that already worked. Included in the audit record for that reason.
        """
        by_id = {c.case_id: c for c in self.candidate.case_results}
        rows = []
        for base in self.baseline.case_results[:limit]:
            cand = by_id.get(base.case_id)
            if cand is None:
                continue
            rows.append(
                {
                    "case_id": base.case_id,
                    "expected": base.expected,
                    "baseline_output": base.actual,
                    "candidate_output": cand.actual,
                    "baseline_score": round(base.score, 4),
                    "candidate_score": round(cand.score, 4),
                    "delta": round(cand.score - base.score, 4),
                    "regressed": cand.score < base.score,
                }
            )
        return rows

    def to_dict(self) -> Dict[str, Any]:
        return {
            "packet_id": self.packet.packet_id,
            "capability": self.packet.sender_capability.as_str(),
            "baseline": self.baseline.summary(),
            "candidate": self.candidate.summary(),
            "case_diff": self.case_diff(),
            "improvement": round(self.improvement, 4),
            "regressions": self.regressions,
            "scores": self.scores.model_dump(),
            "similarity_is_semantic": self.candidate.similarity_is_semantic,
            "caveat": (
                "Scores use a lexical similarity proxy, not semantic embedding. "
                "Directional only; not a substitute for native-speaker or expert review."
            ),
        }


class Evaluator:
    def __init__(
        self,
        harness: Optional[BenchmarkHarness] = None,
        regression_tolerance: float = 0.02,
        max_control_movement: float = 0.05,
        sprt: Optional[SprtConfig] = None,
    ) -> None:
        self.harness = harness or BenchmarkHarness()
        #: How much a non-targeted capability may drop before we call it a
        #: regression. Small but non-zero to absorb metric noise.
        self.regression_tolerance = regression_tolerance
        #: Control-movement bound (Gate 1, audit 2026-08-17). ``regressed`` above
        #: only catches DROPS on a non-target (control) suite; this bound catches
        #: movement in EITHER direction (|delta|). A control suite that IMPROVED
        #: under packet injection -- the packet bleeding into a capability it was
        #: not targeting, e.g. via prompt conditioning -- is no longer invisible
        #: to Gate 1. Mirrors the symmetric bound added to Gate 2
        #: (:class:`asea.deepapply.evaluator.DeepApplyEvaluator`) so a packet can
        #: no longer slip through the half of the gate that was still drop-only.
        self.max_control_movement = max_control_movement
        #: SPRT early-stop config (B2, audit 2026-08-17). ``None`` (the default)
        #: disables statistical early-stop and makes evaluate() byte-identical to
        #: pre-B2. When set, the candidate held-out run is stopped the moment the
        #: SPRT reaches its REJECT boundary at 95% confidence -- an ASYMMETRIC
        #: test: early-REJECT is allowed (stop and fail a clearly-failing packet
        #: after a handful of cases), early-PROMOTE is FORBIDDEN (a good packet
        #: runs the FULL held-out set, so the regular gate still sees every
        #: case). The early stop is recorded in ``scores.sprt`` and a new HARD
        #: gate check ``no_statistical_early_reject`` fails the packet regardless
        #: of its (partial, therefore optimistic) aggregate -- a packet that the
        #: SPRT stopped is REJECTED, never silently passed on a short run.
        self.sprt = sprt

    def evaluate(
        self,
        packet: SkillPacket,
        receiver: ModuleAdapter,
        target_suite: BenchmarkSuite,
        regression_suites: Optional[List[BenchmarkSuite]] = None,
    ) -> EvaluationReport:
        if packet.distilled_skill is None:
            raise EvaluationError(
                "packet {} has no distilled_skill; distil before evaluating".format(
                    packet.packet_id
                )
            )

        skills = [packet.redacted_for_receiver()]

        # Baseline runs to COMPLETION regardless of SPRT: the SPRT only ever
        # stops a CANDIDATE run to REJECT it, and rejecting requires a per-case
        # comparison against a full baseline. A partial baseline would let a
        # failing packet look merely "unverified" rather than "worse", so the
        # baseline is never short-circuited.
        baseline = self.harness.run(receiver, target_suite, split="heldout")

        sprt: Optional[SPRT] = None
        if self.sprt is not None:
            # Build a per-case REJECT oracle: after each scored candidate case,
            # mark it regressed vs the matching baseline case, feed the SPRT,
            # and stop the harness the instant the SPRT says REJECT. By the
            # SPRT's asymmetry this callback can only EVER request a stop to
            # reject -- ``SPRT.should_stop`` is True only on REJECT -- so the
            # harness early-stop is always "stop, this is failing", never "stop,
            # this passed". A good packet runs to completion and scores.sprt
            # stays None (byte-identical to pre-B2).
            baseline_by_id = {c.case_id: c for c in baseline.case_results}
            sprt = SPRT(self.sprt)

            def _stop_callback(results):
                latest = results[-1]
                base = baseline_by_id.get(latest.case_id)
                regressed = base is not None and latest.score < base.score
                sprt.update(regressed)
                return sprt.should_stop()

            candidate = self.harness.run(
                receiver, target_suite, split="heldout", skills=skills,
                stop_callback=_stop_callback,
            )
        else:
            candidate = self.harness.run(
                receiver, target_suite, split="heldout", skills=skills
            )

        regressions = self._check_regressions(receiver, skills, regression_suites or [])
        regressed = [r for r in regressions if r["regressed"]]
        moved = [r for r in regressions if r["moved"]]

        scores = EvaluationScores(
            schema_compliance=metrics.schema_compliance(packet),
            semantic_similarity=candidate.similarity,
            task_success=candidate.task_success,
            language_preservation=candidate.language_preservation,
            hallucination_risk=candidate.hallucination_risk,
            aggregate=metrics.aggregate(
                metrics.schema_compliance(packet),
                candidate.similarity,
                candidate.task_success,
                candidate.language_preservation,
                candidate.hallucination_risk,
            ),
            baseline_score=baseline.score,
            candidate_score=candidate.score,
            regression_detected=bool(regressed),
            regression_detail=(
                "; ".join(
                    "{}: {:.4f} -> {:.4f}".format(
                        r["suite_id"], r["baseline"], r["candidate"]
                    )
                    for r in regressed
                )
                or None
            ),
            control_movement_detected=bool(moved),
            control_movement_detail=(
                "; ".join(
                    "{}: {:.4f} -> {:.4f} (|delta| {:+.4f} > {:.2f})".format(
                        r["suite_id"], r["baseline"], r["candidate"],
                        abs(r["delta"]), self.max_control_movement,
                    )
                    for r in moved
                )
                or None
            ),
        )

        # If the SPRT stopped the candidate run early, stamp the stop record onto
        # the scores so the hard gate check can fail the packet. The candidate
        # run is PARTIAL here -- ``candidate.case_results`` is shorter than the
        # full suite -- so the aggregate above is over a partial, optimistic
        # sample. That is exactly why the gate MUST reject on the SPRT record and
        # not trust the partial aggregate, and why ``case_count`` below reflects
        # cases actually scored (so the audit is honest about what was run).
        if sprt is not None and sprt.should_stop():
            scores.sprt = sprt.stop_record()

        report = EvaluationReport(packet, baseline, candidate, regressions, scores)

        # Gate counts are computed over ALL held-out cases, not the
        # case_diff(limit=50) artefact (which caps the audit row list at 50 for
        # readability). Adversarial audit (2026-08-13): using the capped rows
        # made any case at index >= 50 invisible to the gate's hard
        # case_regression_limit check, so a packet could exceed the configured
        # regression ratio and still promote. The audit row list stays capped;
        # the counts the gate reads do not.
        by_id = {c.case_id: c for c in candidate.case_results}
        scores.case_count = sum(
            1 for base in baseline.case_results if by_id.get(base.case_id) is not None
        )
        scores.case_regression_count = sum(
            1 for base in baseline.case_results
            if (cand := by_id.get(base.case_id)) is not None and cand.score < base.score
        )

        packet.scores = scores
        packet.evaluator_score = scores.aggregate
        packet.promotion_status = PromotionStatus.EVALUATED
        return report

    # -- internals --------------------------------------------------------

    def _check_regressions(
        self,
        receiver: ModuleAdapter,
        skills: List[Dict[str, Any]],
        suites: List[BenchmarkSuite],
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for suite in suites:
            if not suite.split("regression"):
                continue
            before = self.harness.run(receiver, suite, split="regression")
            after = self.harness.run(receiver, suite, split="regression", skills=skills)
            delta = after.score - before.score
            out.append(
                {
                    "suite_id": suite.suite_id,
                    "baseline": round(before.score, 4),
                    "candidate": round(after.score, 4),
                    "delta": round(delta, 4),
                    "regressed": delta < -self.regression_tolerance,
                    # Control-movement (audit 2026-08-17): |delta| in EITHER
                    # direction. A control that IMPROVED (delta > 0) is movement,
                    # not benign -- the packet touched a capability it was not
                    # targeting. ``regressed`` only sees drops; this sees
                    # movement. Gate 1 reads control_movement_detected (below).
                    "moved": abs(delta) > self.max_control_movement,
                }
            )
        return out
