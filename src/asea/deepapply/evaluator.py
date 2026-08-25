"""Gate 2 evaluator: does the trained adapter actually make the receiver better?

This is a before/after A/B on the **held-out** split, identical in discipline to
:mod:`asea.evaluator.evaluator`, but the "candidate" is the receiver *with the
adapter attached* (weights), not the receiver *with a packet injected* (prompt):

    baseline  = receiver alone            (harness.run(receiver, suite, "heldout"))
    candidate = receiver + adapter          (harness.run(adapted, suite, "heldout"))

plus a regression sweep over suites the transfer is *not* targeting. Reuses
:class:`BenchmarkHarness` (the same harness, similarity backend and metrics the
packet evaluator uses) -- only the conditioning differs. The harness runs any
:class:`ModuleAdapter`, so we pass it the baseline receiver and the
adapter-attached receiver in turn.

What this measures honestly: whether attaching the adapter changes the
receiver's held-out outputs for the better, under whatever similarity backend the
harness was constructed with. What it does not measure: real-world adequacy, or
correctness beyond the suite's reference strings.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..benchmarks.harness import BenchmarkHarness, BenchmarkSuite, SuiteResult
from ..core.interfaces import ModuleAdapter
from ..core.protocol import EvaluationScores, PromotionStatus
from ..evaluator import metrics
from .adapter_packet import AdapterPacket


class DeepApplyEvaluationReport:
    def __init__(
        self,
        adapter: AdapterPacket,
        baseline: SuiteResult,
        candidate: SuiteResult,
        regressions: List[Dict[str, Any]],
        scores: EvaluationScores,
    ) -> None:
        self.adapter = adapter
        self.baseline = baseline
        self.candidate = candidate
        self.regressions = regressions
        self.scores = scores

    @property
    def improvement(self) -> float:
        return self.candidate.score - self.baseline.score

    def case_diff(self, limit: int = 50) -> List[Dict[str, Any]]:
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
            "adapter_id": self.adapter.adapter_id,
            "baseline": self.baseline.summary(),
            "candidate": self.candidate.summary(),
            "case_diff": self.case_diff(),
            "improvement": round(self.improvement, 4),
            "regressions": self.regressions,
            "scores": self.scores.model_dump(),
            "similarity_is_semantic": self.candidate.similarity_is_semantic,
        }


class DeepApplyEvaluator:
    def __init__(
        self,
        harness: Optional[BenchmarkHarness] = None,
        regression_tolerance: float = 0.02,
        max_control_movement: float = 0.05,
    ) -> None:
        self.harness = harness or BenchmarkHarness()
        #: How much a non-targeted capability may drop before we call it a
        #: regression. Same default as the packet evaluator.
        self.regression_tolerance = regression_tolerance
        #: How much a CONTROL (non-target) suite may move in EITHER direction
        #: before we call it control movement (audit 2026-08-17). The regression
        #: flag below only catches DROPS; this bound catches movement (|delta|),
        #: so a control suite that IMPROVED -- training bleeding into a
        #: capability it was not targeting -- is no longer invisible to Gate 2.
        self.max_control_movement = max_control_movement

    def evaluate(
        self,
        adapter: AdapterPacket,
        artifact: Any,
        receiver: ModuleAdapter,
        target_suite: BenchmarkSuite,
        regression_suites: Optional[List[BenchmarkSuite]] = None,
    ) -> DeepApplyEvaluationReport:
        """Run the held-out A/B + regression sweep for a trained adapter.

        ``artifact`` is the :class:`AdapterArtifact` returned by the trainer; its
        ``attach(receiver)`` produces the adapter-conditioned module. The harness
        runs that module exactly as it runs any other, so the evaluation reuses
        the same similarity backend and metrics the packet path uses.

        Memory discipline: baseline and candidate are run SEQUENTIALLY, not
        co-resident. The baseline module is loaded once, runs the held-out target
        suite plus every regression suite, then is unloaded; the candidate
        (``artifact.attach``) is loaded once, runs the same suites, then is
        unloaded. On a memory-bounded GPU this keeps the peak at ONE model copy
        (e.g. one 4-bit 7B ~5.6 GB on an 8 GB card) instead of two co-resident
        (~11 GB). This is score-equivalent: each ``harness.run`` uses the same
        deterministic decoding (``do_sample=False``) regardless of when the
        weights were materialized, so the A/B scores are byte-identical to the
        co-resident ordering. Modules without an ``unload`` (mocks/small CPU
        models in the SILT suite) are left as-is -- nothing to free.
        """
        regression_suites = regression_suites or []

        # --- Baseline phase: receiver loaded once, runs target + regressions. ---
        baseline = self.harness.run(receiver, target_suite, split="heldout")
        baseline_reg: Dict[str, SuiteResult] = {}
        for suite in regression_suites:
            if not suite.split("regression"):
                continue
            baseline_reg[suite.suite_id] = self.harness.run(
                receiver, suite, split="regression"
            )
        _safe_unload(receiver)

        # --- Candidate phase: attach loads the base+LoRA once, runs the same. -
        adapted = artifact.attach(receiver)
        candidate = self.harness.run(adapted, target_suite, split="heldout")
        candidate_reg: Dict[str, SuiteResult] = {}
        for suite in regression_suites:
            if not suite.split("regression"):
                continue
            candidate_reg[suite.suite_id] = self.harness.run(
                adapted, suite, split="regression"
            )
        _safe_unload(adapted)

        # Reassemble the regression list in the original suite order.
        regressions: List[Dict[str, Any]] = []
        for suite in regression_suites:
            if not suite.split("regression"):
                continue
            before = baseline_reg[suite.suite_id]
            after = candidate_reg[suite.suite_id]
            delta = after.score - before.score
            regressions.append(
                {
                    "suite_id": suite.suite_id,
                    "baseline": round(before.score, 4),
                    "candidate": round(after.score, 4),
                    "delta": round(delta, 4),
                    "regressed": delta < -self.regression_tolerance,
                    # Control-movement (audit 2026-08-17): |delta| in EITHER
                    # direction. A control that IMPROVED (delta > 0) is movement,
                    # not benign -- the adapter touched a capability it was
                    # not targeting. ``regressed`` only sees drops; this sees
                    # movement. Gate 2 reads control_movement_detected (below).
                    "moved": abs(delta) > self.max_control_movement,
                }
            )
        regressed = [r for r in regressions if r["regressed"]]
        moved = [r for r in regressions if r["moved"]]

        scores = EvaluationScores(
            schema_compliance=1.0,  # adapter is structurally valid; checked by the gate
            semantic_similarity=candidate.similarity,
            task_success=candidate.task_success,
            language_preservation=candidate.language_preservation,
            hallucination_risk=candidate.hallucination_risk,
            aggregate=metrics.aggregate(
                1.0,
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
                    "{}: {:.4f} -> {:.4f}".format(r["suite_id"], r["baseline"], r["candidate"])
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

        report = DeepApplyEvaluationReport(adapter, baseline, candidate, regressions, scores)

        # Gate counts are over ALL held-out cases, not the capped case_diff list
        # (same discipline as the packet evaluator, audit 2026-08-13).
        by_id = {c.case_id: c for c in candidate.case_results}
        scores.case_count = sum(
            1 for base in baseline.case_results if by_id.get(base.case_id) is not None
        )
        scores.case_regression_count = sum(
            1 for base in baseline.case_results
            if (cand := by_id.get(base.case_id)) is not None and cand.score < base.score
        )

        adapter.scores = scores
        adapter.evaluator_score = scores.aggregate
        adapter.promotion_status = PromotionStatus.EVALUATED
        return report


def _safe_unload(module: ModuleAdapter) -> None:
    """Release a module's weights if it exposes ``unload``; no-op otherwise.

    Mock adapters and small CPU models (the SILT suite) have nothing to free, so
    a missing ``unload`` is fine. Real HF connectors and the adapted LoRA module
    define ``unload`` to drop weights + empty the CUDA cache, letting the
    evaluator keep only one model resident at a time on a bounded GPU.
    """
    fn = getattr(module, "unload", None)
    if callable(fn):
        try:
            fn()
        except Exception:
            # Unload must never break the A/B: a failure to free is a
            # memory regression, not a correctness one. The next phase's
            # load will raise on a genuine OOM if it actually can't fit.
            pass