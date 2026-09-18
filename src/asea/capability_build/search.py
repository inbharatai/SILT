"""Minimum-capability structural reduction search (brief §T, §U).

The search objective, stated honestly:

    minimise Size(M) subject to
        TargetScore(M)   >= spec.minimum_retention_ratio * TargetScore(teacher)
        ControlDrop(M)   <= spec.maximum_control_regression
        HardwareBudget(M) satisfied

Every rule inherited from the compiler's discipline:

  * A parameter-count decrease is NEVER itself evidence of capability
    retention; only held-out functional evaluation counts.
  * Search over reduction CANDIDATES only -- a candidate is never an
    admission; admission is downstream (DeepApply/Gate 2, SiltSpring).
  * Every iteration is recorded, including rejects and failures (failed
    experiments stay visible).
  * Thresholds are operator configuration from the spec, never guarantees
    baked into the search.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from .errors import CapabilityBuildError
from .schema import CapabilitySpec

#: Candidate contract: a caller-supplied builder turns a "keep plan" into
#: a candidate model plus its size, and an evaluator scores it on target
#: and control held-out cases through the host oracle (upstream).
#: ``evaluate(candidate) -> {"target_score": float, "control_scores": dict,
#: "size_bytes": int}`` -- scores are functional pass rates, never
#: parameter counts.
Evaluator = Callable[[Any], Dict[str, Any]]


def search_with_reference(
    spec: CapabilitySpec,
    *,
    teacher_target_score: float,
    candidate_plan: Sequence[Dict[str, Any]],
    build: Callable[[Dict[str, Any]], Any],
    evaluate: Evaluator,
) -> Dict[str, Any]:
    if not candidate_plan:
        raise CapabilityBuildError(
            "the reduction candidate plan is empty; nothing to search"
        )
    if teacher_target_score <= 0:
        raise CapabilityBuildError(
            "teacher reference score must be positive and MEASURED; a "
            "search against an unmeasured teacher is meaningless"
        )
    threshold = spec.minimum_retention_ratio * teacher_target_score
    iterations: List[Dict[str, Any]] = []
    accepted: Optional[Dict[str, Any]] = None
    for index, plan in enumerate(candidate_plan):
        iteration: Dict[str, Any] = {"index": index, "plan": plan}
        iterations.append(iteration)
        try:
            candidate = build(plan)
            scores = evaluate(candidate)
        except Exception as exc:
            iteration["status"] = "failed"
            iteration["error"] = "%s: %s" % (type(exc).__name__, exc)
            continue
        target = float(scores.get("target_score", 0.0))
        control_scores = dict(scores.get("control_scores") or {})
        size = int(scores.get("size_bytes", 0))
        control_regression = max(
            [0.0] + [max(0.0, base - got) for base, got in
                     ((1.0, v) for v in control_scores.values())]
        ) if control_scores else 0.0
        retention = target / teacher_target_score
        ok = (
            target >= threshold
            and control_regression <= spec.maximum_control_regression
        )
        iteration.update({
            "status": "accepted" if ok else "rejected",
            "target_score": target,
            "retention_ratio": retention,
            "control_scores": control_scores,
            "control_regression": control_regression,
            "size_bytes": size,
            "note": (
                "parameter decrease is not evidence; only this functional "
                "held-out score counts"
            ),
        })
        if ok:
            accepted = {
                "index": index,
                "plan": plan,
                "target_score": target,
                "retention_ratio": retention,
                "control_scores": control_scores,
                "size_bytes": size,
                "admission": "CANDIDATE_UNADMITTED",
            }
            break
    return {
        "schema": "silt.capability_search.v1",
        "threshold": threshold,
        "iterations": iterations,
        "accepted": accepted,
        "failed_experiments_visible": any(
            i["status"] in ("failed", "rejected") for i in iterations
        ),
        "autoactivated": False,
    }