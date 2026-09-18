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

from typing import Any, Callable, Dict, List, Sequence

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
    teacher_control_baselines: Dict[str, float],
    candidate_plan: Sequence[Dict[str, Any]],
    build: Callable[[Dict[str, Any]], Any],
    evaluate: Evaluator,
) -> Dict[str, Any]:
    """Run the minimum-capability search over ALL candidates honestly.

    The objective, enforced (not advertised):

      * ``min size_bytes`` over every ACCEPTED candidate -- the whole plan
        is evaluated; the search never stops at the first passing
        candidate when a smaller passing one follows.
      * A candidate is accepted ONLY with: a MEASURED target score, a
        MEASURED result for EVERY control group named in the spec, a
        MEASURED positive size in bytes, and size within the spec's
        ``max_model_storage_bytes`` budget (when set). Anything less is
        recorded as ``rejected`` with the reason -- missing measurements
        are never treated as passing.
      * Control regression is computed against the MEASURED
        ``teacher_control_baselines`` (the same control cases scored on
        the teacher/reference). A baseline of ``1.0`` is never assumed; a
        control group without a measured baseline is an error, and a
        candidate reporting a control group with no measured baseline is
        rejected.
    """
    if not candidate_plan:
        raise CapabilityBuildError(
            "the reduction candidate plan is empty; nothing to search"
        )
    if teacher_target_score <= 0:
        raise CapabilityBuildError(
            "teacher reference score must be positive and MEASURED; a "
            "search against an unmeasured teacher is meaningless"
        )
    if not teacher_control_baselines:
        raise CapabilityBuildError(
            "teacher control baselines must be MEASURED and supplied; "
            "control regression against an assumed 1.0 baseline is a "
            "fabrication and this search refuses to run"
        )
    for group, baseline in teacher_control_baselines.items():
        if not isinstance(baseline, (int, float)) or isinstance(baseline, bool):
            raise CapabilityBuildError(
                "control baseline for %r is not a measured number" % group
            )
    missing_baselines = [g for g in spec.controls if g not in teacher_control_baselines]
    if missing_baselines:
        raise CapabilityBuildError(
            "no measured baseline for control group(s): %s; measure the "
            "teacher on the control cases first" % ", ".join(missing_baselines)
        )
    threshold = spec.minimum_retention_ratio * teacher_target_score
    storage_budget = spec.hardware_budget.max_model_storage_bytes
    iterations: List[Dict[str, Any]] = []
    passing: List[Dict[str, Any]] = []
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
        rejection_reasons: List[str] = []
        target = scores.get("target_score")
        if not isinstance(target, (int, float)) or isinstance(target, bool):
            iteration["status"] = "rejected"
            iteration["reason"] = "target score not measured"
            iterations[-1] = iteration
            continue
        target = float(target)
        control_scores_raw = scores.get("control_scores")
        control_scores = (
            dict(control_scores_raw) if isinstance(control_scores_raw, dict) else {}
        )
        if not control_scores:
            rejection_reasons.append("control results not measured (empty)")
        else:
            for group in spec.controls:
                if group not in control_scores:
                    rejection_reasons.append(
                        "control group %r not measured" % group
                    )
                elif not isinstance(control_scores[group], (int, float)) or isinstance(
                    control_scores[group], bool
                ):
                    rejection_reasons.append(
                        "control group %r score is not a number" % group
                    )
            for group in control_scores:
                if group not in teacher_control_baselines:
                    rejection_reasons.append(
                        "no measured baseline for control group %r" % group
                    )
        size = scores.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            rejection_reasons.append(
                "candidate size not measured (size_bytes missing or not "
                "a positive integer)"
            )
        if rejection_reasons:
            iteration["status"] = "rejected"
            iteration["reason"] = "; ".join(rejection_reasons)
            iteration["target_score"] = target
            iteration["size_bytes"] = size
            continue
        size = int(size)
        if storage_budget is not None and size > storage_budget:
            iteration.update({
                "status": "rejected",
                "reason": "size %d exceeds max_model_storage_bytes %d"
                          % (size, storage_budget),
                "target_score": target,
                "size_bytes": size,
            })
            continue
        control_regression = max(
            max(0.0, float(teacher_control_baselines[g]) - float(control_scores[g]))
            for g in control_scores
        )
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
        if not ok:
            iteration["reason"] = (
                "retention %s below threshold %s or control regression %s "
                "above tolerance %s"
                % (retention, threshold, control_regression,
                   spec.maximum_control_regression)
            )
        else:
            passing.append({
                "index": index,
                "plan": plan,
                "target_score": target,
                "retention_ratio": retention,
                "control_scores": control_scores,
                "control_regression": control_regression,
                "size_bytes": size,
            })
    accepted = (
        min(passing, key=lambda entry: entry["size_bytes"]) if passing else None
    )
    if accepted is not None:
        accepted = dict(accepted)
        accepted["admission"] = "CANDIDATE_UNADMITTED"
        accepted["selection"] = (
            "smallest passing candidate by measured size_bytes out of %d "
            "passing candidates; all %d plan entries were evaluated"
            % (len(passing), len(candidate_plan))
        )
    return {
        "schema": "silt.capability_search.v1",
        "threshold": threshold,
        "control_baselines": dict(teacher_control_baselines),
        "storage_budget_bytes": storage_budget,
        "iterations": iterations,
        "accepted": accepted,
        "failed_experiments_visible": any(
            i["status"] in ("failed", "rejected") for i in iterations
        ),
        "autoactivated": False,
    }