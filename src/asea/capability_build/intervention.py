"""Causal intervention machinery (brief §K) -- net-new.

The compiler's ``apply_selection`` physically prunes experts; no temporary
masking path ever existed in SILT. This module adds one, with a fixed
protocol every intervention MUST follow:

  1. ``verify_unchanged`` the teacher against its parameter hashes BEFORE
     anything runs (teacher weights are read-only in aggregate).
  2. Measure baseline performance on the judge over target AND control
     cases (seeded, identical inputs for both arms).
  3. ``temporary_mask`` exactly one component.
  4. Measure masked performance over the SAME cases in the SAME order.
  5. ``restore_mask`` immediately.
  6. ``verify_unchanged`` again -- the teacher's parameters must hash
     bit-identically to step 1. A restore that does not verify raises
     :class:`InterventionInvalid` and the entry is NEVER recorded as valid.
  7. Emit an :class:`InterventionEntry` with
     ``target_drop = Perf(base,target) - Perf(masked,target)`` (same for
     control), the case counts, and ``restored_and_verified``.

Scores are computed by a caller-supplied ``judge`` (functional-correctness
verdicts owned by the host oracle upstream -- the teacher never grades
itself). The judge returns a boolean per case; performance is the mean.

Honesty rules (binding):

  * A component that fails to restore leaves the run marked failed; there
    is no "best effort" recovery that keeps the score.
  * ``restored_and_verified=False`` entries may be RECORDED (failed
    experiments stay visible) but downstream footprint builders must
    exclude them from causal evidence.
  * The intervention score is evidence about ONE component under ONE
    workload with ONE seed-set; it is never a general importance ranking,
    and the ~18B active parameters are never claimed to be an
    identifiable 18B subset.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, Iterable, List, Sequence

from .errors import InterventionInvalid
from .schema import InterventionEntry

#: Judge contract: takes (case, adapter_state_note) -> bool (functional
#: verdict). ``adapter_state_note`` is a dict the adapter may use to pass
#: e.g. generation determinism knobs; judges ignore what they don't need.
Judge = Callable[[Dict[str, Any], Dict[str, Any]], bool]


class InterventionTarget:
    """One addressable teacher component: ``("expert", layer, expert_id)``
    or ``("layer", layer)`` today; the adapter validates existence."""

    __slots__ = ("kind", "layer", "expert_id")

    def __init__(self, kind: str, layer: int, expert_id: int = -1) -> None:
        if kind not in ("expert", "layer"):
            raise InterventionInvalid(
                "component kind must be 'expert' or 'layer', got %r" % kind
            )
        if layer < 0 or expert_id < -1:
            raise InterventionInvalid("layer/expert indices must be >= 0")
        if kind == "layer" and expert_id != -1:
            raise InterventionInvalid("a layer target carries no expert id")
        if kind == "expert" and expert_id < 0:
            raise InterventionInvalid("an expert target requires expert_id >= 0")
        self.kind = kind
        self.layer = layer
        self.expert_id = expert_id

    @property
    def key(self) -> str:
        if self.kind == "layer":
            return "layer:%d" % self.layer
        return "expert:%d/%d" % (self.layer, self.expert_id)

    def as_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "layer": self.layer, "expert_id": self.expert_id}


def _mean_performance(
    judge: Judge, cases: Sequence[Dict[str, Any]], adapter, seed: int
) -> float:
    """Mean functional verdict over cases, evaluated in seeded order."""
    order = list(cases)
    random.Random(seed).shuffle(order)
    verdicts = 0
    if not order:
        raise InterventionInvalid("cannot score an empty case set")
    for case in order:
        if judge(case, {}):
            verdicts += 1
    return verdicts / len(order)


def run_intervention(
    adapter,
    target: InterventionTarget,
    *,
    target_cases: Sequence[Dict[str, Any]],
    control_cases: Sequence[Dict[str, Any]],
    judge: Judge,
    seed: int,
) -> InterventionEntry:
    """Execute one mask-measure-restore-verify intervention.

    ``adapter`` is an :class:`asea.capability_build.adapters.base.MoEArchitectureAdapter`
    (or anything honouring the same contract: ``verify_unchanged``,
    ``temporary_mask``, ``restore_mask``). The teacher is verified unchanged
    before AND after; a mismatch at either point aborts with
    :class:`InterventionInvalid` and no entry is returned.
    """
    if not target_cases:
        raise InterventionInvalid("intervention needs a non-empty target set")
    if not control_cases:
        raise InterventionInvalid("intervention needs a non-empty control set")

    # 1. Teacher must start in its verified, unmodified state.
    before = adapter.verify_unchanged()
    if not before.get("unchanged", False):
        raise InterventionInvalid(
            "teacher parameters do not match their recorded hashes before "
            "the intervention; refusing to touch a modified teacher "
            "(detail: %s)" % before.get("detail")
        )

    # 2. Baseline performance, both groups, seeded order.
    base_target = _mean_performance(judge, target_cases, adapter, seed)
    base_control = _mean_performance(judge, control_cases, adapter, seed)

    # 3. Mask, measure, unmask -- the ONLY mutation window.
    adapter.temporary_mask(target)
    try:
        masked_target = _mean_performance(judge, target_cases, adapter, seed)
        masked_control = _mean_performance(judge, control_cases, adapter, seed)
    finally:
        # 5. Restore happens even if measurement raised.
        adapter.restore_mask(target)

    # 6. The restore must be provable: parameters hash-identical to step 1.
    after = adapter.verify_unchanged()
    restored_and_verified = bool(after.get("unchanged", False))

    target_drop = base_target - masked_target
    control_drop = base_control - masked_control
    entry = InterventionEntry(
        component=target.key,
        target_drop=target_drop,
        control_drop=control_drop,
        target_cases=len(target_cases),
        control_cases=len(control_cases),
        restored_and_verified=restored_and_verified,
        seed=seed,
    )
    if not restored_and_verified:
        # Recorded (failed experiments stay visible) but flagged; the
        # detail explains WHY it is not causal evidence.
        raise InterventionInvalid(
            "intervention on %s did not restore-and-verify (detail: %s); "
            "the component score %r is NOT valid causal evidence"
            % (target.key, after.get("detail"), target_drop)
        )
    return entry


def attempt_intervention(
    adapter,
    target: InterventionTarget,
    *,
    target_cases: Sequence[Dict[str, Any]],
    control_cases: Sequence[Dict[str, Any]],
    judge: Judge,
    seed: int,
) -> Dict[str, Any]:
    """Non-raising wrapper for search loops: returns
    ``{"entry": InterventionEntry, "ok": True}`` or
    ``{"entry": None, "ok": False, "error": "..."}``. A failed intervention
    stays visible in the search record; it is never silently dropped."""
    try:
        entry = run_intervention(
            adapter,
            target,
            target_cases=target_cases,
            control_cases=control_cases,
            judge=judge,
            seed=seed,
        )
        return {"entry": entry, "ok": True, "error": None}
    except InterventionInvalid as exc:
        return {"entry": None, "ok": False, "error": str(exc)}


def causal_evidence_only(
    interventions: Dict[str, InterventionEntry]
) -> Dict[str, InterventionEntry]:
    """Filter an intervention map down to the entries usable as causal
    evidence: restored, verified, and measured on non-empty case sets.
    Failed attempts are excluded from evidence but the caller keeps them in
    the run record (failure history)."""
    return {
        key: entry
        for key, entry in interventions.items()
        if entry.restored_and_verified
        and entry.target_cases > 0
        and entry.control_cases > 0
    }