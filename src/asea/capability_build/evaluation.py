"""Functional evaluation for capability-build (brief §M).

The truth signal is FUNCTIONAL CORRECTNESS through the existing external
host oracle -- :func:`asea.certification.function_oracle.evaluate_functions`
plus the Linux-only code sandbox. Nothing is evaluated in-process and the
teacher never grades itself; verdicts are host-owned.

Platform honesty (binding): the sandbox uses unshare/seccomp/chroot and
FAILS CLOSED on Windows/macOS. On a Windows host, ``evaluate``-family
commands must either run under WSL2/Linux or report ``BLOCKED_RESOURCE`` --
never a weaker local execution path, never a silent skip.

This module is deliberately thin: it routes capability-build cases to the
existing oracle with target/control grouping, and turns the oracle's
BLOCKED/partial outcomes into typed results. It adds no new execution
mechanism whatsoever.
"""

from __future__ import annotations

import platform
import re
from typing import Any, Dict, List, Optional

from .errors import BlockedResource, InterventionInvalid


def sandbox_supported() -> bool:
    """True only where the code sandbox actually works (Linux)."""
    return platform.system() == "Linux"


def require_sandbox() -> None:
    """Raise :class:`BlockedResource` unless the host can run the oracle's
    sandbox. Called before any candidate evaluation so a Windows-native run
    fails fast with the exact remedy instead of mid-run."""
    if not sandbox_supported():
        raise BlockedResource(
            requirement=(
                "functional evaluation needs the Linux code sandbox "
                "(unshare/seccomp/chroot), which fails closed on %s"
                % platform.system()
            ),
            remedy="run the evaluate/validate stages inside WSL2 or a "
            "Linux container (the CI matrix and the live pilot do this); "
            "native Windows runs must report BLOCKED",
        )


def evaluate_code_cases(
    source: str, cases: List[Dict[str, Any]], limits=None
) -> Dict[str, Any]:
    """Run one candidate source against data-only function cases through
    the existing oracle. The result is the oracle's own dict (verdicts,
    provenance digests, resource flags) with capability-build group
    bookkeeping layered on top; no verdict is recomputed here.

    Cases here carry the capability-build ``group`` (target/control) next
    to the oracle's exact ``{id, function, args, kwargs, expected}`` frame;
    the wrapper strips the bookkeeping before the oracle sees the suite
    (the oracle rejects any extra key, by contract).

    Accounting contract (audit 2026-09-19, correcting the defect that
    silently dropped candidate-error checks from every denominator): the
    denominator is the AUTHORED oracle count, never the reported judged
    count. A check the oracle could not judge -- the candidate failed to
    import, raised on call, timed out or hit a resource limit (the oracle
    returns such cases absent, with a ``candidate_error`` diagnostic) --
    counts as FAILED, not as absent. Rates can only be deflated by
    failures; a crashed candidate must never inflate one.
    ``judged_pass_rate`` (judged checks only) is exposed for diagnostics
    and is never the headline number."""
    require_sandbox()
    from asea.certification.function_oracle import evaluate_functions

    if not isinstance(source, str) or not source.strip():
        raise ValueError("candidate source must be a non-empty string")
    if not cases:
        raise ValueError("case list must be non-empty")
    oracle_cases: List[Dict[str, Any]] = []
    groups: Dict[Any, Any] = {}
    for case in cases:
        if not isinstance(case, dict) or "group" not in case:
            raise ValueError("every case must carry a 'group' (target/control)")
        missing = [k for k in ("id", "function", "args", "kwargs", "expected")
                   if k not in case]
        if missing:
            raise ValueError(
                "case %r is missing oracle field(s) %s (the oracle needs "
                "exactly id, function, args, kwargs, expected)"
                % (case.get("id"), ", ".join(missing))
            )
        oracle_cases.append({
            "id": case["id"], "function": case["function"],
            "args": case["args"], "kwargs": case["kwargs"],
            "expected": case["expected"],
        })
        if case["id"] in groups:
            raise ValueError("duplicate oracle case id %r" % (case["id"],))
        groups[case["id"]] = case["group"]
    result = evaluate_functions(source, oracle_cases, limits=limits)
    oracle_status = result.get("status")
    if not isinstance(oracle_status, str) or not oracle_status:
        raise ValueError(
            "the oracle returned no status; refusing to guess one "
            "(a missing status must never default to 'measured')"
        )
    verdicts: Dict[Any, bool] = {}
    for case_result in result.get("cases", []):
        if not isinstance(case_result, dict):
            raise ValueError("oracle returned a non-object case entry")
        cid = case_result.get("id")
        if cid not in groups:
            raise ValueError(
                "oracle returned a case id that was not authored: %r" % (cid,)
            )
        if cid in verdicts:
            raise ValueError(
                "oracle returned duplicate verdicts for case %r" % (cid,)
            )
        matched = case_result.get("matched")
        if not isinstance(matched, bool):
            raise ValueError(
                "oracle case %r carries no boolean verdict; refusing to "
                "grade on a partial oracle response" % (cid,)
            )
        verdicts[cid] = matched
    authored = len(oracle_cases)
    judged = len(verdicts)
    # Unjudged = authored checks the oracle could not run (candidate
    # import/call error, timeout, resource kill). Each counts as FAILED.
    unjudged = authored - judged
    passed = sum(1 for matched in verdicts.values() if matched)
    group_counts: Dict[str, Dict[str, int]] = {
        group: {"passed": 0, "total": 0, "unjudged": 0}
        for group in groups.values()
    }
    for cid, group in groups.items():
        bucket = group_counts[group]
        bucket["total"] += 1
        if cid in verdicts:
            if verdicts[cid]:
                bucket["passed"] += 1
        else:
            bucket["unjudged"] += 1
    summary_groups = {
        group: {
            "passed": counts["passed"],
            "total": counts["total"],
            "unjudged": counts["unjudged"],
            "pass_rate": counts["passed"] / counts["total"],
        }
        for group, counts in group_counts.items()
    }
    return {
        "oracle": result,
        "status": oracle_status,
        "authored_cases": authored,
        "judged_cases": judged,
        "unjudged_cases": unjudged,
        "passed_cases": passed,
        "failed_cases": authored - passed,
        "pass_rate": passed / authored,
        "judged_pass_rate": (passed / judged) if judged else None,
        "blocked": result.get("status") == "BLOCKED"
        or result.get("returncode") == "BLOCKED"
        or result.get("reason") is not None and "preflight" in str(result.get("reason")),
        "groups": summary_groups,
    }


def functional_judge(case: Dict[str, Any], _state_note: Dict[str, Any]) -> bool:
    """REFUSED on purpose (audit 2026-09-18): this used to replay a
    pre-judged ``case['expected_verdict']`` and was documented as the
    intervention ``judge``. A replay judge CANNOT serve an intervention:
    ``run_intervention`` scores the base and masked arms with the same
    judge, so replayed verdicts are identical in both arms and every
    intervention silently reports ``target_drop == 0`` -- fabricated
    non-causality, the exact defect class this project refuses.

    Use :func:`make_worker_judge` instead: it regenerates the teacher's
    output under the CURRENT state through the worker's ``generate`` op
    and grades it against the case's oracle frames through the host
    oracle. This stub refuses the replay shortcut at the library boundary
    so no wiring can reach for it and get silent zeros.
    """
    raise InterventionInvalid(
        "functional_judge cannot be used as an intervention judge: replaying "
        "pre-judged verdicts is blind to the masked state, so base and masked "
        "arms score identically and every drop would silently be 0. Use "
        "make_worker_judge (regenerate through the worker, grade through the "
        "host oracle) -- the CLI's intervene command does"
    )


_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)

#: The five fields the host oracle needs per frame -- enforced before the
#: worker is touched so a malformed case set is a named refusal, never a
#: mid-intervention crash.
_ORACLE_FRAME_FIELDS = ("id", "function", "args", "kwargs", "expected")


def extract_candidate_code(completion: str) -> Optional[str]:
    """Extract the candidate source from a teacher completion -- the SAME
    rule the judged pilot uses: the LAST fenced python block, else the
    whole response when it contains a ``def``, else nothing (a completion
    with no extractable code is a candidate error, which the authored-
    denominator accounting counts as FAILED -- never silently absent)."""
    if not isinstance(completion, str) or not completion.strip():
        return None
    blocks = _FENCE_RE.findall(completion)
    if blocks:
        return blocks[-1]
    if "def " in completion:
        return completion
    return None


def make_worker_judge(client, max_new_tokens: int = 512):
    """Build the REAL intervention judge: regenerate the teacher's output
    under the CURRENT (base or masked) state through the worker's
    ``generate`` op, then grade it against the case's oracle frames
    through the host oracle (audit 2026-09-19, item 8: replaces the old
    "generation+judging stage not wired" refusal).

    The case must carry ``oracle`` frames ``[{id, function, args, kwargs,
    expected}]``; the verdict is the authored-denominator result over the
    case's frames -- every frame must pass, a candidate error fails.

    State contamination guard: the worker reports its live masks with
    every generation; a judge scoring the "base" phase on a masked
    teacher, or the "masked" phase on an unmasked one, refuses by name
    instead of recording a measurement that is not what it says.
    """
    if max_new_tokens <= 0:
        raise InterventionInvalid("max_new_tokens must be positive")

    def judge(case: Dict[str, Any], note: Dict[str, Any]) -> bool:
        frames = case.get("oracle")
        if not isinstance(frames, list) or not frames:
            raise InterventionInvalid(
                "case %r carries no oracle frames; an intervention verdict "
                "must be measured against authored checks, not guessed "
                "(add [{id, function, args, kwargs, expected}] per case)"
                % case.get("sample_id")
            )
        seen_ids = set()
        oracle_cases = []
        for frame in frames:
            if not isinstance(frame, dict):
                raise InterventionInvalid(
                    "case %r has a non-object oracle frame" % case.get("sample_id")
                )
            missing = [f for f in _ORACLE_FRAME_FIELDS if f not in frame]
            if missing:
                raise InterventionInvalid(
                    "case %r has an oracle frame missing %s"
                    % (case.get("sample_id"), ", ".join(missing))
                )
            if frame["id"] in seen_ids:
                raise InterventionInvalid(
                    "case %r reuses oracle frame id %r"
                    % (case.get("sample_id"), frame["id"])
                )
            seen_ids.add(frame["id"])
            oracle_cases.append({
                "id": frame["id"], "function": frame["function"],
                "args": frame["args"], "kwargs": frame["kwargs"],
                "expected": frame["expected"], "group": case["group"],
            })
        result = client.request("generate", {
            "prompts": [{
                "sample_id": case["sample_id"],
                "group": case["group"],
                "prompt": case["prompt"],
            }],
            "max_new_tokens": max_new_tokens,
        })
        per = (result.get("per_prompt") or [None])[0]
        if not per or per.get("sample_id") != case["sample_id"]:
            raise InterventionInvalid(
                "worker returned no completion for case %r" % case.get("sample_id")
            )
        # the state the worker ACTUALLY generated under must match the
        # state this verdict will be attributed to
        live = list(result.get("active_masks") or [])
        masked_phase = note.get("phase") == "masked"
        if masked_phase and not live:
            raise InterventionInvalid(
                "judge asked to score the masked phase for %r but the worker "
                "reported NO active masks -- the baseline and masked arms "
                "must not be contaminated" % case.get("sample_id")
            )
        if not masked_phase and live:
            raise InterventionInvalid(
                "judge asked to score the base phase for %r but the worker "
                "reported active mask(s) %s -- a masked teacher's baseline "
                "is not a baseline" % (case.get("sample_id"), live)
            )
        source = extract_candidate_code(per.get("completion"))
        if source is None:
            # no extractable code: candidate error -> FAILED under the
            # authored-denominator accounting, never silently absent
            return False
        graded = evaluate_code_cases(source, oracle_cases)
        return bool(
            graded["passed_cases"] == graded["authored_cases"]
        )

    return judge