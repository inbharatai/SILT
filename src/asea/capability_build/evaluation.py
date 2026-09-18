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
from typing import Any, Dict, List

from .errors import BlockedResource


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
    (the oracle rejects any extra key, by contract)."""
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
        groups[case["id"]] = case["group"]
    result = evaluate_functions(source, oracle_cases, limits=limits)
    passed = 0
    total = 0
    group_scores: Dict[str, Dict[str, int]] = {}
    for case_result in result.get("cases", []):
        if not isinstance(case_result, dict):
            continue
        # The oracle's per-case entry is {'id', 'matched'} (host-owned).
        matched = case_result.get("matched")
        if not isinstance(matched, bool):
            continue
        total += 1
        passed += int(matched)
        group = groups.get(case_result.get("id"))
        if group is not None:
            bucket = group_scores.setdefault(group, {"passed": 0, "total": 0})
            bucket["total"] += 1
            bucket["passed"] += int(matched)
    summary_groups = {
        group: {
            "passed": counts["passed"],
            "total": counts["total"],
            "pass_rate": counts["passed"] / counts["total"],
        }
        for group, counts in group_scores.items()
    }
    return {
        "oracle": result,
        "judged_cases": total,
        "passed_cases": passed,
        "pass_rate": (passed / total) if total else None,
        "blocked": result.get("status") == "BLOCKED"
        or result.get("returncode") == "BLOCKED"
        or result.get("reason") is not None and "preflight" in str(result.get("reason")),
        "groups": summary_groups,
    }


def functional_judge(case: Dict[str, Any], _state_note: Dict[str, Any]) -> bool:
    """Adapt a pre-judged case to the intervention ``judge`` contract.

    The cases used by causal interventions are judged ONCE by the host
    oracle (or recorded teacher-repair ground truth) and carry their verdict
    in ``case['expected_verdict']``; the intervention replays the SAME
    verdicts per case, varying only which teacher component is masked.
    This keeps intervention scoring deterministic and identical across arms
    without re-invoking the sandbox per token.
    """
    verdict = case.get("expected_verdict")
    if not isinstance(verdict, bool):
        raise ValueError(
            "intervention cases must carry a pre-judged boolean "
            "'expected_verdict' (host-owned; the teacher never grades itself)"
        )
    return verdict