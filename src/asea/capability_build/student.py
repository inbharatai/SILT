"""Student baselines for capability-build (brief §O, §Q).

A student baseline measures what a SMALLER receiver already does on the
same case sets the teacher was measured on -- the measured gap
``Gap = TeacherScore - StudentScore`` is the only justification for any
transfer work. No gap, no distillation.

Honesty rules (binding):

  * Student connectors here are LOCAL (a local Ollama daemon or an
    in-process open-weight model). No remote consent flag exists for the
    student side because no remote student is ever silently used.
  * Outcomes are judged by the host oracle upstream -- a student never
    grades itself, and a baseline is reported as numbers only, never as
    "capability X".
  * A student that cannot run (model missing, platform blocked) raises
    :class:`BlockedResource` with requirement+remedy -- never a fabricated
    baseline.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence

from .errors import BlockedResource
from .schema import EvaluationSplits

#: Judge contract: (case, student_output) -> bool (host-owned verdict).
StudentJudge = Callable[[Dict[str, Any], str], bool]


def local_ollama_student(
    model: str, host: str = "http://localhost:11434", max_new_tokens: int = 512
) -> Dict[str, Any]:
    """Describe a local Ollama student. The connector itself is the
    existing :class:`asea.modules.real.ollama.OllamaConnector` (local
    daemon, no ML deps, no network beyond localhost); this function only
    validates the request and returns the wiring the caller needs, so the
    remote-consent machinery can never be confused with student runs."""
    if not host.startswith("http://localhost") and not host.startswith(
        "http://127.0.0.1"
    ):
        raise BlockedResource(
            requirement=(
                "student connector host %r is not local; remote students "
                "are not supported in this build" % host
            ),
            remedy="run a local Ollama daemon (ollama serve) and use "
                   "http://localhost:11434",
        )
    return {
        "kind": "ollama_local",
        "model": model,
        "host": host,
        "max_new_tokens": max_new_tokens,
    }


def measure_student_baseline(
    student: Dict[str, Any],
    cases: Sequence[Dict[str, Any]],
    *,
    infer: Callable[[Dict[str, Any], Dict[str, Any]], str],
    judge: StudentJudge,
) -> Dict[str, Any]:
    """Run every case through the student and judge it with the host-owned
    verdict. ``infer(student, case) -> output`` abstracts the connector so
    tests can drive the MECHANISM deterministically.

    Returns per-group pass rates and the measured gap inputs. Numbers
    only: the result never says "the student has capability X".
    """
    if not cases:
        raise BlockedResource(
            requirement="student baseline needs a non-empty case set",
            remedy="build the case sets first (data/capability_v1/)",
        )
    groups: Dict[str, Dict[str, int]] = {}
    per_case = []
    for case in cases:
        output = infer(student, case)
        verdict = bool(judge(case, output))
        bucket = groups.setdefault(case["group"], {"passed": 0, "total": 0})
        bucket["total"] += 1
        bucket["passed"] += int(verdict)
        per_case.append({
            "sample_id": case.get("sample_id"),
            "group": case.get("group"),
            "verdict": verdict,
        })
    summary = {
        "groups": {
            group: {
                "passed": counts["passed"],
                "total": counts["total"],
                "pass_rate": counts["passed"] / counts["total"],
            }
            for group, counts in groups.items()
        },
        "per_case": per_case,
        "note": "baseline numbers only; not a capability claim",
    }
    return summary


def measured_gap(teacher_rate: float, student_rate: float) -> Dict[str, Any]:
    """The only transfer justification: a measured, actionable gap. A gap
    within noise is reported as such -- transfer work without a gap is
    unjustified and must not start."""
    gap = teacher_rate - student_rate
    return {
        "teacher_rate": teacher_rate,
        "student_rate": student_rate,
        "gap": gap,
        "actionable": gap > 0.05,
        "note": (
            "transfer proceeds only when the gap is measurable and "
            "actionable; a zero gap means no distillation work is justified"
        ),
    }