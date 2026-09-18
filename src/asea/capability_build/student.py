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

from typing import Any, Callable, Dict, List, Optional, Sequence
from urllib.parse import urlsplit

from .errors import BlockedResource
from .schema import EvaluationSplits

#: Judge contract: (case, student_output) -> True/False (host-owned
#: verdict) or None to ABSTAIN (unjudged). Abstentions are excluded from
#: the pass rate and counted separately -- never folded into failures.
StudentJudge = Callable[[Dict[str, Any], str], Optional[bool]]

#: The ONLY hostnames a student connector may use. Literal loopback names
#: and addresses -- no DNS names at all, because a resolvable name is not
#: provably local (``localhost.example.invalid`` parses as a hostname and
#: is NOT localhost; a public name that happens to resolve to 127.0.0.1
#: today may not tomorrow).
_LOCAL_HOSTNAMES = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})


def validate_local_student_host(host: str) -> str:
    """Parse ``host`` and refuse anything that is not a literal loopback.

    This is a real parse (:func:`urllib.parse.urlsplit`), not a string
    prefix check: ``http://localhost.example.invalid:11434`` and
    ``http://127.0.0.1.evil.example:11434`` are REFUSED here even though a
    ``startswith("http://localhost")`` check would accept both.

    Honest limits (binding):

      * The connector posts directly to this URL and does NOT follow
        redirects; a 3xx from the daemon surfaces as an error, never as a
        silent second hop to an unvalidated host.
      * A URL can only prove which host was *addressed*. That the selected
        model executes locally is enforced by the caller running the actual
        inference through this local connector only -- there is no remote
        student path in this build, so no silent escape route exists.
    """
    try:
        parts = urlsplit(host)
    except ValueError as exc:
        raise BlockedResource(
            requirement="student connector host %r does not parse as a URL (%s)"
                         % (host, exc),
            remedy="use http://localhost:11434",
        ) from exc
    if parts.scheme != "http":
        raise BlockedResource(
            requirement="student connector host %r must be a plain http:// URL"
                         % host,
            remedy="use http://localhost:11434",
        )
    if parts.username or parts.password or parts.query or parts.fragment:
        raise BlockedResource(
            requirement="student connector host %r carries userinfo, query or "
                        "fragment; a local daemon needs none of these" % host,
            remedy="use http://localhost:11434",
        )
    if parts.path not in ("", "/"):
        raise BlockedResource(
            requirement="student connector host %r carries a path; a local "
                        "Ollama daemon is addressed at its root" % host,
            remedy="use http://localhost:11434",
        )
    name = (parts.hostname or "").lower()
    if name not in _LOCAL_HOSTNAMES:
        raise BlockedResource(
            requirement=(
                "student connector host %r is not a literal loopback address "
                "(parsed hostname %r); remote students are not supported in "
                "this build" % (host, name)
            ),
            remedy="run a local Ollama daemon (ollama serve) and use "
                   "http://localhost:11434",
        )
    return name


def local_ollama_student(
    model: str, host: str = "http://localhost:11434", max_new_tokens: int = 512
) -> Dict[str, Any]:
    """Describe a local Ollama student. The connector itself is the
    existing :class:`asea.modules.real.ollama.OllamaConnector` (local
    daemon, no ML deps, no network beyond localhost); this function only
    validates the request and returns the wiring the caller needs, so the
    remote-consent machinery can never be confused with student runs.
    The host is validated by :func:`validate_local_student_host` -- a real
    URL parse, never a string prefix check."""
    validate_local_student_host(host)
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
    unjudged = 0
    for case in cases:
        output = infer(student, case)
        verdict = judge(case, output)
        if verdict is None:
            # An abstaining judge is NOT a failing verdict: folding unjudged
            # cases into "passed: 0" would manufacture a lower pass rate the
            # judge never issued (audit 2026-09-18). Exclude from the rate
            # and count them separately, honestly.
            unjudged += 1
            per_case.append({
                "sample_id": case.get("sample_id"),
                "group": case.get("group"),
                "verdict": None,
                "unjudged": True,
            })
            continue
        verdict = bool(verdict)
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
        "unjudged_excluded": unjudged,
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