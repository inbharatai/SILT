"""CapabilityTrace construction, validation and loading.

A trace is ONE teacher observation under ONE evidence class
(:data:`asea.capability_build.schema.CAPABILITY_TRACE_SCHEMA`). The schema
enforces class separation; this module enforces the binding contract on top:

  * Every internal observation is bound to an external functional outcome
    through the shared ``sample_id`` plus ``prompt_hash`` -- a trace whose
    router data cannot be tied back to a judged case is evidence of nothing.
  * ``model_revision`` must be the exact checkpoint the teacher answered
    from; a moving tag is rejected.
  * Malformed input raises :class:`asea.capability_build.errors.SpecInvalid`-
    style typed errors; nothing is guessed or defaulted into existence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from asea.artifacts import safe_file

from .errors import CapabilityBuildError, EvidenceClassError
from .schema import (
    CAPABILITY_TRACE_SCHEMA,
    TRACE_CLASS_BEHAVIOURAL,
    TRACE_CLASS_INTERNAL,
    BehaviouralRecord,
    CapabilityTrace,
    Group,
    InternalRecord,
    TraceOutcome,
)

_MAX_TRACE_BYTES = 64 * 1024 * 1024


def prompt_hash(text: str) -> str:
    """sha256 over the exact prompt bytes; the shared key that ties an
    internal observation to an external functional outcome."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validated_group(group: str) -> Group:
    if group not in ("target", "control"):
        raise CapabilityBuildError(
            "trace group must be 'target' or 'control', got %r" % group
        )
    return group


def make_behavioural_trace(
    *,
    capability_id: str,
    sample_id: str,
    model_revision: str,
    prompt: str,
    group: str,
    outcome: TraceOutcome,
    behavioural: BehaviouralRecord,
) -> CapabilityTrace:
    """Build one ``behavioural_remote`` trace. Refuses an internal record
    (EvidenceClassError) -- the two classes are never mixed."""
    return CapabilityTrace(
        capability_id=capability_id,
        sample_id=sample_id,
        model_revision=model_revision,
        prompt_hash=prompt_hash(prompt),
        trace_class=TRACE_CLASS_BEHAVIOURAL,
        group=_validated_group(group),
        outcome=outcome,
        behavioural=behavioural,
        internal=None,
    )


def make_internal_trace(
    *,
    capability_id: str,
    sample_id: str,
    model_revision: str,
    prompt: str,
    group: str,
    outcome: TraceOutcome,
    internal: InternalRecord,
) -> CapabilityTrace:
    """Build one ``internal_open_weight`` trace from worker observations.
    Refuses a behavioural transcript (EvidenceClassError).

    The outcome is REQUIRED to come from the same judged sample as the
    internal observations (same ``sample_id``); the caller enforces this
    when pairing worker output with oracle verdicts. An internal trace with
    ``outcome.success is None`` is still recordable (an unevaluated
    observation), but it can never be used as causal evidence downstream.
    """
    return CapabilityTrace(
        capability_id=capability_id,
        sample_id=sample_id,
        model_revision=model_revision,
        prompt_hash=prompt_hash(prompt),
        trace_class=TRACE_CLASS_INTERNAL,
        group=_validated_group(group),
        outcome=outcome,
        behavioural=None,
        internal=internal,
    )


def validate_trace_payload(raw: Any) -> CapabilityTrace:
    """Validate an already-parsed trace payload; typed errors on failure.
    The store's ``artifact_sha256`` content-address key (added after the
    trace was written, like a receipt's) is tolerated and stripped."""
    if isinstance(raw, dict):
        raw = {k: v for k, v in raw.items() if k != "artifact_sha256"}
    if not isinstance(raw, dict) or raw.get("schema") != CAPABILITY_TRACE_SCHEMA:
        raise CapabilityBuildError(
            "trace must carry schema '%s'" % CAPABILITY_TRACE_SCHEMA
        )
    try:
        return CapabilityTrace.model_validate(raw)
    except ValueError as exc:
        # pydantic raises (or wraps) EvidenceClassErrorValue for class mixing.
        raise EvidenceClassError("trace validation failed: %s" % exc) from exc


def load_trace(path) -> CapabilityTrace:
    """Load and validate a trace from a bounded JSON file."""
    try:
        resolved = safe_file(path)
    except Exception as exc:
        raise CapabilityBuildError("trace path rejected: %s" % exc) from exc
    if not resolved.is_file():
        raise CapabilityBuildError("trace path is not a file: %s" % resolved)
    if resolved.stat().st_size > _MAX_TRACE_BYTES:
        raise CapabilityBuildError("trace exceeds %d bytes" % _MAX_TRACE_BYTES)
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise CapabilityBuildError("trace is not valid JSON: %s" % exc) from exc
    return validate_trace_payload(raw)


def load_trace_bundle(paths) -> List[CapabilityTrace]:
    """Load many traces; all must validate. Used by footprinting, which
    aggregates target and control traces into enrichment statistics."""
    return [load_trace(p) for p in paths]


def trace_fingerprint(trace: CapabilityTrace) -> str:
    from asea.artifacts import digest

    return digest(trace.model_dump(mode="json", by_alias=True))


def traces_same_class(traces: List[CapabilityTrace]) -> Optional[str]:
    """The single evidence class shared by every trace, or None when the
    list is empty or mixed (mixed is refused by callers, never averaged)."""
    if not traces:
        return None
    classes = {t.trace_class for t in traces}
    if len(classes) != 1:
        return None
    return classes.pop()