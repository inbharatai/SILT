"""CapabilityFootprint construction (brief §J, §L).

A footprint records "computational components associated with and
experimentally important to a defined capability under a defined workload"
-- NEVER "these neurons contain coding knowledge".

The honesty discipline inherited verbatim from the compiler
(``asea.compiler.core``): **routing/usage enrichment is correlation only --
usage evidence, not causal expert importance.** Enrichment is computed per
component as target-usage vs control-usage from ``internal_open_weight``
traces, with these guards:

  * Zero observations are recorded as ``null`` -- "unknown is not zero
    importance" (compiler REAP discipline).
  * Enrichment scores are log-ratios with a symmetric epsilon; a component
    seen in target but never in control is marked
    ``"control_unobserved": true`` rather than scored infinite.
  * A component with no observations in EITHER group carries no entry at
    all -- absence of evidence is recorded by omission plus a limitation
    sentence, never as a zero.

Only entries under ``causal_interventions`` (built by
:mod:`asea.capability_build.intervention`) carry causal evidence, and only
for the components actually intervened on with the exact seeds recorded.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from .errors import FootprintInvalid
from .schema import (
    CAPABILITY_FOOTPRINT_SCHEMA,
    TRACE_CLASS_BEHAVIOURAL,
    TRACE_CLASS_INTERNAL,
    CapabilityFootprint,
    CapabilityTrace,
    FootprintEvidence,
    InterventionEntry,
)
from .trace import traces_same_class

#: Symmetric epsilon for log-ratio enrichment (both groups get the same
#: pseudo-count, so the ratio is well-defined and symmetric in direction).
_EPS = 0.5

_CORRELATION_LIMITATION = (
    "Routing and usage enrichment in 'layers'/'experts' is usage evidence, "
    "not causal expert importance (correlation only)."
)
_BEHAVIOURAL_LIMITATION = (
    "This footprint rests on behavioural_remote evidence: the teacher "
    "exposed prompts/responses only, so no internal component claims are "
    "made beyond what causal interventions on an open-weight teacher "
    "could verify (none are recorded unless listed under "
    "causal_interventions)."
)
_SAMPLE_SIZE_LIMITATION = (
    "Enrichment statistics are computed over the recorded case counts "
    "only and do not support claims beyond this workload and split."
)


def _component_usage(internal: Any) -> Dict[str, float]:
    """Pull a flat {component_key: usage_mass} map out of one internal
    record (InternalRecord model or its dict form). Layer-level masses
    live under ``layers[<layer>].usage_mass`` and expert-level under
    ``layers[<layer>].experts[<expert>].usage_mass``."""
    if internal is None:
        return {}
    if hasattr(internal, "layers"):
        layers = internal.layers or {}
    elif isinstance(internal, dict):
        layers = internal.get("layers") or {}
    else:
        layers = {}
    usage: Dict[str, float] = {}
    for layer_id, layer_data in layers.items():
        if not isinstance(layer_data, dict):
            continue
        mass = layer_data.get("usage_mass")
        if isinstance(mass, (int, float)) and not isinstance(mass, bool):
            usage["layer:%s" % layer_id] = float(mass)
        for expert_id, expert_data in (layer_data.get("experts") or {}).items():
            if not isinstance(expert_data, dict):
                continue
            emass = expert_data.get("usage_mass")
            if isinstance(emass, (int, float)) and not isinstance(emass, bool):
                usage["expert:%s/%s" % (layer_id, expert_id)] = float(emass)
    return usage


def compute_enrichment(traces: List[CapabilityTrace]) -> Dict[str, Any]:
    """Target-vs-control usage enrichment per component.

    Requires ``internal_open_weight`` traces (a behavioural teacher exposes
    no internals; asking for enrichment from it is a caller bug, refused
    with :class:`FootprintInvalid`).
    """
    evidence_class = traces_same_class(traces)
    if evidence_class is None:
        raise FootprintInvalid(
            "enrichment needs a non-empty, single-class set of internal "
            "traces (empty or mixed evidence classes are refused)"
        )
    if evidence_class != TRACE_CLASS_INTERNAL:
        raise FootprintInvalid(
            "usage enrichment requires internal_open_weight traces; "
            "behavioural_remote evidence exposes no internal components"
        )
    target_mass: Dict[str, float] = {}
    control_mass: Dict[str, float] = {}
    target_cases: Dict[str, int] = {}
    control_cases: Dict[str, int] = {}
    for trace in traces:
        group = trace.group
        usage = _component_usage(trace.internal)
        mass_map = target_mass if group == "target" else control_mass
        case_map = target_cases if group == "target" else control_cases
        for component, value in usage.items():
            mass_map[component] = mass_map.get(component, 0.0) + value
            case_map[component] = case_map.get(component, 0) + 1
    components: Dict[str, Dict[str, Any]] = {}
    for component in sorted(set(target_mass) | set(control_mass)):
        t = target_mass.get(component, 0.0)
        c = control_mass.get(component, 0.0)
        t_seen = target_cases.get(component, 0)
        c_seen = control_cases.get(component, 0)
        if t_seen == 0 and c_seen == 0:
            # No observations in either group: recorded by omission, never
            # as a zero (unknown is not zero importance).
            continue
        entry: Dict[str, Any] = {
            "target_usage_mass": t,
            "control_usage_mass": c,
            "target_cases": t_seen,
            "control_cases": c_seen,
            "enrichment_log_ratio": math.log((t + _EPS) / (c + _EPS)),
            "correlation_only": True,
        }
        if c_seen == 0:
            entry["control_unobserved"] = True
        components[component] = entry
    return {
        "components": components,
        "target_traces": sum(1 for t in traces if t.group == "target"),
        "control_traces": sum(1 for t in traces if t.group == "control"),
    }


def build_footprint(
    *,
    capability: str,
    teacher: str,
    teacher_revision: str,
    spec_fingerprint: str,
    evidence_class: str,
    target_cases: int,
    control_cases: int,
    layers: Optional[Dict[str, Dict[str, Any]]] = None,
    experts: Optional[Dict[str, Dict[str, Any]]] = None,
    causal_interventions: Optional[Dict[str, InterventionEntry]] = None,
    confidence: Optional[Dict[str, Any]] = None,
    extra_limitations: Optional[List[str]] = None,
) -> CapabilityFootprint:
    """Assemble a schema-valid footprint with mandatory limitations.

    The limitation list is not optional decoration: the correlation
    discipline sentence is ALWAYS included, a behavioural-evidence sentence
    is added when the evidence class is remote, and the sample-size sentence
    is always included. A footprint without limitations is invalid on its
    face (schema-enforced).
    """
    if evidence_class not in (TRACE_CLASS_BEHAVIOURAL, TRACE_CLASS_INTERNAL):
        raise FootprintInvalid("unknown evidence class: %r" % evidence_class)
    if target_cases < 0 or control_cases < 0:
        raise FootprintInvalid("case counts must be non-negative")
    limitations: List[str] = [_CORRELATION_LIMITATION, _SAMPLE_SIZE_LIMITATION]
    if evidence_class == TRACE_CLASS_BEHAVIOURAL:
        limitations.insert(1, _BEHAVIOURAL_LIMITATION)
    if extra_limitations:
        limitations.extend(extra_limitations)
    return CapabilityFootprint(
        capability=capability,
        teacher=teacher,
        teacher_revision=teacher_revision,
        spec_fingerprint=spec_fingerprint,
        evidence=FootprintEvidence(
            trace_class=evidence_class,
            target_cases=target_cases,
            control_cases=control_cases,
        ),
        layers=layers or {},
        experts=experts or {},
        causal_interventions=causal_interventions or {},
        confidence=confidence or {},
        limitations=limitations,
    )


def validate_footprint_payload(raw: Any) -> CapabilityFootprint:
    """Validate an already-parsed footprint payload."""
    if not isinstance(raw, dict) or raw.get("schema") != CAPABILITY_FOOTPRINT_SCHEMA:
        raise FootprintInvalid(
            "footprint must carry schema '%s'" % CAPABILITY_FOOTPRINT_SCHEMA
        )
    try:
        return CapabilityFootprint.model_validate(raw)
    except ValueError as exc:
        raise FootprintInvalid("footprint validation failed: %s" % exc) from exc