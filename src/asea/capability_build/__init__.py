"""Capability Build: teacher-side capability footprinting and student builds.

A research/build layer ON TOP of the existing SILT mechanisms -- never a
replacement. The universal Pipeline (L3 skill packets, Gate 1), DeepApply
(Gate 2) and SiltSpring (compressed-state certification) keep their contracts
unchanged; this package adds the experimental question the public brief asks:

    Given a large teacher model, what is the minimum computational artifact
    that reproduces a narrowly defined capability in a much smaller student,
    while proving on untouched held-out tests that the capability survived
    and unrelated capabilities did not materially regress?

Design rules carried from the rest of SILT:

  * The package imports with core dependencies only (pydantic). Heavy
    runtimes (torch, transformers) are lazy imports behind the same
    optional-extra discipline as ``asea.deepapply``.
  * Two evidence classes exist and are NEVER mixed: ``behavioural_remote``
    (what an API/cloud teacher legitimately exposes) and
    ``internal_open_weight`` (router/expert observations from locally
    controlled open weights). A trace, a footprint or a receipt always
    states which evidence class was actually obtained.
  * Routing usage is usage evidence, NOT causal expert importance -- the
    compiler's discipline (asea/compiler/core.py) is inherited verbatim;
    causal claims require the explicit intervention machinery in
    :mod:`asea.capability_build.intervention`.
  * Unmeasured values are the literal token ``NOT_MEASURED``; they are never
    estimated and never silently defaulted.
  * A resource that cannot run here is reported as ``BLOCKED_RESOURCE``
    naming the requirement and the remedy; never a silent skip, never a
    fabricated result.
  * Nothing in this package activates a deployment automatically.
"""

from __future__ import annotations

__all__ = [
    "NOT_MEASURED",
    "BLOCKED_RESOURCE",
    "TRACE_CLASS_BEHAVIOURAL",
    "TRACE_CLASS_INTERNAL",
    "CapabilitySpec",
    "CapabilityTrace",
    "CapabilityFootprint",
    "CapabilityBuildReceipt",
    "HONESTY_NOTE",
]


#: The single allowed marker for an unmeasured metric. Never estimated.
NOT_MEASURED = "NOT_MEASURED"

#: The single allowed marker for a run that could not obtain a required
#: resource (hardware, runtime, weights). Always accompanied by an
#: ``error`` naming the requirement and the remedy.
BLOCKED_RESOURCE = "BLOCKED_RESOURCE"

TRACE_CLASS_BEHAVIOURAL = "behavioural_remote"
TRACE_CLASS_INTERNAL = "internal_open_weight"

#: Carried verbatim in every signed artifact this package emits (same
#: contract as asea.capability_diff / asea.unlearning): the signature is a
#: LOCAL HMAC, tamper-evidence to the holder of the same local key, not a
#: portable third-party attestation.
HONESTY_NOTE = (
    "Signature is a local HMAC over the report bytes with a workspace-local "
    "key. It proves the report was not modified after generation to the "
    "holder of that same key. It is NOT a portable attestation, NOT proof "
    "of authorship to third parties, and NOT a quality certificate."
)


def _lazy_models():
    """Import the pydantic models on demand (keeps `import asea` light)."""
    from .schema import (
        CapabilityBuildReceipt,
        CapabilityFootprint,
        CapabilitySpec,
        CapabilityTrace,
    )
    return {
        "CapabilitySpec": CapabilitySpec,
        "CapabilityTrace": CapabilityTrace,
        "CapabilityFootprint": CapabilityFootprint,
        "CapabilityBuildReceipt": CapabilityBuildReceipt,
    }


def __getattr__(name):  # pragma: no cover - trivial lazy forwarding
    models = _lazy_models()
    if name in models:
        return models[name]
    raise AttributeError(name)