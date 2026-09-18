"""MoEArchitectureAdapter -- the adapter contract for teacher instrumentation.

Every teacher interaction for the ``internal_open_weight`` evidence class
goes through an adapter. The contract is deliberately narrow and mirrors
what the compiler already proved safe:

  * ``inspect_model`` / ``parameter_inventory`` are read-only (safetensors
    headers or in-memory metadata; the compiler never pickles weights and
    neither does an adapter).
  * ``register_router_hooks`` / ``collect_routing`` / ``collect_expert_outputs``
    reproduce the compiler's ``RoutingTelemetry`` semantics: analytic-FP32
    probability mass vs native-postcast top-1 vs post-capacity dispatched
    count, REAP conditional-mean expert outputs, zero observations = null
    ("unknown is not zero importance").
  * ``temporary_mask`` / ``temporary_replace`` / ``restore_mask`` /
    ``verify_unchanged`` implement the intervention protocol
    (:mod:`asea.capability_build.intervention`): the ONLY paths that touch
    live tensors, always paired with hash verification so the teacher ends
    byte-identical to how it started. Teacher weights are read-only in
    aggregate: an adapter that cannot prove restoration must raise, not
    shrug.
  * ``structural_reduce`` is the physical-reduction entry point for the
    reduction search -- it produces a CANDIDATE artifact, never an
    admission.

Adapters never claim causal importance from usage telemetry; that
discipline lives in the compiler and is restated in every artifact this
package emits.

This module is stdlib-only (bare ``import asea`` sanity gate); concrete
adapters lazy-import torch/transformers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class MoEArchitectureAdapter(ABC):
    """Read-only instrumentation + reversible intervention on one MoE model."""

    #: Architecture identity, e.g. ``"switch_transformers"`` or
    #: ``"glm5_next_moe"``. Used for exact-architecture detection.
    arch: str = "unknown"

    @abstractmethod
    def inspect_model(self) -> Dict[str, Any]:
        """Read-only architecture inspection: layer counts, expert counts,
        experts-per-token, shared experts, dense-vs-sparse layer map,
        router scoring function, hidden size, dtype, position capacity,
        quantization state. Must succeed BEFORE any behavior is allowed;
        a mismatch with the expected architecture raises a typed error."""

    @abstractmethod
    def parameter_inventory(self) -> Dict[str, Any]:
        """Full per-component parameter accounting (counts by module kind,
        total, trainable-vs-frozen). Numbers only; no quality claims."""

    @abstractmethod
    def enumerate_sparse_layers(self) -> List[int]:
        """Indices of MoE (sparse MLP) layers; dense layers are excluded."""

    @abstractmethod
    def register_router_hooks(self) -> None:
        """Install forward hooks on every router (and, for REAP-style
        expert-output collection, per-expert hooks). Idempotent: a second
        call must not double-count."""

    @abstractmethod
    def collect_routing(self, reset: bool = True) -> Dict[str, Any]:
        """Accumulated routing telemetry since the last reset, keyed by
        layer: analytic-FP32 probability mass, native-postcast top-1,
        post-capacity dispatched counts. ``reset=True`` clears after
        reading."""

    @abstractmethod
    def collect_expert_outputs(self, reset: bool = True) -> Dict[str, Any]:
        """REAP-style per-expert output statistics (gate-weighted
        conditional mean L2). Zero observations = null, never zero."""

    # -- intervention protocol ------------------------------------------------

    @abstractmethod
    def temporary_mask(
        self, target, *, scale: float = 0.0
    ) -> Dict[str, Any]:
        """Reversibly disable one :class:`InterventionTarget` in the LIVE
        model (router-expert masking or whole-layer bypass per kind).
        Returns a handle describing exactly what was changed so
        ``restore_mask`` can prove reversal. No disk writes."""

    @abstractmethod
    def restore_mask(self, target) -> Dict[str, Any]:
        """Undo exactly the change recorded by ``temporary_mask``. Must
        restore original parameters/tensors; call ``verify_unchanged``
        afterwards to prove it."""

    @abstractmethod
    def verify_unchanged(self) -> Dict[str, Any]:
        """Hash the live parameters against the hashes recorded at
        ``freeze_baseline`` (called automatically on first use) and return
        ``{"unchanged": bool, "detail": ...}``. The teacher is read-only in
        aggregate: a False here aborts every dependent operation."""

    # -- reduction -----------------------------------------------------------

    @abstractmethod
    def structural_reduce(self, keep_components) -> Dict[str, Any]:
        """Produce a structurally reduced CANDIDATE model artifact keeping
        only ``keep_components`` (e.g. a set of expert indices per layer).
        Writes a NEW artifact path; NEVER mutates the teacher and NEVER
        claims quality -- evaluation is downstream and external."""


def adapter_for_arch(arch: str) -> Optional[str]:
    """Map an architecture identity to the adapter implementing it. Used by
    the worker to refuse unknown architectures with a typed error instead of
    guessing."""
    return {
        "switch_transformers": "asea.capability_build.adapters.switch.SwitchAdapter",
        "glm5_next_moe": "asea.capability_build.adapters.glm53_flash.Glm53FlashAdapter",
    }.get(arch)