"""Abstract interfaces every pluggable component implements.

The universal core talks only to these ABCs. Swapping a mock Qwen for a real
Qwen means writing one subclass of :class:`ModuleAdapter`; nothing in the core,
the filters, the gate or the audit layer changes.
"""

from __future__ import annotations

import abc
import hashlib
from typing import Any, Dict, List, Optional

from .protocol import (
    CapabilityKey,
    CapabilityManifest,
    Gap,
    Modality,
    SkillPacket,
)


class ModuleAdapter(abc.ABC):
    """A model, corpus or service plugged into the adapter.

    Implementations must be honest about ``is_mock``. That flag rides into every
    packet's provenance and blocks promotion under strict policy, which is the
    whole point: a mock must never be able to launder itself into approved
    learning data.
    """

    #: Subclasses that do not perform real inference MUST set this True.
    is_mock: bool = False

    def __init__(self, module_id: str, display_name: Optional[str] = None) -> None:
        self.module_id = module_id
        self.display_name = display_name or module_id

    # -- identity ---------------------------------------------------------

    @abc.abstractmethod
    def manifest(self) -> CapabilityManifest:
        """Publish what this module claims it can do."""

    def fingerprint(self) -> str:
        """A stable identity string for this module.

        Used to key caches of a module's *deterministic* outputs -- today, the
        teacher's extraction scores during gap negotiation (A1, audit
        2026-08-17). The default is a sha256 over the manifest's identity
        fields (module_id, version, sorted capabilities, max learning level,
        is_mock). That is correct for the bundled mocks and any module whose
        outputs are a pure function of its manifest.

        Real HF connectors SHOULD override this to also hash the model path,
        revision and quantization config, so that re-quantizing or swapping a
        checkpoint invalidates any score cached against the old weights. The
        default deliberately does NOT look at weights: a manifest has no way
        to see them, and pretending it does would be worse than admitting the
        limitation here.

        Validity contract (binding): this is a correct cache key ONLY under
        the SILT deterministic-decoding discipline (``do_sample=False``). A
        module whose outputs are NOT a pure function of (fingerprint, suite)
        must not be cached -- override ``fingerprint`` to be unique per
        non-deterministic run, or leave the teacher cache off. Serving a
        cached score computed under different decoding is a silent
        correctness regression and is exactly what the key is meant to
        prevent.
        """
        m = self.manifest()
        cap = ",".join(sorted(c.as_str() for c in m.capabilities))
        raw = "|".join(
            [
                self.module_id,
                str(getattr(m, "version", "0")),
                cap,
                str(getattr(m, "max_learning_level", "")),
                str(bool(getattr(m, "is_mock", False))),
            ]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    # -- behaviour --------------------------------------------------------

    @abc.abstractmethod
    def infer(self, capability: CapabilityKey, prompt: Any) -> Any:
        """Produce output for one probe input.

        For a receiver this is the *baseline* behaviour with no injected skill.
        """

    #: Largest number of cases this module wants pushed through a single
    #: forward pass. Default 1 -> the harness loops ``infer`` (byte-identical to
    #: the pre-A2 behaviour, and the only safe default for a 4-bit 7B on an
    #: 8 GB card where two cases co-resident OOM). Small CPU models and
    #: generously-VRAM'd real connectors override this higher to batch cases and
    #: amortise the forward pass. The harness caps any effective batch at its
    #: own ``max_batch_size`` AND at this per-module value, so a module cannot
    #: be forced to batch beyond what it declared safe.
    preferred_batch_size: int = 1

    def infer_batch(self, capability: CapabilityKey, prompts: List[Any]) -> List[Any]:
        """Produce outputs for a batch of probes in one call.

        Default implementation loops :meth:`infer` -- bit-identical results to
        the pre-A2 single-case path, so a module that does not implement a real
        batched forward still gets CORRECT answers (it simply gets no
        speedup). A real connector overrides this with a single batched forward
        pass; it MUST preserve case order (output ``i`` answers ``prompts[i]``)
        and MUST raise a typed error (the harness wraps any exception in
        :class:`asea.core.errors.BatchedInferenceError`) rather than silently
        truncating or dropping cases. Returning the wrong number of outputs is a
        correctness bug, not a perf one -- the harness asserts the count matches.

        LOAD-BEARING ORDER OBLIGATION: the harness enforces COUNT (wrong count
        -> :class:`~asea.core.errors.InferenceCountMismatchError`) but it
        CANNOT enforce ORDER. Verifying that output ``i`` actually answers
        ``prompts[i]`` requires ground-truth labels, which would defeat the
        held-out split the whole evaluator rests on. So order preservation is
        an obligation on THIS method, not a check the harness can perform. A
        real connector that reorders internally for efficiency (length-sorted
        batching, hash-bucketed continuous batching, an unsorted
        ``async.gather``) MUST un-sort back to prompt order before returning;
        if it does not, the harness will silently attach every output to the
        wrong ``case_id`` and every downstream score and gate will be
        corrupted with no error raised. This boundary is pinned (not hidden) by
        ``tests/test_batched_inference.py::test_order_preservation_is_a_module_obligation_*``.
        """
        return [self.infer(capability, prompt) for prompt in prompts]

    def infer_with_skills_batch(
        self, capability: CapabilityKey, prompts: List[Any], skills: List[Dict[str, Any]]
    ) -> List[Any]:
        """Batched counterpart of :meth:`infer_with_skills`.

        Default loops :meth:`infer_with_skills` -- bit-identical to the single-
        case conditioned path. ``skills`` is the SAME payload for every prompt
        in the batch (a suite is evaluated under one packet's redacted skill), so
        a real connector overrides this with one batched, skill-conditioned
        forward pass over the whole chunk. Same order/count contract as
        :meth:`infer_batch`.
        """
        return [self.infer_with_skills(capability, prompt, skills) for prompt in prompts]

    def infer_with_skills(
        self, capability: CapabilityKey, prompt: Any, skills: List[Dict[str, Any]]
    ) -> Any:
        """Produce output while conditioned on approved skill payloads.

        ``skills`` are the outputs of :meth:`SkillPacket.redacted_for_receiver`.
        Default implementation ignores them, which correctly models a module
        that cannot consume skills (it will simply never show improvement and
        so will never pass the promotion gate).
        """
        return self.infer(capability, prompt)

    def confidence(self, capability: CapabilityKey, prompt: Any, output: Any) -> float:
        """Self-reported confidence in ``output``. Treated as a weak signal only."""
        return 0.5

    def supports(self, capability: CapabilityKey) -> bool:
        return self.manifest().supports(capability)


class Extractor(abc.ABC):
    """Turns sender behaviour on a gap set into raw candidate packets.

    Registered per-modality. This is the first half of the honest seam: the core
    orchestrates extraction identically for every modality, but *what* counts as
    an extractable signal is modality-specific.
    """

    modality: Modality

    @abc.abstractmethod
    def extract(
        self,
        sender: ModuleAdapter,
        receiver: ModuleAdapter,
        gap: Gap,
        probes: List[Dict[str, Any]],
    ) -> List[SkillPacket]:
        """Probe the sender and emit EXTRACTED-status packets."""


class Distiller(abc.ABC):
    """Compresses filtered packets into a reusable payload.

    Second half of the seam: a pronunciation lexicon and a bug-fix pattern
    cannot share a compression algorithm, and pretending otherwise would be
    dishonest. They do share the envelope, the gate and the audit trail.
    """

    modality: Modality

    @abc.abstractmethod
    def distill(self, packets: List[SkillPacket]) -> List[SkillPacket]:
        """Populate ``distilled_skill`` and ``packet_type``; set DISTILLED."""


class MetricPlugin(abc.ABC):
    """Optional modality-specific scoring, layered on the universal metrics."""

    modality: Modality

    @abc.abstractmethod
    def score(self, expected: Any, actual: Any, packet: SkillPacket) -> float:
        """Return task success in 0..1."""

    def fingerprint(self) -> str:
        """Stable identity for this metric, used to key caches of scores it
        produced (A1, audit 2026-08-17). Default = class identity. A metric whose
        INSTANCE config changes its score MUST override this to include that
        config, or a score cached under one configuration is silently served
        under another. The bundled metrics have no config, so the default is
        correct for them; it exists so a future configurable metric cannot
        forget to invalidate the cache."""
        return "{}.{}".format(type(self).__module__, type(self).__name__)


class SimilarityBackend(abc.ABC):
    """Pluggable similarity.

    The bundled implementation is lexical (edit distance + token F1). That is a
    weak proxy for meaning and is labelled as such everywhere it is used. Drop in
    an embedding model here to make it real.
    """

    @abc.abstractmethod
    def similarity(self, a: str, b: str) -> float:
        """Return 0..1."""

    @property
    def is_semantic(self) -> bool:
        """False for lexical proxies. Surfaced in reports so nobody is misled."""
        return False

    def fingerprint(self) -> str:
        """Stable identity for this backend, used to key caches of scores it
        produced (A1, audit 2026-08-17). Default = class identity + is_semantic.
        A backend whose INSTANCE config changes the score -- e.g.
        :class:`LexicalSimilarity`'s ``char_weight`` / ``token_weight`` -- MUST
        override this to include that config, or a score cached under one
        configuration is silently served under another (a stale-serve bug). The
        bundled :class:`LexicalSimilarity` overrides accordingly.
        """
        return "{}.{}|is_semantic={}".format(
            type(self).__module__, type(self).__name__, self.is_semantic
        )
