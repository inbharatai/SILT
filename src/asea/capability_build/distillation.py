"""Neural student distillation for capability-build (brief §P, §Q, §R).

This is a SEPARATE path from ``asea.distill.strategies`` -- the existing
text packet distillers remain the L3 skill-packet mechanism, untouched.
This module builds the data side of sequence-level knowledge distillation
from teacher traces and hands actual training to DeepApply (LoRA through
Gate 2); it never trains anything itself and never replaces the existing
distillers.

What sequence-level KD means here, honestly:

  * Training pairs come ONLY from teacher traces whose outcome was judged
    successful by the host oracle (or whose repair was verified). Failed
    teacher attempts are kept as negative examples, clearly labelled.
  * Cross-family (GLM -> Qwen) means NO vocabulary or logit compatibility
    is assumed. Full-vocab KL distillation is only permitted where the
    recovery path's tokenizer-compatibility gates would pass; for cross-
    family pairs this build emits SEQUENCE-LEVEL pairs only (prompt ->
    response text), and says so.
  * Leakage discipline is inherited verbatim from the specialist recovery
    path: no case may appear in training and development/held-out/final
    splits; content hashes, family IDs, and tokenized-content overlap are
    checked BEFORE any training data is emitted.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence

from .errors import CapabilityBuildError
from .schema import CapabilityTrace, EvaluationSplits

_PAIR_SCHEMA = "silt.capability_kd_pairs.v1"


class LeakageError(CapabilityBuildError):
    """A training pair overlaps a protected split (ID, exact content,
    family, or tokenized-content collision). Never silently dropped --
    the whole build is refused."""


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _token_shape(text: str) -> str:
    """Coarse tokenized-content shape: lowercased whitespace/alphanumeric
    stream -- catches near-duplicate leakage that exact-hash checks miss.
    Deliberately cheap; it is a guard, not a tokenizer."""
    import re

    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def build_sequence_pairs(
    traces: Sequence[CapabilityTrace],
    *,
    protected_content_hashes: Optional[set] = None,
    protected_sample_ids: Optional[set] = None,
    protected_families: Optional[set] = None,
    protected_prompts: Optional[set] = None,
    sample_families: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Build sequence-level KD pairs from JUDGED behavioural traces.

    Only successful-outcome traces become positive pairs; failures are
    retained as labelled negatives (failed experiments stay visible). Any
    overlap with the protected splits raises :class:`LeakageError` and
    refuses the WHOLE pair set -- one collision poisons the build.

    Protected means: sample ids, full content hashes, PROMPT hashes,
    and FAMILIES of every non-training split. ``protected_families`` is
    enforced, not advisory: when it is supplied, every trace's family must
    be resolvable through ``sample_families`` (a trace whose family cannot
    be verified is refused -- unverifiable is not safe), and a trace whose
    family belongs to a protected split raises :class:`LeakageError`.
    ``protected_prompts`` are sha256 hex digests of protected prompts and
    catch exact prompt reuse even when the response differs.
    """
    protected_content_hashes = protected_content_hashes or set()
    protected_sample_ids = protected_sample_ids or set()
    protected_families = protected_families or set()
    protected_prompts = protected_prompts or set()
    if protected_families and sample_families is None:
        raise CapabilityBuildError(
            "protected families were supplied without a sample->family "
            "lookup; family protection cannot be verified and the build is "
            "refused (unverifiable is not safe)"
        )
    positives: List[Dict[str, Any]] = []
    negatives: List[Dict[str, Any]] = []
    seen_hashes: set = set()
    seen_shapes: set = set()
    for trace in traces:
        if trace.behavioural is None:
            # Internal traces are instrumentation, not teaching material.
            continue
        prompt = trace.behavioural.prompt
        response = trace.behavioural.response
        content = _content_hash(prompt + "\x00" + response)
        if trace.sample_id in protected_sample_ids:
            raise LeakageError(
                "trace sample %r appears in a protected split" % trace.sample_id
            )
        if content in protected_content_hashes:
            raise LeakageError(
                "trace content hash collides with a protected split at "
                "sample %r" % trace.sample_id
            )
        if _content_hash(prompt) in protected_prompts:
            raise LeakageError(
                "trace prompt at sample %r is the exact prompt of a "
                "protected-split case" % trace.sample_id
            )
        if protected_families:
            family = (sample_families or {}).get(trace.sample_id)
            if family is None:
                raise LeakageError(
                    "family of trace sample %r cannot be resolved; a trace "
                    "from outside the approved dataset is never teaching "
                    "material" % trace.sample_id
                )
            if family in protected_families:
                raise LeakageError(
                    "trace sample %r belongs to family %r, which is "
                    "protected (a family never crosses splits)"
                    % (trace.sample_id, family)
                )
        if content in seen_hashes:
            raise LeakageError("duplicate training pair at sample %r" % trace.sample_id)
        shape = _token_shape(prompt)
        if shape in seen_shapes and shape:
            raise LeakageError(
                "near-duplicate training prompt (token-shape collision) at "
                "sample %r" % trace.sample_id
            )
        seen_hashes.add(content)
        seen_shapes.add(shape)
        pair = {
            "sample_id": trace.sample_id,
            "group": trace.group,
            "prompt": prompt,
            "response": response,
            "source": "behavioural_remote",
        }
        if trace.outcome.success is True:
            positives.append(pair)
        elif trace.outcome.success is False:
            negatives.append(pair)
        # success is None (UNJUDGED) traces contribute NOTHING: unjudged
        # teacher output must never become teaching material.
    return {
        "schema": _PAIR_SCHEMA,
        "positives": positives,
        "negatives": negatives,
        "unjudged_excluded": sum(
            1 for t in traces
            if t.behavioural is not None and t.outcome.success is None
        ),
        "sequence_level_only": True,
        "compatibility_note": (
            "cross-family (GLM->Qwen) pairs carry text only; no vocabulary, "
            "logit or hidden-state alignment is assumed or performed"
        ),
        "leakage_checks": [
            "sample_id disjointness", "exact content hash",
            "protected prompt hash", "protected family",
            "token-shape near-duplicate", "within-set duplicate",
        ],
    }


def hand_to_deepapply(pairs: Dict[str, Any]) -> Dict[str, Any]:
    """Describe the DeepApply hand-off (the ONLY training path: LoRA via
    DeepApply's TrainerBackend, intake gated by Gate 2). This build wires
    the data contract; the training run itself is a DeepApply invocation
    that this package never bypasses or shortcuts.

    Returns the hand-off descriptor. DeepApply decides admission; nothing
    here pre-approves anything.
    """
    if pairs.get("schema") != _PAIR_SCHEMA:
        raise CapabilityBuildError("pairs payload is not a KD pair set")
    if not pairs.get("positives"):
        raise CapabilityBuildError(
            "no positive pairs: distillation has nothing to teach "
            "(transfer without a measured gap is unjustified)"
        )
    return {
        "sink": "asea.deepapply.DeepApplyRunner.from_pipeline",
        "intake": "Gate 2 (L4-only, re-enforces Gate 1)",
        "training": "LoRA via TrainerBackend (standard/streamed/zeroforge)",
        "evaluation": "independent held-out A/B by Gate 2 -- the trainer "
                      "never certifies itself",
        "pairs": {
            "positives": len(pairs["positives"]),
            "negatives": len(pairs["negatives"]),
            "sequence_level_only": pairs["sequence_level_only"],
        },
        "pre_approval": False,
        "autoactivated": False,
    }