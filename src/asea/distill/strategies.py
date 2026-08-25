"""Distillation strategies.

Distillation here means *compression into a reusable, inspectable payload* --
not weight surgery. Many extracted signals collapse into one packet carrying a
glossary, a lexicon, a set of exemplars or a rule list.

Two invariants every strategy upholds:

  1. The emitted packet's ``sender_output`` is ``None``. Raw model text does not
     travel to the receiver; only the curated ``distilled_skill`` does. The raw
     signals remain in the candidate store for audit.
  2. Where a human-verified reference exists it is preferred over the sender's
     own output as the taught value. The sender is used to *locate* the gap and
     to supply value only when no reference exists.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core.errors import DistillationError
from ..core.interfaces import Distiller
from ..core.protocol import (
    Modality,
    OriginKind,
    PacketType,
    PromotionStatus,
    Provenance,
    SkillPacket,
)


class BaseDistiller(Distiller):
    modality: Modality = Modality.TEXT
    packet_type: PacketType = PacketType.EXEMPLAR

    #: Max entries per emitted packet; oversized payloads defeat retrieval.
    max_entries: int = 200

    def distill(self, packets: List[SkillPacket]) -> List[SkillPacket]:
        if not packets:
            return []
        groups: Dict[str, List[SkillPacket]] = {}
        for p in packets:
            if p.modality != self.modality:
                raise DistillationError(
                    "{} received a '{}' packet".format(
                        type(self).__name__, p.modality.value
                    )
                )
            groups.setdefault(p.sender_capability.as_str(), []).append(p)

        return [self._emit(group) for group in groups.values()]

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def taught_value(packet: SkillPacket) -> Any:
        """Prefer the verified reference; fall back to the sender's output."""
        reference = packet.notes.get("reference")
        return reference if reference is not None else packet.sender_output

    def _merge_provenance(self, packets: List[SkillPacket]) -> Provenance:
        chain: List[str] = []
        for p in packets:
            for hop in p.provenance.chain:
                if hop not in chain:
                    chain.append(hop)
        origins = {p.provenance.origin_kind for p in packets}
        # Weakest link wins: one model-generated member makes the whole packet
        # model-generated.
        if OriginKind.MODEL_GENERATED in origins:
            origin = OriginKind.MODEL_GENERATED
        elif OriginKind.CURATED_CORPUS in origins:
            origin = OriginKind.CURATED_CORPUS
        else:
            origin = OriginKind.HUMAN_VERIFIED
        return Provenance(
            origin_kind=origin,
            chain=chain,
            synthetic_depth=max(p.provenance.synthetic_depth for p in packets),
            is_mock=any(p.provenance.is_mock for p in packets),
            source_reference=next(
                (p.provenance.source_reference for p in packets
                 if p.provenance.source_reference),
                None,
            ),
        )

    def _emit(self, group: List[SkillPacket]) -> SkillPacket:
        head = group[0]
        payload = self.build_payload(group)
        confidence = sum(p.confidence_score for p in group) / len(group)

        return SkillPacket(
            task_type=head.task_type,
            source_module=head.source_module,
            target_module=head.target_module,
            sender_capability=head.sender_capability,
            receiver_gap=head.receiver_gap,
            modality=head.modality,
            language=head.language,
            domain=head.domain,
            raw_input_reference=",".join(
                str(p.raw_input_reference) for p in group if p.raw_input_reference
            )
            or None,
            sender_output=None,  # invariant 1: raw output does not travel
            packet_type=self.packet_type,
            distilled_skill=payload,
            confidence_score=confidence,
            safety_score=min(
                [p.safety_score for p in group if p.safety_score is not None] or [1.0]
            ),
            provenance=self._merge_provenance(group),
            learning_level=head.learning_level,
            promotion_status=PromotionStatus.DISTILLED,
            notes={
                "member_packet_ids": [p.packet_id for p in group],
                "member_count": len(group),
                "distiller": type(self).__name__,
            },
        )

    def build_payload(self, group: List[SkillPacket]) -> Dict[str, Any]:
        raise NotImplementedError


class TextDistiller(BaseDistiller):
    """Emits a glossary of source->target mappings for text and translation."""

    modality = Modality.TEXT
    packet_type = PacketType.GLOSSARY

    def build_payload(self, group: List[SkillPacket]) -> Dict[str, Any]:
        entries = []
        seen = set()
        for p in group[: self.max_entries]:
            source = p.notes.get("prompt")
            target = self.taught_value(p)
            key = str(source).strip().casefold()
            if not source or target is None or key in seen:
                continue
            seen.add(key)
            entries.append(
                {
                    "source": source,
                    "target": target,
                    "confidence": round(p.confidence_score, 4),
                }
            )
        return {
            "entries": entries,
            "usage": "exact-match lookup, then fall back to base behaviour",
            "language": group[0].language,
        }


class TTSDistiller(BaseDistiller):
    """Emits a pronunciation lexicon.

    SCOPE LIMIT (stated in the payload itself so downstream consumers see it):
    symbolic only. Grapheme-to-phoneme entries and stress rules transfer; voice
    timbre and learned prosody embeddings do not.
    """

    modality = Modality.SPEECH_TTS
    packet_type = PacketType.LEXICON

    def build_payload(self, group: List[SkillPacket]) -> Dict[str, Any]:
        entries = []
        seen = set()
        for p in group[: self.max_entries]:
            grapheme = p.notes.get("grapheme") or p.notes.get("prompt")
            phoneme = self.taught_value(p)
            key = str(grapheme).strip().casefold()
            if not grapheme or phoneme is None or key in seen:
                continue
            seen.add(key)
            entries.append(
                {
                    "grapheme": grapheme,
                    "phoneme": phoneme,
                    "confidence": round(p.confidence_score, 4),
                }
            )
        return {
            "entries": entries,
            "scope": "symbolic_g2p_only",
            "not_transferable": ["voice_timbre", "acoustic_prosody_embeddings"],
            "language": group[0].language,
        }


class CodeDistiller(BaseDistiller):
    """Emits bug-fix exemplars keyed by failing pattern."""

    modality = Modality.CODE
    packet_type = PacketType.EXEMPLAR

    def build_payload(self, group: List[SkillPacket]) -> Dict[str, Any]:
        examples = []
        seen = set()
        for p in group[: self.max_entries]:
            buggy = p.notes.get("prompt")
            fixed = self.taught_value(p)
            key = str(buggy).strip()
            if not buggy or fixed is None or key in seen:
                continue
            seen.add(key)
            examples.append(
                {
                    "buggy": buggy,
                    "fixed": fixed,
                    "test_command": p.notes.get("test_command"),
                    "verified_by_tests": bool(p.notes.get("test_command")),
                    "confidence": round(p.confidence_score, 4),
                }
            )
        return {
            "examples": examples,
            "usage": "retrieve nearest failing pattern before generating a fix",
        }


class StructuredDistiller(BaseDistiller):
    """Emits explicit rules (triage red flags, policy constraints)."""

    modality = Modality.STRUCTURED
    packet_type = PacketType.RULE

    def build_payload(self, group: List[SkillPacket]) -> Dict[str, Any]:
        rules = []
        seen = set()
        for p in group[: self.max_entries]:
            condition = p.notes.get("prompt")
            action = self.taught_value(p)
            key = str(condition).strip().casefold()
            if not condition or action is None or key in seen:
                continue
            seen.add(key)
            rules.append(
                {
                    "condition": condition,
                    "action": action,
                    "category": p.notes.get("rule_category"),
                    "confidence": round(p.confidence_score, 4),
                }
            )
        return {
            "rules": rules,
            "usage": "deterministic rule match; never override with generation",
        }
