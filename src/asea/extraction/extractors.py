"""Modality-specific extractors.

An extractor probes the sender on the negotiated gap set and records what came
back. It deliberately does *no* judging -- relevance and safety are separate
stages so that every discarded signal is discarded by a named, auditable filter
rather than silently inside extraction.

Every packet leaves here with status EXTRACTED and an empty ``distilled_skill``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core.errors import ExtractionError
from ..core.interfaces import Extractor, ModuleAdapter
from ..core.protocol import (
    Domain,
    Gap,
    Modality,
    OriginKind,
    PromotionStatus,
    Provenance,
    SkillPacket,
)


class BaseExtractor(Extractor):
    """Shared probing loop. Subclasses only describe their modality."""

    modality: Modality = Modality.TEXT

    def extract(
        self,
        sender: ModuleAdapter,
        receiver: ModuleAdapter,
        gap: Gap,
        probes: List[Dict[str, Any]],
    ) -> List[SkillPacket]:
        if not probes:
            raise ExtractionError(
                "no probes supplied for capability '{}'".format(gap.capability.as_str())
            )
        capability = gap.capability
        if capability.modality != self.modality:
            raise ExtractionError(
                "extractor for '{}' received capability of modality '{}'".format(
                    self.modality.value, capability.modality.value
                )
            )

        packets: List[SkillPacket] = []
        for probe in probes:
            prompt = probe.get("prompt")
            expected = probe.get("expected")
            output = sender.infer(capability, prompt)

            provenance = Provenance(
                origin_kind=(
                    OriginKind.CURATED_CORPUS
                    if probe.get("meta", {}).get("human_verified")
                    else OriginKind.MODEL_GENERATED
                ),
                chain=[sender.module_id],
                # A model producing the content adds one synthetic generation.
                synthetic_depth=0 if probe.get("meta", {}).get("human_verified") else 1,
                is_mock=sender.is_mock,
                source_reference=probe.get("source_reference"),
            )

            packet = SkillPacket(
                task_type=capability.task_type,
                source_module=sender.module_id,
                target_module=receiver.module_id,
                sender_capability=capability,
                receiver_gap=gap,
                modality=capability.modality,
                language=capability.language,
                domain=capability.domain,
                raw_input_reference=probe.get("case_id"),
                sender_output=output,
                confidence_score=sender.confidence(capability, prompt, output),
                provenance=provenance,
                promotion_status=PromotionStatus.EXTRACTED,
                notes=self.annotate(probe, output, expected),
            )
            packets.append(packet)
        return packets

    def annotate(
        self, probe: Dict[str, Any], output: Any, expected: Any
    ) -> Dict[str, Any]:
        """Modality hook for extra bookkeeping carried alongside the packet."""
        return {
            "prompt": probe.get("prompt"),
            "reference": expected,
            "extractor": type(self).__name__,
        }


class TextExtractor(BaseExtractor):
    """Translation, glossary and general text competence."""

    modality = Modality.TEXT


class TTSExtractor(BaseExtractor):
    """Pronunciation and prosody.

    Note the honest scope limit: what is extractable between arbitrary TTS
    systems is the *symbolic* layer -- grapheme-to-phoneme mappings, lexicon
    entries, stress and schwa-deletion rules. Acoustic identity (voice timbre,
    learned prosody embeddings) is entangled with the acoustic model and vocoder
    and is NOT transferable through this adapter. See docs/feasibility_review.md.
    """

    modality = Modality.SPEECH_TTS

    def annotate(self, probe, output, expected):
        base = super().annotate(probe, output, expected)
        base["symbolic_only"] = True
        base["grapheme"] = probe.get("prompt")
        return base


class CodeExtractor(BaseExtractor):
    """Bug-fix and test-passing patterns."""

    modality = Modality.CODE

    def annotate(self, probe, output, expected):
        base = super().annotate(probe, output, expected)
        base["test_command"] = probe.get("meta", {}).get("test_command")
        return base


class StructuredExtractor(BaseExtractor):
    """Rule-shaped domain knowledge (triage red flags, policy rules)."""

    modality = Modality.STRUCTURED

    def annotate(self, probe, output, expected):
        base = super().annotate(probe, output, expected)
        base["rule_category"] = probe.get("meta", {}).get("category")
        return base
