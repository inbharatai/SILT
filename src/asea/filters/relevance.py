"""Relevance filter.

Drops signals that cannot help. Four independent reasons, each recorded by name
so the audit trail explains every discard:

  * ``sender_incorrect``  -- the sender missed the reference. Transferring this
    is how a teacher's mistakes become a student's beliefs.
  * ``receiver_competent`` -- the receiver already handles this case. Adding it
    is noise that dilutes retrieval and inflates apparent gains.
  * ``no_delta`` -- sender and receiver produced equivalent output.
  * ``duplicate`` -- content-identical to a signal already kept in this batch.

Note the asymmetry: we require the sender to be *right*, not merely confident.
Self-reported confidence is treated as a weak tiebreaker only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..core.interfaces import ModuleAdapter, SimilarityBackend
from ..core.protocol import PromotionStatus, SkillPacket
from ..evaluator.similarity import LexicalSimilarity


class RelevancePolicy:
    def __init__(
        self,
        sender_correctness_floor: float = 0.75,
        receiver_competence_ceiling: float = 0.85,
        min_delta: float = 0.05,
    ) -> None:
        self.sender_correctness_floor = sender_correctness_floor
        self.receiver_competence_ceiling = receiver_competence_ceiling
        self.min_delta = min_delta


class RelevanceFilter:
    def __init__(
        self,
        policy: Optional[RelevancePolicy] = None,
        similarity: Optional[SimilarityBackend] = None,
    ) -> None:
        self.policy = policy or RelevancePolicy()
        self.similarity = similarity or LexicalSimilarity()

    def apply(
        self, packets: List[SkillPacket], receiver: ModuleAdapter
    ) -> Tuple[List[SkillPacket], List[SkillPacket]]:
        kept: List[SkillPacket] = []
        dropped: List[SkillPacket] = []
        seen_hashes = set()

        for packet in packets:
            reason = self._reject_reason(packet, receiver, seen_hashes)
            if reason is None:
                packet.promotion_status = PromotionStatus.FILTERED
                seen_hashes.add(self._signal_hash(packet))
                kept.append(packet)
            else:
                packet.rejection_reason = reason
                packet.promotion_status = PromotionStatus.REJECTED
                dropped.append(packet)
        return kept, dropped

    # -- internals --------------------------------------------------------

    def _signal_hash(self, packet: SkillPacket) -> str:
        prompt = str(packet.notes.get("prompt", ""))
        return "{}|{}".format(packet.sender_capability.as_str(), prompt.strip().casefold())

    def _reject_reason(
        self, packet: SkillPacket, receiver: ModuleAdapter, seen: set
    ) -> Optional[str]:
        reference = packet.notes.get("reference")
        prompt = packet.notes.get("prompt")
        sender_output = packet.sender_output

        if self._signal_hash(packet) in seen:
            return "duplicate: identical probe already retained in this batch"

        if reference is not None:
            sender_correct = self.similarity.similarity(str(reference), str(sender_output))
            packet.notes["sender_correctness"] = round(sender_correct, 4)
            if sender_correct < self.policy.sender_correctness_floor:
                return (
                    "sender_incorrect: sender matched the reference at {:.2f}, "
                    "below floor {:.2f}".format(
                        sender_correct, self.policy.sender_correctness_floor
                    )
                )

        receiver_output = receiver.infer(packet.sender_capability, prompt)
        packet.notes["receiver_baseline_output"] = receiver_output

        if reference is not None:
            receiver_correct = self.similarity.similarity(
                str(reference), str(receiver_output)
            )
            packet.notes["receiver_correctness"] = round(receiver_correct, 4)
            if receiver_correct >= self.policy.receiver_competence_ceiling:
                return (
                    "receiver_competent: receiver already scores {:.2f} on this "
                    "case".format(receiver_correct)
                )

        agreement = self.similarity.similarity(str(sender_output), str(receiver_output))
        if (1.0 - agreement) < self.policy.min_delta:
            return (
                "no_delta: sender and receiver outputs agree at {:.2f}; nothing "
                "to teach".format(agreement)
            )

        return None
