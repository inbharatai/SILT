"""Promoted packets -> a traceable training set.

This reuses the existing L4/L5 dataset flattening (asea.distill.export) so the
training data deep-apply consumes is byte-identical in provenance to the dataset
a human would take to an external trainer. Each row carries its source
``packet_id``; the :class:`AdapterPacket` records the full list.

Intake guards (binding):

* Only ``PROMOTED`` packets may enter a training set -- anything else raises
  :class:`DeepApplyIntakeError` with a named reason. This is Gate 1's verdict
  being enforced at the door of Gate 2.
* Under ``strict_no_mock`` (default), mock-provenance packets are refused -- a
  mock cannot launder itself into training data, same guard as
  ``export_dataset``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

from ..core.protocol import Domain, PromotionStatus, SkillPacket
from ..distill.export import _rows_from_packet  # same package tree; reuse, do not duplicate
from .errors import DeepApplyIntakeError


class TrainingDataset:
    """A flattened, hashed, traceable training set built from approved packets."""

    def __init__(
        self,
        rows: List[Dict[str, Any]],
        manifest: Dict[str, Any],
    ) -> None:
        self.rows = rows
        self.manifest = manifest

    @property
    def dataset_hash(self) -> str:
        return self.manifest["dataset_hash"]

    @property
    def source_packet_ids(self) -> List[str]:
        return list(self.manifest["source_packet_ids"])

    @property
    def synthetic_depth(self) -> int:
        return self.manifest["synthetic_depth_max"]

    @property
    def source_domains(self) -> List[Domain]:
        return [Domain(d) for d in self.manifest["source_domains"]]

    @property
    def contains_mock(self) -> bool:
        return self.manifest["contains_mock"]

    @property
    def safety_floor(self) -> float:
        """Min safety_score over source packets (the adapter inherits the weakest)."""
        return self.manifest["min_safety_score"]


def build_training_dataset(
    packets: List[SkillPacket],
    strict_no_mock: bool = True,
) -> TrainingDataset:
    """Build a traceable training set from packets.

    Refuses anything not PROMOTED (named reason) and, under ``strict_no_mock``,
    any mock-provenance packet. Returns the flattened rows plus a manifest
    carrying the dataset hash, source packet ids, propagated synthetic_depth
    (max), source domains, and the inherited safety floor (min).
    """
    accepted: List[SkillPacket] = []
    for packet in packets:
        if packet.promotion_status != PromotionStatus.PROMOTED:
            raise DeepApplyIntakeError(
                "refusing non-PROMOTED packet {} at training intake (status '{}'); "
                "only packets that passed Gate 1 may enter training data".format(
                    packet.packet_id, packet.promotion_status.value
                )
            )
        if strict_no_mock and packet.provenance.is_mock:
            raise DeepApplyIntakeError(
                "refusing mock-provenance packet {} at training intake; "
                "a mock cannot launder itself into training data".format(packet.packet_id)
            )
        accepted.append(packet)

    rows: List[Dict[str, Any]] = []
    for packet in accepted:
        rows.extend(_rows_from_packet(packet))

    blob = json.dumps(rows, sort_keys=True, ensure_ascii=False, default=str)
    dataset_hash = hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # Provenance propagation: depth = max over sources; risk = max severity
    # over source domains (computed in AdapterPacket via source_domains).
    source_domains = sorted({p.domain.value for p in accepted})
    # Honest "unknown": if every source packet has safety_score=None, the
    # inherited floor is None, NOT 1.0. A default of 1.0 would present unknown
    # safety as perfect safety and let it clear the hard safety_threshold
    # check at Gate 2 (gate2.py:148, hard=True). None propagates to
    # AdapterPacket.safety_score, which hard-fails that check ("unset" safety
    # cannot pass a hard gate) -- the correct outcome for unmeasured safety.
    min_safety = min(
        (p.safety_score for p in accepted if p.safety_score is not None),
        default=None,
    )

    manifest = {
        "row_count": len(rows),
        "packet_count": len(accepted),
        "source_packet_ids": [p.packet_id for p in accepted],
        "source_packet_ids_sorted": sorted(p.packet_id for p in accepted),
        "synthetic_depth_max": max((p.provenance.synthetic_depth for p in accepted), default=0),
        "source_domains": source_domains,
        "contains_mock": any(p.provenance.is_mock for p in accepted),
        "min_safety_score": min_safety,
        "dataset_hash": dataset_hash,
    }
    return TrainingDataset(rows, manifest)