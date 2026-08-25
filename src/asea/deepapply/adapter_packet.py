"""The AdapterPacket -- the typed record of a trained adapter.

Mirrors the discipline of :class:`SkillPacket`: Pydantic ``extra="forbid"``,
``validate_assignment=True``, terminal-state validators, and a stable content
hash for the duplicate-content guard. An adapter is *not* a SkillPacket; it has
its own envelope, its own gate (:class:`DeepApplyGate`) and its own store
(:class:`AdapterStore`). What it shares is the principle: the gate measures the
artifact's outcome, never the trainer that produced it.

Provenance and risk propagate from the source packets:

* ``synthetic_depth = max(source packets' synthetic_depth)`` -- an adapter
  trained on depth-2 knowledge is depth-2 knowledge; the Gate 2 ceiling applies
  independently of Gate 1.
* ``risk_domain = max-severity source domain`` -- if ANY source packet is
  medical/legal/finance, the adapter is HIGH risk and parks at PENDING_HUMAN
  regardless of scores. Not disableable.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..core.protocol import (
    Domain,
    EvaluationScores,
    LearningLevel,
    PromotionStatus,
    Provenance,
    RiskTier,
    risk_tier_for_domain,
)


def max_risk_tier(domains: List[Domain]) -> RiskTier:
    """Highest risk tier across a set of domains. LOW if empty."""
    if not domains:
        return RiskTier.LOW
    tiers = [risk_tier_for_domain(d) for d in domains]
    if RiskTier.HIGH in tiers:
        return RiskTier.HIGH
    if RiskTier.MEDIUM in tiers:
        return RiskTier.MEDIUM
    return RiskTier.LOW


class AdapterPacket(BaseModel):
    """The universal unit of deep-apply -- a trained, gated LoRA adapter.

    Moves through: TRAINED -> EVALUATED -> (PENDING_HUMAN) -> PROMOTED | REJECTED
    | ROLLED_BACK. Every transition is audited in the same chain as packets.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    adapter_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    version: int = Field(default=1, ge=1)

    # -- identity ---------------------------------------------------------
    base_model: str
    #: v1 fingerprints by base_model id + LoRA config hash, NOT by a full weight
    #: hash. Honest limit: two downloads of the same id are assumed identical.
    base_model_fingerprint: str
    #: The receiver module the adapter was trained on (for no_self_lineage).
    target_module: str

    # -- provenance (propagated from source packets) ---------------------
    source_packet_ids: List[str] = Field(default_factory=list)
    source_domains: List[Domain] = Field(default_factory=list)
    synthetic_depth: int = Field(default=0, ge=0)
    provenance: Provenance
    learning_level: LearningLevel = LearningLevel.L4_PEFT_CANDIDATE

    # -- training metadata ----------------------------------------------
    lora_config: Dict[str, Any] = Field(default_factory=dict)
    training_config_hash: str = ""
    dataset_hash: str = ""
    seed: int = 0
    backend: str = ""
    backend_version: str = ""
    trainable_param_count: int = Field(default=0, ge=0)
    training_loss: float = Field(default=0.0)
    #: Path/ref into the adapter store (or None for a not-yet-admitted artifact).
    adapter_artifact_ref: Optional[str] = None

    # -- streamed-backend metadata (parity is the admission bar) ----------
    #: Storage tier the streamed backend banked the frozen base to ("ram"/"disk").
    #: Empty for the standard backend (no banking).
    storage_tier: str = ""
    #: True only if a pre-train parity check passed for this exact config. A
    #: configuration whose parity is unverified is recorded False here -- never
    #: silently True. The streamed backend refuses (ParityError) on a parity
    #: FAILURE, so a stored adapter with parity_verified=True is one that passed.
    parity_verified: bool = False
    #: sha256 of the full parity report block (diffs, device, dtype, fingerprint)
    #: -- lets the audit chain and the packet agree on WHAT was verified.
    parity_report_hash: str = ""
    #: siltstream config fingerprint (sha256[:16] of model shape + tier + device
    #: + LoRA config) -- the exact configuration the parity check covered.
    config_fingerprint: str = ""

    # -- evaluation + gate -----------------------------------------------
    safety_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    evaluator_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    scores: Optional[EvaluationScores] = None

    promotion_status: PromotionStatus = PromotionStatus.DRAFT
    rejection_reason: Optional[str] = None
    human_approved_by: Optional[str] = None
    rollback_token: Optional[str] = None

    notes: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default="")

    # -- validation ------------------------------------------------------

    @model_validator(mode="after")
    def _check_terminal_states(self) -> "AdapterPacket":
        if self.promotion_status == PromotionStatus.REJECTED and not self.rejection_reason:
            raise ValueError("REJECTED adapters must carry a rejection_reason")
        if self.promotion_status == PromotionStatus.PROMOTED:
            if self.adapter_artifact_ref is None:
                raise ValueError("PROMOTED adapters must have an adapter_artifact_ref")
            if self.rollback_token is None:
                raise ValueError("PROMOTED adapters must have a rollback_token")
        return self

    # -- helpers ---------------------------------------------------------

    @property
    def risk_tier(self) -> RiskTier:
        return max_risk_tier(self.source_domains)

    @property
    def requires_human_approval(self) -> bool:
        return self.risk_tier == RiskTier.HIGH

    def content_hash(self) -> str:
        """Stable hash of the adapter identity (excludes volatile bookkeeping).

        Bound to base model + LoRA config + dataset + source ids + backend, so
        a re-train with different data or hyper-parameters is a different
        adapter (a certificate for yesterday's adapter is a lie about today's).
        """
        payload = {
            "base_model": self.base_model,
            "base_model_fingerprint": self.base_model_fingerprint,
            "lora_config": self.lora_config,
            "dataset_hash": self.dataset_hash,
            "source_packet_ids": sorted(self.source_packet_ids),
            "backend": self.backend,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()