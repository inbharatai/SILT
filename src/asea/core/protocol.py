"""Universal packet protocol.

This module is the contract that every sender, receiver, extractor, distiller,
evaluator and gate in the system agrees on. It is deliberately modality-neutral:
nothing in here knows what Assamese, TTS or Python bug-fixing *is*. Modality
specific behaviour lives in plugins registered against these types.

Nothing here performs model inference.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------


class Modality(str, Enum):
    """Transport-level content type of a skill packet."""

    TEXT = "text"
    AUDIO_ASR = "audio_asr"
    SPEECH_TTS = "speech_tts"
    OCR = "ocr"
    CODE = "code"
    STRUCTURED = "structured"


class Domain(str, Enum):
    """Subject area. Drives risk tiering, *not* processing logic."""

    LANGUAGE = "language"
    TRANSLATION = "translation"
    PRONUNCIATION = "pronunciation"
    SOFTWARE = "software"
    MEDICAL = "medical"
    LEGAL = "legal"
    FINANCE = "finance"
    EDUCATION = "education"
    GENERAL = "general"


class RiskTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


#: Domains where a wrong answer can cause physical, legal or financial harm.
#: Packets in these domains can never reach PROMOTED without a human approval
#: record. Enforced in asea.promotion.gate, not merely documented.
HIGH_RISK_DOMAINS = frozenset({Domain.MEDICAL, Domain.LEGAL, Domain.FINANCE})


def risk_tier_for_domain(domain: Domain) -> RiskTier:
    if domain in HIGH_RISK_DOMAINS:
        return RiskTier.HIGH
    if domain in (Domain.EDUCATION,):
        return RiskTier.MEDIUM
    return RiskTier.LOW


class LearningLevel(int, Enum):
    """How deeply a packet is allowed to affect the receiver.

    L0-L3 are fully implemented and reversible: they never touch model weights.
    L4-L5 are *export only* in this codebase. We emit a validated dataset and a
    training job spec; we do not train. See docs/feasibility_review.md.
    """

    L0_INTERACTION = 0
    L1_CONTEXT = 1
    L2_MEMORY_RAG = 2
    L3_SKILL_PACKET = 3
    L4_PEFT_CANDIDATE = 4
    L5_DISTILL_DATASET = 5


#: Levels this codebase can actually apply to a live receiver.
APPLICABLE_LEVELS = frozenset(
    {
        LearningLevel.L0_INTERACTION,
        LearningLevel.L1_CONTEXT,
        LearningLevel.L2_MEMORY_RAG,
        LearningLevel.L3_SKILL_PACKET,
    }
)


class PromotionStatus(str, Enum):
    DRAFT = "draft"
    EXTRACTED = "extracted"
    FILTERED = "filtered"
    DISTILLED = "distilled"
    EVALUATED = "evaluated"
    PENDING_HUMAN = "pending_human_approval"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


class PacketType(str, Enum):
    """Shape of the distilled payload. Distillers declare which they emit."""

    GLOSSARY = "glossary"
    CORRECTION_PAIR = "correction_pair"
    EXEMPLAR = "exemplar"
    RULE = "rule"
    LEXICON = "lexicon"


class OriginKind(str, Enum):
    HUMAN_VERIFIED = "human_verified"
    CURATED_CORPUS = "curated_corpus"
    MODEL_GENERATED = "model_generated"


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


class Provenance(BaseModel):
    """Where a packet came from and how far it is from ground truth.

    ``synthetic_depth`` is the anti-collapse counter: 0 means the content traces
    to human-verified or curated data, 1 means it was produced by a model from
    such data, 2 means a model consumed model output, and so on. The promotion
    gate refuses packets above a configured ceiling because recursively training
    on model output degrades the receiver.
    """

    model_config = ConfigDict(extra="forbid")

    origin_kind: OriginKind
    chain: List[str] = Field(
        default_factory=list,
        description="Ordered module ids the content passed through, oldest first.",
    )
    synthetic_depth: int = Field(default=0, ge=0)
    is_mock: bool = Field(
        default=False,
        description="True if any producing module was a mock. Blocks strict promotion.",
    )
    source_reference: Optional[str] = Field(
        default=None, description="Dataset id, file path or corpus citation."
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def extended(self, module_id: str, synthetic: bool, is_mock: bool) -> "Provenance":
        """Return a copy with one more hop recorded. Provenance is append-only."""
        return Provenance(
            origin_kind=self.origin_kind,
            chain=list(self.chain) + [module_id],
            synthetic_depth=self.synthetic_depth + (1 if synthetic else 0),
            is_mock=self.is_mock or is_mock,
            source_reference=self.source_reference,
            created_at=self.created_at,
        )


# --------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------


class EvaluationScores(BaseModel):
    """Per-metric evaluator output, all normalised to 0..1 where 1 is best.

    ``hallucination_risk`` is the exception and is inverted at aggregation time
    (see asea.evaluator.evaluator) so callers can read every stored field as
    "higher is better" except this one.
    """

    model_config = ConfigDict(extra="forbid")

    schema_compliance: float = Field(ge=0.0, le=1.0)
    semantic_similarity: float = Field(ge=0.0, le=1.0)
    task_success: float = Field(ge=0.0, le=1.0)
    language_preservation: float = Field(ge=0.0, le=1.0)
    hallucination_risk: float = Field(ge=0.0, le=1.0)
    aggregate: float = Field(ge=0.0, le=1.0)

    #: Benchmark deltas, populated by the harness.
    baseline_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    candidate_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    regression_detected: bool = False
    regression_detail: Optional[str] = None

    #: Control-movement bound (Gate 2, audit 2026-08-17). The regression sweep
    #: runs over CONTROL (non-target) suites; ``regression_detected`` only fires
    #: on score DROPS (delta < -tolerance), so a control suite that IMPROVED
    #: (training bled into a capability it was not targeting) was invisible to
    #: the gate. ``control_movement_detected`` fires on |delta| > the bound in
    #: EITHER direction, closing that hole. Gate-strengthening: a new HARD
    #: check (``no_control_movement``) reads this bool.
    control_movement_detected: bool = False
    control_movement_detail: Optional[str] = None

    #: SPRT early-stop record (B2, audit 2026-08-17). Populated ONLY when the
    #: evaluator was constructed with an ``SprtConfig`` AND the candidate held-out
    #: run was stopped early by the SPRT -- which, by the SPRT's asymmetry, can
    #: ONLY ever be a REJECT (early-PROMOTE is forbidden: the SPRT's
    #: ``should_stop`` returns True only on REJECT). When present, the candidate
    #: run is PARTIAL (``case_count`` reflects cases actually scored, not the
    #: full suite), and a new HARD gate check ``no_statistical_early_reject``
    #: fails the packet regardless of its (incomplete, therefore optimistic)
    #: aggregate. ``None`` means SPRT was disabled (the default) OR the run
    #: completed without an early stop -- both byte-identical to pre-B2.
    sprt: Optional[Dict[str, Any]] = None

    #: Per-case outcomes on the held-out split. An aggregate gain can hide a
    #: case that previously worked and now does not; these fields let the gate
    #: see that, which an average cannot.
    case_count: int = 0
    case_regression_count: int = 0

    @property
    def case_regression_ratio(self) -> float:
        if not self.case_count:
            return 0.0
        return self.case_regression_count / self.case_count

    @property
    def improvement(self) -> Optional[float]:
        if self.baseline_score is None or self.candidate_score is None:
            return None
        return self.candidate_score - self.baseline_score


# --------------------------------------------------------------------------
# Capability manifests and gaps
# --------------------------------------------------------------------------


class CapabilityKey(BaseModel):
    """A single addressable competence, e.g. translation/as->en.

    Capability keys are the vocabulary of the handshake. Both sides publish
    them; the gap engine performs set arithmetic over them. Keys are opaque to
    the core -- plugins decide what strings mean.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_type: str
    modality: Modality
    domain: Domain = Domain.GENERAL
    language: Optional[str] = Field(
        default=None, description="BCP-47-ish tag, e.g. 'as', 'en', 'hi', 'as->en'."
    )

    def as_str(self) -> str:
        return "{}/{}/{}/{}".format(
            self.task_type, self.modality.value, self.domain.value, self.language or "-"
        )

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.as_str()


class CapabilityManifest(BaseModel):
    """What a module claims it can do. Published during handshake.

    Claims are *not* trusted: the gap engine cross-checks them against measured
    benchmark results before deciding a transfer is warranted.
    """

    model_config = ConfigDict(extra="forbid")

    module_id: str
    display_name: str
    roles: List[str] = Field(default_factory=list)  # "sender" and/or "receiver"
    capabilities: List[CapabilityKey] = Field(default_factory=list)
    max_learning_level: LearningLevel = LearningLevel.L3_SKILL_PACKET
    is_mock: bool = False
    version: str = "0.1.0"

    def capability_set(self) -> set:
        return {c.as_str() for c in self.capabilities}

    def supports(self, key: CapabilityKey) -> bool:
        return key.as_str() in self.capability_set()


class Gap(BaseModel):
    """One measured deficiency in the receiver that the sender can cover."""

    model_config = ConfigDict(extra="forbid")

    capability: CapabilityKey
    receiver_score: float = Field(ge=0.0, le=1.0)
    sender_score: float = Field(ge=0.0, le=1.0)
    declared_only: bool = Field(
        default=False,
        description="True when derived from manifests alone with no benchmark evidence.",
    )

    @property
    def headroom(self) -> float:
        return max(0.0, self.sender_score - self.receiver_score)


# --------------------------------------------------------------------------
# The packet itself
# --------------------------------------------------------------------------


class SkillPacket(BaseModel):
    """The universal unit of transfer.

    A packet moves through: EXTRACTED -> FILTERED -> DISTILLED -> EVALUATED ->
    (PENDING_HUMAN) -> PROMOTED | REJECTED. Every transition is audited.

    Invariant enforced by the pipeline: ``sender_output`` (raw model output) is
    never handed to the receiver. Only ``distilled_skill`` crosses the boundary.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    packet_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    version: int = Field(default=1, ge=1)

    task_type: str
    source_module: str
    target_module: str

    sender_capability: CapabilityKey
    receiver_gap: Optional[Gap] = None

    modality: Modality
    language: Optional[str] = None
    domain: Domain = Domain.GENERAL

    raw_input_reference: Optional[str] = Field(
        default=None,
        description="Pointer (dataset id + row) to the probe input. Not the input itself.",
    )
    sender_output: Optional[Any] = Field(
        default=None,
        description="Raw sender response. Retained for audit ONLY. Never applied to receiver.",
    )

    packet_type: Optional[PacketType] = None
    distilled_skill: Optional[Dict[str, Any]] = Field(
        default=None,
        description="The only field the receiver is ever allowed to consume.",
    )

    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    evaluator_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    safety_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    scores: Optional[EvaluationScores] = None

    provenance: Provenance
    learning_level: LearningLevel = LearningLevel.L3_SKILL_PACKET
    promotion_status: PromotionStatus = PromotionStatus.DRAFT
    rejection_reason: Optional[str] = None

    human_approved_by: Optional[str] = None
    rollback_token: Optional[str] = None

    notes: Dict[str, Any] = Field(default_factory=dict)

    # -- validation -------------------------------------------------------

    @field_validator("task_type", "source_module", "target_module")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must be a non-empty string")
        return v

    @model_validator(mode="after")
    def _check_terminal_states(self) -> "SkillPacket":
        if self.promotion_status == PromotionStatus.REJECTED and not self.rejection_reason:
            raise ValueError("REJECTED packets must carry a rejection_reason")
        if self.promotion_status == PromotionStatus.PROMOTED:
            if self.distilled_skill is None:
                raise ValueError("PROMOTED packets must have a distilled_skill")
            if self.rollback_token is None:
                raise ValueError("PROMOTED packets must have a rollback_token")
        return self

    # -- helpers ----------------------------------------------------------

    @property
    def risk_tier(self) -> RiskTier:
        return risk_tier_for_domain(self.domain)

    @property
    def requires_human_approval(self) -> bool:
        return self.risk_tier == RiskTier.HIGH

    def content_hash(self) -> str:
        """Stable hash of the semantic payload (excludes volatile bookkeeping)."""
        payload = {
            "task_type": self.task_type,
            "capability": self.sender_capability.as_str(),
            "distilled_skill": self.distilled_skill,
            "packet_type": self.packet_type.value if self.packet_type else None,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def redacted_for_receiver(self) -> Dict[str, Any]:
        """Exactly what the receiver is permitted to see.

        This is the enforcement point for "raw model output never trains another
        model". Note the deliberate absence of ``sender_output``.
        """
        return {
            "packet_id": self.packet_id,
            "packet_type": self.packet_type.value if self.packet_type else None,
            "capability": self.sender_capability.as_str(),
            "language": self.language,
            "domain": self.domain.value,
            "distilled_skill": self.distilled_skill,
        }
