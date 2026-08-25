"""SILT -- Skill Interchange Layer with Trust-gating.

(Package imports as ``asea``, the working name this was built under:
Adaptive Skill Extraction Adapter.)

A universal, local-first adapter for controlled AI-to-AI skill transfer:
one module acts as Sender/Teacher, another as Receiver/Learner, and every
signal that crosses between them is filtered, distilled, evaluated and gated.

This package does NOT train models and does NOT claim capability copying.
See docs/feasibility_review.md for the honest scope statement.
"""

__version__ = "0.1.0"

from .core.protocol import (  # noqa: F401
    CapabilityKey,
    CapabilityManifest,
    Domain,
    Gap,
    LearningLevel,
    Modality,
    PacketType,
    PromotionStatus,
    Provenance,
    RiskTier,
    SkillPacket,
)
