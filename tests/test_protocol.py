"""Packet protocol: schema validation and the invariants that carry safety weight."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from asea.core.protocol import (
    APPLICABLE_LEVELS,
    CapabilityKey,
    Domain,
    LearningLevel,
    Modality,
    PacketType,
    PromotionStatus,
    Provenance,
    OriginKind,
    RiskTier,
    SkillPacket,
    risk_tier_for_domain,
)


def test_capability_key_is_stable_and_hashable():
    a = CapabilityKey(task_type="translate", modality=Modality.TEXT, language="as->en")
    b = CapabilityKey(task_type="translate", modality=Modality.TEXT, language="as->en")
    assert a.as_str() == b.as_str() == "translate/text/general/as->en"
    assert {a, b} == {a}  # frozen models hash by value


def test_rejects_unknown_fields(capability, clean_provenance):
    with pytest.raises(ValidationError):
        SkillPacket(
            task_type="t", source_module="s", target_module="r",
            sender_capability=capability, modality=Modality.TEXT,
            provenance=clean_provenance, not_a_real_field=1,
        )


def test_empty_module_id_rejected(capability, clean_provenance):
    with pytest.raises(ValidationError):
        SkillPacket(
            task_type="t", source_module="   ", target_module="r",
            sender_capability=capability, modality=Modality.TEXT,
            provenance=clean_provenance,
        )


def test_scores_must_be_normalised(capability, clean_provenance, packet_factory):
    with pytest.raises(ValidationError):
        packet_factory(capability, clean_provenance, confidence_score=1.4)


def test_rejected_requires_reason(capability, clean_provenance, packet_factory):
    with pytest.raises(ValidationError):
        packet_factory(
            capability, clean_provenance, promotion_status=PromotionStatus.REJECTED
        )
    ok = packet_factory(
        capability, clean_provenance,
        promotion_status=PromotionStatus.REJECTED, rejection_reason="because",
    )
    assert ok.rejection_reason == "because"


def test_promoted_requires_payload_and_rollback(capability, clean_provenance, packet_factory):
    with pytest.raises(ValidationError):
        packet_factory(
            capability, clean_provenance,
            promotion_status=PromotionStatus.PROMOTED,
            distilled_skill={"entries": [1]},
        )  # no rollback token
    with pytest.raises(ValidationError):
        packet_factory(
            capability, clean_provenance,
            promotion_status=PromotionStatus.PROMOTED, rollback_token="tok",
        )  # no payload


def test_redaction_never_leaks_raw_sender_output(capability, clean_provenance, packet_factory):
    packet = packet_factory(
        capability, clean_provenance,
        sender_output="RAW MODEL TEXT THAT MUST NOT TRAVEL",
        distilled_skill={"entries": [{"source": "a", "target": "b"}]},
        packet_type=PacketType.GLOSSARY,
    )
    view = packet.redacted_for_receiver()
    assert "sender_output" not in view
    assert "RAW MODEL TEXT" not in str(view)
    assert view["distilled_skill"]["entries"][0]["target"] == "b"


def test_content_hash_tracks_payload_not_bookkeeping(capability, clean_provenance, packet_factory):
    p1 = packet_factory(
        capability, clean_provenance,
        distilled_skill={"entries": [{"source": "a", "target": "b"}]},
        packet_type=PacketType.GLOSSARY,
    )
    p2 = packet_factory(
        capability, clean_provenance,
        distilled_skill={"entries": [{"source": "a", "target": "b"}]},
        packet_type=PacketType.GLOSSARY,
        confidence_score=0.9,  # bookkeeping differs
    )
    assert p1.content_hash() == p2.content_hash()

    p3 = packet_factory(
        capability, clean_provenance,
        distilled_skill={"entries": [{"source": "a", "target": "DIFFERENT"}]},
        packet_type=PacketType.GLOSSARY,
    )
    assert p3.content_hash() != p1.content_hash()


@pytest.mark.parametrize(
    "domain,tier",
    [
        (Domain.MEDICAL, RiskTier.HIGH),
        (Domain.LEGAL, RiskTier.HIGH),
        (Domain.FINANCE, RiskTier.HIGH),
        (Domain.EDUCATION, RiskTier.MEDIUM),
        (Domain.TRANSLATION, RiskTier.LOW),
    ],
)
def test_risk_tiers(domain, tier):
    assert risk_tier_for_domain(domain) == tier


def test_high_risk_packets_declare_human_requirement(clean_provenance, packet_factory):
    cap = CapabilityKey(task_type="triage", modality=Modality.STRUCTURED, domain=Domain.MEDICAL)
    packet = packet_factory(cap, clean_provenance, domain=Domain.MEDICAL)
    assert packet.requires_human_approval is True


def test_training_levels_are_not_applicable():
    """L4/L5 must never be treated as something this system can apply."""
    assert LearningLevel.L4_PEFT_CANDIDATE not in APPLICABLE_LEVELS
    assert LearningLevel.L5_DISTILL_DATASET not in APPLICABLE_LEVELS
    assert LearningLevel.L3_SKILL_PACKET in APPLICABLE_LEVELS


def test_provenance_extension_is_append_only():
    prov = Provenance(origin_kind=OriginKind.HUMAN_VERIFIED, chain=["a"])
    extended = prov.extended("b", synthetic=True, is_mock=True)
    assert prov.chain == ["a"], "original must not be mutated"
    assert extended.chain == ["a", "b"]
    assert extended.synthetic_depth == 1
    assert extended.is_mock is True
