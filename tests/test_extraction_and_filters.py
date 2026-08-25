"""Extraction, relevance filtering and safety filtering."""

from __future__ import annotations

import pytest

from asea.core.errors import ExtractionError
from asea.core.protocol import Domain, Gap, Modality, PromotionStatus
from asea.extraction.extractors import CodeExtractor, TextExtractor
from asea.filters.relevance import RelevanceFilter, RelevancePolicy
from asea.filters.safety import SafetyFilter, SafetyPolicy
from asea.modules.mock.zoo import make_generic_receiver, make_generic_sender, text_cap


@pytest.fixture
def cap():
    return text_cap("translate", "as->en", Domain.TRANSLATION)


@pytest.fixture
def gap(cap):
    return Gap(capability=cap, receiver_score=0.2, sender_score=0.9)


def probes(pairs):
    return [
        {"case_id": "c{}".format(i), "prompt": p, "expected": e,
         "meta": {"human_verified": True}}
        for i, (p, e) in enumerate(pairs)
    ]


# -- extraction -------------------------------------------------------------


def test_extraction_produces_packets_with_provenance(cap, gap):
    sender = make_generic_sender(
        capabilities=[cap], knowledge={cap.as_str(): {"মই": "I"}}
    )
    receiver = make_generic_receiver(capabilities=[cap])
    packets = TextExtractor().extract(sender, receiver, gap, probes([("মই", "I")]))

    assert len(packets) == 1
    p = packets[0]
    assert p.promotion_status == PromotionStatus.EXTRACTED
    assert p.sender_output == "I"
    assert p.distilled_skill is None, "extraction must not distil"
    assert p.provenance.chain == [sender.module_id]
    assert p.provenance.is_mock is True
    assert p.raw_input_reference == "c0"


def test_human_verified_probes_do_not_increment_synthetic_depth(cap, gap):
    sender = make_generic_sender(capabilities=[cap], knowledge={cap.as_str(): {"মই": "I"}})
    receiver = make_generic_receiver(capabilities=[cap])
    verified = TextExtractor().extract(sender, receiver, gap, probes([("মই", "I")]))
    assert verified[0].provenance.synthetic_depth == 0

    unverified = TextExtractor().extract(
        sender, receiver, gap,
        [{"case_id": "u", "prompt": "মই", "expected": "I", "meta": {}}],
    )
    assert unverified[0].provenance.synthetic_depth == 1


def test_extractor_rejects_wrong_modality(cap, gap):
    sender = make_generic_sender(capabilities=[cap])
    receiver = make_generic_receiver(capabilities=[cap])
    with pytest.raises(ExtractionError, match="modality"):
        CodeExtractor().extract(sender, receiver, gap, probes([("x", "y")]))


def test_extractor_refuses_empty_probe_set(cap, gap):
    sender = make_generic_sender(capabilities=[cap])
    receiver = make_generic_receiver(capabilities=[cap])
    with pytest.raises(ExtractionError, match="no probes"):
        TextExtractor().extract(sender, receiver, gap, [])


# -- relevance --------------------------------------------------------------


def _extract(cap, gap, sender_table, receiver_table, pairs):
    sender = make_generic_sender(capabilities=[cap], knowledge={cap.as_str(): sender_table})
    receiver = make_generic_receiver(
        capabilities=[cap], knowledge={cap.as_str(): receiver_table}, fallback="echo"
    )
    packets = TextExtractor().extract(sender, receiver, gap, probes(pairs))
    return packets, receiver


def test_relevance_drops_incorrect_sender(cap, gap):
    packets, receiver = _extract(
        cap, gap, {"চাহ": "coffee"}, {}, [("চাহ", "tea")]
    )
    kept, dropped = RelevanceFilter().apply(packets, receiver)
    assert not kept and len(dropped) == 1
    assert "sender_incorrect" in dropped[0].rejection_reason
    assert dropped[0].promotion_status == PromotionStatus.REJECTED


def test_relevance_drops_competent_receiver(cap, gap):
    packets, receiver = _extract(cap, gap, {"ভাল": "good"}, {"ভাল": "good"}, [("ভাল", "good")])
    kept, dropped = RelevanceFilter().apply(packets, receiver)
    assert not kept
    assert "receiver_competent" in dropped[0].rejection_reason


def test_relevance_drops_when_no_delta(cap, gap):
    """Sender and receiver agree, so there is nothing to teach."""
    packets, receiver = _extract(cap, gap, {"মই": "I"}, {"মই": "I"}, [("মই", "I")])
    # Raise the competence ceiling so the competent-receiver rule cannot fire
    # first; we want to isolate the no_delta path.
    flt = RelevanceFilter(RelevancePolicy(receiver_competence_ceiling=1.01))
    kept, dropped = flt.apply(packets, receiver)
    assert not kept
    assert "no_delta" in dropped[0].rejection_reason


def test_relevance_drops_duplicates(cap, gap):
    packets, receiver = _extract(
        cap, gap, {"মই": "I"}, {}, [("মই", "I"), ("মই", "I")]
    )
    kept, dropped = RelevanceFilter().apply(packets, receiver)
    assert len(kept) == 1
    assert "duplicate" in dropped[0].rejection_reason


def test_relevance_keeps_a_genuine_signal(cap, gap):
    packets, receiver = _extract(cap, gap, {"ভাত": "rice"}, {}, [("ভাত", "rice")])
    kept, dropped = RelevanceFilter().apply(packets, receiver)
    assert len(kept) == 1 and not dropped
    assert kept[0].promotion_status == PromotionStatus.FILTERED
    assert kept[0].notes["sender_correctness"] == 1.0


def test_relevance_keeps_a_reference_free_signal_with_real_delta(cap, gap):
    """A packet with no reference (notes['reference'] is None) must still be
    judged on the sender-vs-receiver delta alone. Audit 2026-08-13 #14: the
    reference-free path had no test, so weakening the ``no_delta`` guard to
    ``<=`` (instead of ``<``) or skipping it for reference-free packets would
    silently let through signals with nothing to teach. Pin that a real delta
    is KEPT and that neither ``sender_incorrect`` nor ``receiver_competent``
    fires (both are reference-gated and must be skipped when reference is None).
    """
    packets, receiver = _extract(cap, gap, {"ভাত": "rice"}, {}, [("ভাত", "rice")])
    # Strip the reference the extractor derived from the probe's `expected`.
    packets[0].notes["reference"] = None
    kept, dropped = RelevanceFilter().apply(packets, receiver)
    assert len(kept) == 1 and not dropped
    assert kept[0].promotion_status == PromotionStatus.FILTERED
    assert kept[0].rejection_reason is None
    # The two reference-gated checks must not have recorded a score at all.
    assert "sender_correctness" not in kept[0].notes
    assert "receiver_correctness" not in kept[0].notes


def test_relevance_drops_a_reference_free_signal_with_no_delta(cap, gap):
    """The flip side of #14: a reference-free packet whose sender and receiver
    already agree must still be dropped via ``no_delta``. This pins that the
    delta guard is the load-bearing check for reference-free packets -- if it
    were removed, an agreeing pair would be kept (teaching nothing)."""
    packets, receiver = _extract(cap, gap, {"মই": "I"}, {"মই": "I"}, [("মই", "I")])
    packets[0].notes["reference"] = None
    kept, dropped = RelevanceFilter().apply(packets, receiver)
    assert not kept and len(dropped) == 1
    assert "no_delta" in dropped[0].rejection_reason
    assert dropped[0].promotion_status == PromotionStatus.REJECTED
    # Reference-gated checks must not have fired or recorded a score.
    assert "sender_correctness" not in dropped[0].notes
    assert "receiver_correctness" not in dropped[0].notes


# -- safety -----------------------------------------------------------------


def _medical_packet(output, packet_factory, capability, clean_provenance):
    cap = capability.model_copy(update={"domain": Domain.MEDICAL})
    return packet_factory(
        cap, clean_provenance, domain=Domain.MEDICAL, sender_output=output,
        notes={"prompt": "patient presents with symptoms", "reference": output},
    )


def test_safety_blocks_dosage_in_medical(packet_factory, capability, clean_provenance):
    p = _medical_packet("Give 500 mg paracetamol every six hours.", packet_factory,
                        capability, clean_provenance)
    kept, dropped = SafetyFilter().apply([p])
    assert not kept
    assert "dosage_instruction" in dropped[0].rejection_reason
    assert dropped[0].safety_score == 0.0


def test_safety_blocks_diagnostic_certainty(packet_factory, capability, clean_provenance):
    p = _medical_packet("You have appendicitis, no need to see a doctor.", packet_factory,
                        capability, clean_provenance)
    kept, dropped = SafetyFilter().apply([p])
    assert not kept
    assert "diagnostic_certainty" in dropped[0].rejection_reason


def test_safety_blocks_credentials(packet_factory, capability, clean_provenance):
    p = packet_factory(
        capability, clean_provenance,
        sender_output="use api_key = sk-abcdef0123456789abcdef",
        notes={"prompt": "x", "reference": "x"},
    )
    kept, dropped = SafetyFilter().apply([p])
    assert not kept and "credential_leak" in dropped[0].rejection_reason


def test_safety_blocks_pii(packet_factory, capability, clean_provenance):
    p = packet_factory(
        capability, clean_provenance,
        sender_output="contact ram@example.com for details",
        notes={"prompt": "x", "reference": "x"},
    )
    kept, dropped = SafetyFilter().apply([p])
    assert not kept and "pii" in dropped[0].rejection_reason


def test_safety_penalises_medical_without_escalation(packet_factory, capability, clean_provenance):
    p = _medical_packet("Rest at home and drink fluids.", packet_factory,
                        capability, clean_provenance)
    score, findings = SafetyFilter().score(p)
    assert any(f.code == "missing_escalation" for f in findings)
    assert score < 1.0


def test_safety_passes_a_well_formed_triage_rule(packet_factory, capability, clean_provenance):
    p = _medical_packet(
        "Red flag. Seek emergency medical attention without delay.",
        packet_factory, capability, clean_provenance,
    )
    kept, dropped = SafetyFilter().apply([p])
    assert len(kept) == 1 and not dropped
    assert kept[0].safety_score == 1.0


def test_safety_low_risk_domain_allows_numbers_with_units(packet_factory, capability, clean_provenance):
    """The dosage rule must not fire outside high-risk domains."""
    p = packet_factory(
        capability, clean_provenance, domain=Domain.TRANSLATION,
        sender_output="add 500 mg of sugar",
        notes={"prompt": "recipe", "reference": "add 500 mg of sugar"},
    )
    kept, _ = SafetyFilter().apply([p])
    assert len(kept) == 1
