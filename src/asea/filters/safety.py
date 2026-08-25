"""Safety filter.

Assigns every packet a ``safety_score`` in 0..1 (1.0 = no concern detected) and
hard-blocks a small set of categorically unacceptable payloads.

Be clear about what this is: a **rule-based tripwire**, not a safety classifier.
It catches obvious, enumerable failure shapes -- dosage instructions, diagnostic
certainty, self-harm content, credential-looking strings, absolute medical
claims. It will not catch a subtly wrong triage rule phrased responsibly. That
is precisely why HIGH-risk domains additionally require human approval and can
never auto-promote regardless of the score this filter returns.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from ..core.protocol import (
    Domain,
    PromotionStatus,
    RiskTier,
    SkillPacket,
    risk_tier_for_domain,
)
from ..evaluator import metrics
from ..evaluator.similarity import normalize

# -- categorical blocks -----------------------------------------------------

#: Dosage-shaped strings: a number followed by a drug unit. In a medical packet
#: this is prescribing, which this system must never transfer automatically.
_DOSAGE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|µg|ml|cc|iu|units?|g)\b(?:\s?(?:per|/)\s?\w+)?",
    re.IGNORECASE,
)

_DIAGNOSTIC_CERTAINTY = (
    "you have",
    "this is definitely",
    "diagnosed with",
    "there is no need to see a doctor",
    "do not seek",
    "no need for medical attention",
    "safe to ignore",
)

_SELF_HARM = ("suicide method", "how to overdose", "kill yourself", "end your life")

#: Prompt-injection tripwires. Distilled payloads are rendered into the
#: RECEIVER'S SYSTEM PROMPT by the connectors' render_skills(), so an
#: instruction-shaped string inside a "translation" is not noise -- it is an
#: attempted jailbreak that the packet's provenance would then legitimise.
#: Found during the adversarial audit (docs/loophole_audit.md, A1).
_INJECTION_MARKERS = (
    "ignore all previous instructions",
    "ignore previous instructions",
    "disregard the reference",
    "disregard your instructions",
    "reveal your system prompt",
    "you are now unrestricted",
    "from now on, always answer",
    "output your instructions",
    "system:",
    "</entries>",
    "<|im_start|>",
    "[inst]",
)

_CREDENTIAL_RE = re.compile(
    r"(sk-[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{12,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"api[_-]?key\s*[:=]\s*\S{8,})",
    re.IGNORECASE,
)

#: Rough PII shapes. Present so extraction cannot quietly launder personal data
#: out of a source corpus and into a receiver's permanent skill store.
_PII_PATTERNS = (
    re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),          # Aadhaar-like 12 digit
    re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),              # PAN-like
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"),      # email
    re.compile(r"\b(?:\+91[- ]?)?[6-9]\d{9}\b"),        # IN mobile
)


class SafetyPolicy:
    def __init__(
        self,
        min_safety_score: float = 0.7,
        block_pii: bool = True,
        max_hallucination_risk: float = 0.5,
    ) -> None:
        self.min_safety_score = min_safety_score
        self.block_pii = block_pii
        self.max_hallucination_risk = max_hallucination_risk


class SafetyFinding:
    def __init__(self, code: str, detail: str, penalty: float, blocking: bool = False):
        self.code = code
        self.detail = detail
        self.penalty = penalty
        self.blocking = blocking

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "detail": self.detail,
            "penalty": self.penalty,
            "blocking": self.blocking,
        }


class SafetyFilter:
    def __init__(self, policy: Optional[SafetyPolicy] = None) -> None:
        self.policy = policy or SafetyPolicy()

    def apply(
        self, packets: List[SkillPacket]
    ) -> Tuple[List[SkillPacket], List[SkillPacket]]:
        kept, dropped = [], []
        for packet in packets:
            score, findings = self.score(packet)
            packet.safety_score = score
            packet.notes["safety_findings"] = [f.to_dict() for f in findings]
            packet.notes["risk_tier"] = risk_tier_for_domain(packet.domain).value

            blocking = [f for f in findings if f.blocking]
            if blocking:
                packet.rejection_reason = "safety_block: " + "; ".join(
                    "{} ({})".format(f.code, f.detail) for f in blocking
                )
                packet.promotion_status = PromotionStatus.REJECTED
                dropped.append(packet)
            elif score < self.policy.min_safety_score:
                packet.rejection_reason = (
                    "safety_score {:.2f} below minimum {:.2f}".format(
                        score, self.policy.min_safety_score
                    )
                )
                packet.promotion_status = PromotionStatus.REJECTED
                dropped.append(packet)
            else:
                kept.append(packet)
        return kept, dropped

    # -- scoring ----------------------------------------------------------

    def score(self, packet: SkillPacket) -> Tuple[float, List[SafetyFinding]]:
        text = " ".join(
            str(x)
            for x in (packet.sender_output, packet.notes.get("prompt"))
            if x is not None
        )
        lowered = normalize(text)
        raw = str(packet.sender_output or "")
        tier = risk_tier_for_domain(packet.domain)
        findings: List[SafetyFinding] = []

        # -- categorical, always blocking --------------------------------
        if _CREDENTIAL_RE.search(raw):
            findings.append(
                SafetyFinding(
                    "credential_leak",
                    "output contains a credential-shaped string",
                    1.0,
                    blocking=True,
                )
            )
        if any(p in lowered for p in _SELF_HARM):
            findings.append(
                SafetyFinding("self_harm", "self-harm content detected", 1.0, blocking=True)
            )
        hit = next((m for m in _INJECTION_MARKERS if m in lowered), None)
        if hit is not None:
            findings.append(
                SafetyFinding(
                    "prompt_injection",
                    "instruction-shaped content in payload ('{}'); distilled "
                    "skills are rendered into the receiver's prompt".format(hit),
                    1.0,
                    blocking=True,
                )
            )

        # -- high-risk domain rules --------------------------------------
        if tier == RiskTier.HIGH:
            if _DOSAGE_RE.search(raw):
                findings.append(
                    SafetyFinding(
                        "dosage_instruction",
                        "dosage-shaped content in a high-risk domain is prescribing",
                        1.0,
                        blocking=True,
                    )
                )
            if any(p in lowered for p in _DIAGNOSTIC_CERTAINTY):
                findings.append(
                    SafetyFinding(
                        "diagnostic_certainty",
                        "asserts a diagnosis or discourages seeking care",
                        1.0,
                        blocking=True,
                    )
                )
            if packet.domain == Domain.MEDICAL and not self._has_escalation_language(
                lowered
            ):
                findings.append(
                    SafetyFinding(
                        "missing_escalation",
                        "medical guidance with no advice to seek professional care",
                        0.4,
                    )
                )

        # -- PII ----------------------------------------------------------
        if self.policy.block_pii:
            for pattern in _PII_PATTERNS:
                if pattern.search(raw):
                    findings.append(
                        SafetyFinding(
                            "pii",
                            "personal-data-shaped string matched {}".format(
                                pattern.pattern[:24]
                            ),
                            1.0,
                            blocking=True,
                        )
                    )
                    break

        # -- soft signals ---------------------------------------------------
        halluc = metrics.hallucination_risk(
            packet.sender_output, packet.notes.get("reference")
        )
        if halluc > self.policy.max_hallucination_risk:
            findings.append(
                SafetyFinding(
                    "hallucination_risk",
                    "heuristic risk {:.2f} exceeds {:.2f}".format(
                        halluc, self.policy.max_hallucination_risk
                    ),
                    0.3,
                )
            )

        penalty = sum(f.penalty for f in findings if not f.blocking)
        score = max(0.0, min(1.0, 1.0 - penalty))
        if any(f.blocking for f in findings):
            score = 0.0
        return score, findings

    @staticmethod
    def _has_escalation_language(text: str) -> bool:
        return any(
            phrase in text
            for phrase in (
                "seek",
                "consult",
                "doctor",
                "clinician",
                "emergency",
                "medical attention",
                "healthcare",
                "hospital",
            )
        )
