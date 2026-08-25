"""Universal metrics. Modality-agnostic; every score is 0..1.

These are the metrics the core evaluator always computes. Modality-specific
scoring is layered on top via MetricPlugin.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Iterable, Optional, Tuple

from ..core.protocol import PacketType, SkillPacket
from .similarity import normalize, tokenize

# --------------------------------------------------------------------------
# Script ranges for language preservation
# --------------------------------------------------------------------------

#: (name, first, last) Unicode code point ranges.
SCRIPT_RANGES = {
    "bengali_assamese": (0x0980, 0x09FF),  # Assamese and Bodo(Bengali script) share this
    "devanagari": (0x0900, 0x097F),        # Hindi, Bodo(Devanagari), Nepali
    "meitei_mayek": (0xABC0, 0xABFF),      # Manipuri
    "latin": (0x0041, 0x024F),
}

#: Which script a language tag is expected to be written in.
LANGUAGE_SCRIPT = {
    "as": "bengali_assamese",
    "bn": "bengali_assamese",
    "brx": "devanagari",
    "hi": "devanagari",
    "mni": "meitei_mayek",
    "en": "latin",
}


def _script_of(ch: str) -> Optional[str]:
    cp = ord(ch)
    for name, (lo, hi) in SCRIPT_RANGES.items():
        if lo <= cp <= hi:
            return name
    return None


def target_language(language: Optional[str]) -> Optional[str]:
    """Extract the output language from a tag like 'as->en' or plain 'as'."""
    if not language:
        return None
    if "->" in language:
        return language.split("->")[-1].strip()
    return language.strip()


def language_preservation(text: Any, language: Optional[str]) -> float:
    """Fraction of letters written in the script the target language expects.

    Catches the single most common low-resource failure mode: a model asked for
    Assamese silently answering in English or Hindi. Punctuation, digits and
    whitespace are ignored; if the string has no letters at all we return 1.0
    rather than punishing a legitimately numeric answer.
    """
    tgt = target_language(language)
    expected = LANGUAGE_SCRIPT.get(tgt) if tgt else None
    if expected is None:
        return 1.0  # unknown or script-agnostic target: no opinion
    s = unicodedata.normalize("NFC", str(text or ""))
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 1.0
    hits = sum(1 for c in letters if _script_of(c) == expected)
    return hits / len(letters)


# --------------------------------------------------------------------------
# Schema compliance
# --------------------------------------------------------------------------

#: Required keys per packet type. A distilled payload that omits them is
#: structurally unusable by a receiver regardless of how good its content is.
REQUIRED_KEYS: Dict[PacketType, Tuple[str, ...]] = {
    PacketType.GLOSSARY: ("entries",),
    PacketType.CORRECTION_PAIR: ("pairs",),
    PacketType.EXEMPLAR: ("examples",),
    PacketType.RULE: ("rules",),
    PacketType.LEXICON: ("entries",),
}


def schema_compliance(packet: SkillPacket) -> float:
    """1.0 only if the distilled payload is well-formed and non-empty."""
    if packet.packet_type is None or packet.distilled_skill is None:
        return 0.0
    required = REQUIRED_KEYS.get(packet.packet_type, ())
    if not required:
        return 1.0
    present = 0
    for key in required:
        value = packet.distilled_skill.get(key)
        if isinstance(value, (list, dict)) and len(value) > 0:
            present += 1
        elif isinstance(value, str) and value.strip():
            present += 1
    return present / len(required)


# --------------------------------------------------------------------------
# Hallucination heuristics
# --------------------------------------------------------------------------

#: Phrases that signal unearned certainty or fabricated authority. Crude by
#: design: this is a tripwire, not a classifier, and it is documented as such.
HEDGE_ABSENT_ABSOLUTES = (
    "always",
    "never",
    "guaranteed",
    "100%",
    "completely safe",
    "cure",
    "no risk",
    "definitely",
)

FABRICATION_MARKERS = (
    "according to a study",
    "studies show",
    "as everyone knows",
    "it is well known",
    "research proves",
)

_CITATION_RE = re.compile(r"\[\d+\]|\(\d{4}\)|et al\.", re.IGNORECASE)


def hallucination_risk(
    output: Any,
    reference: Optional[Any] = None,
    unsupported_token_threshold: float = 0.6,
) -> float:
    """Heuristic 0..1 where 1.0 is maximum risk.

    Three weak signals, deliberately conservative:
      1. absolute claims with no hedging,
      2. vague appeals to authority,
      3. when a reference exists, a high proportion of output tokens absent
         from it (content invented out of nothing).

    This does NOT detect confident, fluent, plausible falsehoods that share
    vocabulary with the reference. That failure mode requires a real verifier
    and is the biggest open risk in the system. See risk_report.md.
    """
    text = normalize(output)
    if not text:
        return 1.0

    risk = 0.0
    if any(marker in text for marker in HEDGE_ABSENT_ABSOLUTES):
        risk += 0.35
    if any(marker in text for marker in FABRICATION_MARKERS):
        risk += 0.35
    if _CITATION_RE.search(str(output or "")) and reference is None:
        # A citation-shaped string with nothing to check it against.
        risk += 0.2

    if reference is not None:
        ref_tokens = set(tokenize(str(reference)))
        out_tokens = tokenize(text)
        if out_tokens and ref_tokens:
            unsupported = sum(1 for t in out_tokens if t not in ref_tokens) / len(out_tokens)
            if unsupported > unsupported_token_threshold:
                risk += 0.3 * (unsupported - unsupported_token_threshold) / (
                    1.0 - unsupported_token_threshold
                )
    return max(0.0, min(1.0, risk))


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "schema_compliance": 0.20,
    "semantic_similarity": 0.25,
    "task_success": 0.30,
    "language_preservation": 0.15,
    "hallucination_penalty": 0.10,
}


def aggregate(
    schema: float,
    similarity: float,
    task: float,
    language: float,
    halluc_risk: float,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    total = sum(w.values())
    score = (
        w["schema_compliance"] * schema
        + w["semantic_similarity"] * similarity
        + w["task_success"] * task
        + w["language_preservation"] * language
        + w["hallucination_penalty"] * (1.0 - halluc_risk)
    ) / total
    return max(0.0, min(1.0, score))


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0
