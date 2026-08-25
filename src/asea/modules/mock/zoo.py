"""The bundled MOCK module zoo.

Each factory returns a MockModule shaped like the real thing it stands in for:
same roles, same declared capabilities, same learning-level ceiling. Knowledge
tables are injected by the caller (normally from a benchmark suite), so no
language data is hardcoded here.

MOCK STATUS: every module below returns ``is_mock = True``. See base.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...core.protocol import CapabilityKey, Domain, LearningLevel, Modality
from .base import MockModule

# --------------------------------------------------------------------------
# Capability shorthands
# --------------------------------------------------------------------------


def text_cap(task: str, language: str, domain: Domain = Domain.LANGUAGE) -> CapabilityKey:
    return CapabilityKey(
        task_type=task, modality=Modality.TEXT, domain=domain, language=language
    )


def code_cap(task: str = "bug_fix") -> CapabilityKey:
    return CapabilityKey(
        task_type=task, modality=Modality.CODE, domain=Domain.SOFTWARE, language="python"
    )


def tts_cap(language: str) -> CapabilityKey:
    return CapabilityKey(
        task_type="grapheme_to_phoneme",
        modality=Modality.SPEECH_TTS,
        domain=Domain.PRONUNCIATION,
        language=language,
    )


def asr_cap(language: str) -> CapabilityKey:
    return CapabilityKey(
        task_type="transcribe",
        modality=Modality.AUDIO_ASR,
        domain=Domain.LANGUAGE,
        language=language,
    )


def rule_cap(domain: Domain, task: str = "triage") -> CapabilityKey:
    return CapabilityKey(
        task_type=task, modality=Modality.STRUCTURED, domain=domain, language="en"
    )


# --------------------------------------------------------------------------
# Factories
# --------------------------------------------------------------------------


def make_qwen(
    knowledge: Optional[Dict[str, Dict[str, Any]]] = None,
    fallback: str = "echo",
) -> MockModule:
    """MOCK stand-in for a Qwen-family instruct model.

    Modelled as strong on code and English, weak on Assamese -- which is the
    real-world shape this project exists to address. Acts as both sender (code)
    and receiver (Assamese).
    """
    return MockModule(
        module_id="qwen-mock",
        display_name="Qwen (MOCK)",
        roles=["sender", "receiver"],
        capabilities=[
            text_cap("translate", "as->en"),
            text_cap("translate", "hi->en"),
            text_cap("generate", "en", Domain.GENERAL),
            code_cap(),
        ],
        knowledge=knowledge,
        fallback=fallback,
        max_learning_level=LearningLevel.L4_PEFT_CANDIDATE,
    )


def make_gemma(
    knowledge: Optional[Dict[str, Dict[str, Any]]] = None,
    fallback: str = "english",
) -> MockModule:
    """MOCK stand-in for a Gemma-family model.

    Given the ``english`` fallback so it demonstrates the silent
    wrong-language failure the language_preservation metric is built to catch.
    """
    return MockModule(
        module_id="gemma-mock",
        display_name="Gemma (MOCK)",
        roles=["sender", "receiver"],
        capabilities=[
            text_cap("translate", "as->en"),
            text_cap("translate", "hi->en"),
            text_cap("generate", "en", Domain.GENERAL),
            code_cap(),
        ],
        knowledge=knowledge,
        fallback=fallback,
        max_learning_level=LearningLevel.L3_SKILL_PACKET,
    )


def make_ai4bharat_asr(
    knowledge: Optional[Dict[str, Dict[str, Any]]] = None,
) -> MockModule:
    """MOCK stand-in for an AI4Bharat ASR model (IndicConformer family).

    Sender only: an ASR model is a source of transcription competence, not a
    consumer of text skill packets.
    """
    return MockModule(
        module_id="ai4bharat-asr-mock",
        display_name="AI4Bharat ASR (MOCK)",
        roles=["sender"],
        capabilities=[asr_cap("as"), asr_cap("hi"), asr_cap("brx"), asr_cap("mni")],
        knowledge=knowledge,
        fallback="unknown",
        consumes_skills=False,
    )


def make_ai4bharat_tts(
    knowledge: Optional[Dict[str, Dict[str, Any]]] = None,
) -> MockModule:
    """MOCK stand-in for an AI4Bharat TTS/G2P source (IndicTTS family).

    Sender only, and symbolic only: this supplies grapheme-to-phoneme and
    lexicon knowledge. Voice identity is not transferable through this adapter.
    """
    return MockModule(
        module_id="ai4bharat-tts-mock",
        display_name="AI4Bharat TTS/G2P (MOCK)",
        roles=["sender"],
        # Language tags carry the '-ipa' suffix because a G2P front-end emits
        # IPA in Latin script, not the source script. Tagging it 'as' would make
        # the language-preservation metric demand Assamese characters in a
        # phoneme string and fail every correct answer.
        capabilities=[tts_cap("as-ipa"), tts_cap("hi-ipa")],
        knowledge=knowledge,
        fallback="unknown",
        consumes_skills=False,
    )


def make_generic_sender(
    module_id: str = "generic-sender-mock",
    capabilities: Optional[List[CapabilityKey]] = None,
    knowledge: Optional[Dict[str, Dict[str, Any]]] = None,
    display_name: str = "Generic Sender (MOCK)",
) -> MockModule:
    """Template for any expert source: a curated corpus, an API, a human panel."""
    return MockModule(
        module_id=module_id,
        display_name=display_name,
        roles=["sender"],
        capabilities=capabilities or [text_cap("translate", "as->en")],
        knowledge=knowledge,
        fallback="unknown",
        consumes_skills=False,
        base_confidence=0.8,
    )


def make_generic_receiver(
    module_id: str = "generic-receiver-mock",
    capabilities: Optional[List[CapabilityKey]] = None,
    knowledge: Optional[Dict[str, Dict[str, Any]]] = None,
    fallback: str = "echo",
    consumes_skills: bool = True,
    display_name: str = "Generic Receiver (MOCK)",
) -> MockModule:
    """Template for any learner module."""
    return MockModule(
        module_id=module_id,
        display_name=display_name,
        roles=["receiver"],
        capabilities=capabilities or [text_cap("translate", "as->en")],
        knowledge=knowledge,
        fallback=fallback,
        consumes_skills=consumes_skills,
    )
