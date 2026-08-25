"""REAL model connectors. ``is_mock = False`` throughout.

Importing this package is cheap and dependency-free: every heavy import
(``torch``, ``transformers``, ``sentence_transformers``) happens inside a
``load()`` call, so the package still imports on a machine that has none of them
and fails with a clear message only when you actually try to run a model.

Pick a backend by what you have:

    Ollama         laptop, 7B-14B receiver, no Python ML deps   -> OllamaConnector
    HF causal LM   full control, LoRA later, GPU box            -> HFCausalConnector
    HF seq2seq     dedicated MT teacher (NLLB covers Assamese)  -> HFSeq2SeqTranslator
"""

from typing import List, Optional

from ...core.protocol import CapabilityKey, Domain, LearningLevel, Modality
from .embedding import (  # noqa: F401
    HFEmbeddingSimilarity,
    SentenceTransformerSimilarity,
    best_available_similarity,
)
from .hf_causal import HFCausalConnector  # noqa: F401
from .hf_seq2seq import NLLB_CODES, HFSeq2SeqTranslator  # noqa: F401
from .ollama import OllamaConnectionError, OllamaConnector  # noqa: F401
from .prompting import build_messages, render_skills, system_for_capability  # noqa: F401

__all__ = [
    "HFCausalConnector",
    "HFSeq2SeqTranslator",
    "OllamaConnector",
    "OllamaConnectionError",
    "HFEmbeddingSimilarity",
    "SentenceTransformerSimilarity",
    "best_available_similarity",
    "NLLB_CODES",
    "build_messages",
    "render_skills",
    "system_for_capability",
    "translation_capability",
    "generation_capability",
    "code_capability",
    "make_qwen_ollama",
    "make_gemma_ollama",
    "make_qwen_hf",
    "make_nllb_translator",
]


# -- capability helpers -----------------------------------------------------


def translation_capability(src: str, tgt: str) -> CapabilityKey:
    return CapabilityKey(
        task_type="translate",
        modality=Modality.TEXT,
        domain=Domain.TRANSLATION,
        language="{}->{}".format(src, tgt),
    )


def generation_capability(language: str) -> CapabilityKey:
    return CapabilityKey(
        task_type="generate",
        modality=Modality.TEXT,
        domain=Domain.LANGUAGE,
        language=language,
    )


def code_capability(language: str = "python") -> CapabilityKey:
    return CapabilityKey(
        task_type="bug_fix",
        modality=Modality.CODE,
        domain=Domain.SOFTWARE,
        language=language,
    )


# -- ready-made connectors --------------------------------------------------

#: Default receiver capability set for a general instruct model.
_DEFAULT_RECEIVER_CAPS = [
    translation_capability("as", "en"),
    translation_capability("hi", "en"),
    generation_capability("en->as"),
    code_capability(),
]


def make_qwen_ollama(
    model: str = "qwen2.5:7b-instruct",
    capabilities: Optional[List[CapabilityKey]] = None,
    host: str = "http://localhost:11434",
    **kwargs,
) -> OllamaConnector:
    """Real Qwen via a local Ollama server. The recommended laptop receiver."""
    return OllamaConnector(
        model=model,
        capabilities=capabilities or list(_DEFAULT_RECEIVER_CAPS),
        module_id=kwargs.pop("module_id", "qwen-ollama"),
        display_name=kwargs.pop("display_name", "Qwen via Ollama ({})".format(model)),
        host=host,
        **kwargs,
    )


def make_gemma_ollama(
    model: str = "gemma2:9b-instruct",
    capabilities: Optional[List[CapabilityKey]] = None,
    host: str = "http://localhost:11434",
    **kwargs,
) -> OllamaConnector:
    """Real Gemma via a local Ollama server."""
    return OllamaConnector(
        model=model,
        capabilities=capabilities or list(_DEFAULT_RECEIVER_CAPS),
        module_id=kwargs.pop("module_id", "gemma-ollama"),
        display_name=kwargs.pop("display_name", "Gemma via Ollama ({})".format(model)),
        host=host,
        **kwargs,
    )


def make_qwen_hf(
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    capabilities: Optional[List[CapabilityKey]] = None,
    **kwargs,
) -> HFCausalConnector:
    """Real Qwen with weights loaded in-process.

    Use ``Qwen/Qwen2.5-0.5B-Instruct`` on a small machine. Be aware that a 0.5B
    model is a plumbing test, not a capability test.
    """
    return HFCausalConnector(
        model_id=model_id,
        capabilities=capabilities or list(_DEFAULT_RECEIVER_CAPS),
        module_id=kwargs.pop("module_id", "qwen-hf"),
        max_learning_level=kwargs.pop(
            "max_learning_level", LearningLevel.L4_PEFT_CANDIDATE
        ),
        **kwargs,
    )


def make_nllb_translator(
    model_id: str = "facebook/nllb-200-distilled-600M",
    pairs: Optional[List[str]] = None,
    **kwargs,
) -> HFSeq2SeqTranslator:
    """Real NLLB-200 translation teacher.

    ``pairs`` are 'src->tgt' tags, e.g. ``["as->en", "hi->en"]``. NLLB genuinely
    covers Assamese and Manipuri; it does NOT cover Bodo, and asking for it
    raises rather than substituting a near neighbour.
    """
    pairs = pairs or ["as->en", "hi->en"]
    capabilities = []
    for pair in pairs:
        src, tgt = pair.split("->")
        capabilities.append(translation_capability(src.strip(), tgt.strip()))
    return HFSeq2SeqTranslator(
        model_id=model_id,
        capabilities=capabilities,
        module_id=kwargs.pop("module_id", "nllb-teacher"),
        **kwargs,
    )
