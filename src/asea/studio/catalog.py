"""Studio module catalog -- REAL connectors only.

This is the enforcement point for the platform's founding constraint: SILT
Studio never serves a mock. Every entry here constructs a module whose
``is_mock`` is False, and :func:`build` re-verifies that at construction time
so a future edit cannot quietly smuggle a lookup table into the UI.

Mocks still exist in the package for unit tests -- they are simply not
reachable through the Studio.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..core.interfaces import ModuleAdapter
from ..core.protocol import CapabilityKey, Domain, Modality

ROOT = Path(__file__).resolve().parent.parent.parent.parent
CORPORA = ROOT / "data" / "corpora"


def _translation(src: str, tgt: str) -> CapabilityKey:
    return CapabilityKey(
        task_type="translate", modality=Modality.TEXT,
        domain=Domain.TRANSLATION, language="{}->{}".format(src, tgt),
    )


TRIAGE = CapabilityKey(
    task_type="triage", modality=Modality.STRUCTURED,
    domain=Domain.MEDICAL, language="en",
)


def _g2p_as() -> CapabilityKey:
    """Assamese grapheme-to-phoneme (text char -> IPA), the symbolic
    pronunciation-knowledge layer of TTS. Output language is 'as-ipa' because
    the result is IPA in Latin script."""
    return CapabilityKey(
        task_type="grapheme_to_phoneme", modality=Modality.SPEECH_TTS,
        domain=Domain.PRONUNCIATION, language="as-ipa",
    )


def _nllb() -> ModuleAdapter:
    from ..modules.real import make_nllb_translator

    return make_nllb_translator(pairs=["as->en", "hi->en"], dtype="float32")


def _qwen05(roles: Optional[List[str]] = None) -> ModuleAdapter:
    from ..modules.real import HFCausalConnector

    return HFCausalConnector(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        capabilities=[_translation("as", "en"), _translation("hi", "en"), TRIAGE],
        roles=roles or ["sender", "receiver"],
        module_id="qwen2.5-0.5b",
        display_name="Qwen2.5-0.5B-Instruct (HF, real weights)",
        max_new_tokens=48,
    )


def _smollm() -> ModuleAdapter:
    from ..modules.real import HFCausalConnector

    return HFCausalConnector(
        model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        capabilities=[_translation("as", "en"), _translation("hi", "en"), TRIAGE],
        roles=["receiver"],
        module_id="smollm2-360m",
        display_name="SmolLM2-360M-Instruct (HF, real weights)",
        max_new_tokens=48,
    )


def _triage_corpus() -> ModuleAdapter:
    from ..modules.real.corpus import CorpusSender

    return CorpusSender(
        corpus_path=CORPORA / "triage_redflags.json",
        capabilities=[TRIAGE],
        module_id="triage-corpus",
        display_name="Curated triage corpus (real, file-backed; clinically unreviewed sample)",
    )


def _ollama(
    model: str,
    roles: Optional[List[str]] = None,
    capabilities: Optional[List[CapabilityKey]] = None,
    think: Optional[bool] = None,
) -> Callable[[], ModuleAdapter]:
    def factory() -> ModuleAdapter:
        from ..modules.real import OllamaConnector

        return OllamaConnector(
            model=model,
            capabilities=capabilities or [_translation("as", "en"), _translation("hi", "en"), TRIAGE],
            roles=roles or ["sender", "receiver"],
            think=think,
        )

    return factory


#: id -> (factory, human description, note on requirements)
CATALOG: Dict[str, Dict[str, Any]] = {
    "nllb-teacher": {
        "factory": _nllb,
        "roles": ["sender"],
        "description": "NLLB-200-distilled-600M translation teacher (covers Assamese)",
        "requires": "torch + transformers; ~2.5GB download on first use",
        "builtin": True,
    },
    "qwen2.5-0.5b": {
        "factory": _qwen05,
        "roles": ["sender", "receiver"],
        "description": "Qwen2.5-0.5B-Instruct, in-process HF weights",
        "requires": "torch + transformers; plumbing-scale receiver",
        "builtin": True,
    },
    "smollm2-360m": {
        "factory": _smollm,
        "roles": ["receiver"],
        "description": "SmolLM2-360M-Instruct, in-process HF weights",
        "requires": "torch + transformers; plumbing-scale receiver",
        "builtin": True,
    },
    "triage-corpus": {
        "factory": _triage_corpus,
        "roles": ["sender"],
        "description": "Curated triage red-flag corpus (real file-backed source)",
        "requires": "nothing; NOT medical advice, clinically unreviewed sample",
        "builtin": True,
    },
    "qwen2.5-7b-ollama": {
        "factory": _ollama("qwen2.5:7b-instruct"),
        "roles": ["sender", "receiver"],
        "description": "Qwen2.5 7B via local Ollama (the serious receiver)",
        "requires": "ollama serve + ollama pull qwen2.5:7b-instruct",
        "builtin": True,
    },
    "gemma2-9b-ollama": {
        "factory": _ollama("gemma2:9b"),
        "roles": ["sender", "receiver"],
        "description": "Gemma2 9B via local Ollama",
        "requires": "ollama serve + ollama pull gemma2:9b",
        "builtin": True,
    },
    # TTS-pronunciation (grapheme->phoneme) pair: a real Assamese-pronunciation
    # teacher and a receiver that is genuinely weak at Assamese G2P. Both
    # advertise the G2P capability so the handshake finds a shared capability on
    # the tts_pronunciation_as suite. Catalog ADDITIONS only; no
    # core/policy/gate edit. NOTE: this is grapheme->IPA (text), the symbolic
    # TTS layer, not audio synthesis; the suite is an illustrative sample, not a
    # lexicon. ``think=False`` is essential for the GLM teacher: it is a
    # reasoning model whose answer lives in the ``content`` field only when
    # thinking is disabled, otherwise the connector sees an empty string.
    "tts-teacher-as": {
        "factory": _ollama("glm-5.2:cloud", roles=["sender"], capabilities=[_g2p_as()], think=False),
        "roles": ["sender"],
        "description": "GLM (thinking model, think disabled) as an Assamese G2P teacher; real weights",
        "requires": "ollama serve + ollama pull glm-5.2:cloud",
        "builtin": True,
    },
    "tts-learner-zero": {
        "factory": _ollama("qwen3.5:latest", roles=["receiver"], capabilities=[_g2p_as()], think=False),
        "roles": ["receiver"],
        "description": "Qwen3.5 as a receiver weaker than the GLM teacher at exact Assamese G2P; learns from the verified table. think disabled (reasoning model)",
        "requires": "ollama serve + ollama pull qwen3.5:latest",
        "builtin": True,
    },
}

# --- User-added receiver: GLM via local Ollama (real weights, user's model) ---
# Catalog ADDITION only; no existing code, core module, or policy modified.
CATALOG["glm-ollama"] = {
    "factory": _ollama("glm-5.2:cloud"),
    "roles": ["sender", "receiver"],
    "description": "GLM via local Ollama (user's model, real weights)",
    "requires": "ollama serve + the tag from `ollama list`",
    "builtin": True,
}

_cache: Dict[str, ModuleAdapter] = {}

# Thread safety (adversarial audit 2026-08-13 #39): build()'s check-then-act
# cache and listing()'s CATALOG.items() iteration race with add_catalog_entry's
# CATALOG mutation under FastAPI's threadpool + per-job worker threads. The
# worst realistic outcome is a wasted weights load or a
# "dictionary changed size during iteration" RuntimeError on GET /api/catalog;
# the approved store is never touched by this path. The lock serializes the
# cache check/construct/assign and the catalog iteration/mutation within one
# process (the default single-worker uvicorn deployment).
_lock = threading.Lock()


def listing() -> List[Dict[str, Any]]:
    with _lock:
        items = list(CATALOG.items())
    return [
        {"id": key, "roles": entry["roles"], "description": entry["description"],
         "requires": entry["requires"]}
        for key, entry in items
    ]


def build(module_id: str) -> ModuleAdapter:
    """Construct (and cache) a catalog module. Refuses mocks, structurally."""
    # Fast path: a read under the lock is cheap and avoids a torn cache entry.
    with _lock:
        if module_id in _cache:
            return _cache[module_id]
        entry = CATALOG.get(module_id)
        if entry is None:
            raise KeyError(
                "unknown module '{}'; catalog: {}".format(module_id, sorted(CATALOG))
            )
    # Construction loads weights / probes Ollama and can take seconds: do it
    # OUTSIDE the lock so concurrent builds of *different* modules don't
    # serialise. The double-check below closes the narrow window where two
    # callers for the SAME module both miss and both construct.
    module = entry["factory"]()
    if module.is_mock or module.manifest().is_mock:
        # Belt and braces: the Studio must be impossible to point at a mock.
        raise RuntimeError(
            "catalog integrity violation: '{}' constructed a mock".format(module_id)
        )
    with _lock:
        if module_id in _cache:
            # Another thread won the race; discard our build and use theirs.
            return _cache[module_id]
        _cache[module_id] = module
    return module
