"""Real connectors: everything testable without loading weights.

Deliberately split in two:

* the bulk of these tests exercise prompt construction, manifests, language-code
  handling and error paths -- all deterministic, offline, no model download;
* the handful that need actual weights are gated behind ``ASEA_RUN_REAL=1`` so a
  normal ``pytest`` run stays fast and offline.

Run the real ones with::

    ASEA_RUN_REAL=1 python3 -m pytest tests/test_real_connectors.py -q
"""

from __future__ import annotations

import os

import pytest

from asea.core.protocol import CapabilityKey, Domain, Modality
from asea.modules.real import (
    HFCausalConnector,
    HFEmbeddingSimilarity,
    HFSeq2SeqTranslator,
    OllamaConnectionError,
    OllamaConnector,
    code_capability,
    generation_capability,
    make_nllb_translator,
    make_qwen_hf,
    make_qwen_ollama,
    translation_capability,
)
from asea.modules.real.prompting import (
    build_messages,
    render_skills,
    split_pair,
    system_for_capability,
)

REAL = pytest.mark.skipif(
    os.environ.get("ASEA_RUN_REAL") != "1",
    reason="needs real weights; set ASEA_RUN_REAL=1",
)


# -- prompting --------------------------------------------------------------


@pytest.mark.parametrize(
    "tag,expected", [("as->en", ("as", "en")), ("as", (None, "as")), (None, (None, None))]
)
def test_split_pair(tag, expected):
    assert split_pair(tag) == expected


def test_translation_instruction_names_both_languages():
    text = system_for_capability(translation_capability("as", "en"))
    assert "Assamese" in text and "English" in text


def test_generation_instruction_pins_output_language():
    text = system_for_capability(generation_capability("en->as"))
    assert "Assamese" in text


def test_code_instruction_forbids_fences():
    assert "no code fences" in system_for_capability(code_capability())


def test_high_risk_instruction_forbids_diagnosis_and_dosage():
    cap = CapabilityKey(
        task_type="triage", modality=Modality.STRUCTURED, domain=Domain.MEDICAL, language="en"
    )
    text = system_for_capability(cap)
    assert "Never diagnose" in text and "dosage" in text


def test_render_skills_handles_every_payload_shape():
    rendered = render_skills([
        {"distilled_skill": {"entries": [{"source": "ভাত", "target": "rice"}]}},
        {"distilled_skill": {"entries": [{"grapheme": "ক", "phoneme": "k"}]}},
        {"distilled_skill": {"examples": [{"buggy": "== None", "fixed": "is None"}]}},
        {"distilled_skill": {"rules": [{"condition": "chest pain", "action": "escalate"}]}},
        {"distilled_skill": {"pairs": [{"observed": "rn", "corrected": "m"}]}},
    ])
    assert "ভাত = rice" in rendered
    assert "ক -> k" in rendered
    assert "wrong: == None" in rendered
    assert "if [chest pain] then: escalate" in rendered
    assert "rn => m" in rendered


def test_render_skills_empty_is_empty():
    assert render_skills([]) == ""
    assert render_skills([{"distilled_skill": {}}]) == ""


def test_build_messages_omits_reference_block_when_no_skills():
    messages = build_messages(translation_capability("as", "en"), "মই")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "মই"}


def test_build_messages_injects_reference_block():
    # Skills fold into the SINGLE system message (not a second system turn):
    # several HF chat templates honour only the first system message, so a
    # second one would be silently dropped and skills un-injected.
    messages = build_messages(
        translation_capability("as", "en"), "মই ভাত খাওঁ",
        [{"distilled_skill": {"entries": [{"source": "ভাত", "target": "rice"}]}}],
    )
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "ভাত = rice" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "মই ভাত খাওঁ"}


def test_prompt_construction_cannot_leak_raw_sender_output():
    """The prompt builder reads only distilled_skill, by construction."""
    poisoned = [{
        "distilled_skill": {"entries": [{"source": "a", "target": "b"}]},
        "sender_output": "RAW TEACHER TEXT THAT MUST NOT REACH THE STUDENT",
    }]
    messages = build_messages(translation_capability("as", "en"), "a", poisoned)
    assert "RAW TEACHER TEXT" not in " ".join(m["content"] for m in messages)


# -- manifests and honesty --------------------------------------------------


def test_all_real_connectors_report_not_mock():
    for module in (
        make_qwen_ollama(),
        make_qwen_hf("Qwen/Qwen2.5-0.5B-Instruct"),
        make_nllb_translator(),
    ):
        assert module.is_mock is False
        assert module.manifest().is_mock is False


def test_manifest_version_records_the_actual_model():
    assert make_nllb_translator().manifest().version == "facebook/nllb-200-distilled-600M"
    assert make_qwen_ollama(model="qwen2.5:7b-instruct").manifest().version == (
        "ollama:qwen2.5:7b-instruct"
    )


def test_translator_registers_as_sender_only():
    """An MT model has no prompt channel for skills, so it must not be a receiver."""
    manifest = make_nllb_translator().manifest()
    assert manifest.roles == ["sender"]


def test_translator_does_not_pretend_to_consume_skills():
    translator = make_nllb_translator()
    assert HFSeq2SeqTranslator.infer_with_skills is type(translator).__mro__[1].infer_with_skills


def test_real_modules_register_cleanly():
    from asea.registry.registries import ModuleRegistry

    registry = ModuleRegistry()
    registry.register_module(make_nllb_translator())
    registry.register_module(make_qwen_ollama())
    assert len(registry) == 2


# -- language codes ---------------------------------------------------------


def test_nllb_maps_assamese_and_manipuri():
    translator = make_nllb_translator()
    assert translator._code("as") == "asm_Beng"
    assert translator._code("mni") == "mni_Beng"
    assert translator._code("en") == "eng_Latn"


def test_nllb_refuses_bodo_loudly_instead_of_substituting():
    """Silently translating into a neighbouring language would poison packets."""
    with pytest.raises(ValueError, match="Bodo is not covered"):
        make_nllb_translator()._code("brx")


def test_nllb_refuses_unknown_language():
    with pytest.raises(ValueError, match="no language code mapping"):
        make_nllb_translator()._code("xyz")


def test_translator_requires_a_language_pair():
    translator = make_nllb_translator()
    bad = CapabilityKey(task_type="translate", modality=Modality.TEXT, language="as")
    with pytest.raises(ValueError, match="src->tgt"):
        translator.infer(bad, "মই")


# -- output cleaning --------------------------------------------------------


@pytest.mark.parametrize(
    "raw,clean",
    [
        ("```python\nif x is None:\n```", "if x is None:"),
        ('"I eat rice"', "I eat rice"),
        ("'I eat rice'", "I eat rice"),
        ("  spaced  ", "spaced"),
        ("plain", "plain"),
    ],
)
def test_causal_output_cleaning(raw, clean):
    assert HFCausalConnector._clean(raw) == clean


def test_ollama_output_cleaning_matches():
    assert OllamaConnector._clean("```\nfoo\n```") == "foo"
    assert OllamaConnector._clean(None) == ""


# -- failure modes ----------------------------------------------------------


def test_ollama_reports_unreachable_server_clearly():
    connector = make_qwen_ollama(host="http://127.0.0.1:9")  # discard port
    with pytest.raises(OllamaConnectionError, match="cannot reach Ollama"):
        connector.health()


def test_ollama_infer_surfaces_connection_error():
    connector = make_qwen_ollama(host="http://127.0.0.1:9", timeout=2)
    with pytest.raises(OllamaConnectionError):
        connector.infer(translation_capability("as", "en"), "মই")


def test_connectors_do_not_load_weights_on_construction():
    """Construction must stay cheap; loading is explicit."""
    connector = make_qwen_hf("Qwen/Qwen2.5-0.5B-Instruct")
    assert connector._model is None
    embedder = HFEmbeddingSimilarity()
    assert embedder._model is None


def test_embedding_backend_declares_itself_semantic():
    assert HFEmbeddingSimilarity().is_semantic is True


# -- real weights (opt-in) --------------------------------------------------


@REAL
def test_real_nllb_translates_assamese():
    translator = make_nllb_translator(dtype="float32")
    output = translator.infer(translation_capability("as", "en"), "মই ভাত খাওঁ")
    assert "rice" in output.lower()


@REAL
def test_real_qwen_generates_and_reports_confidence():
    receiver = make_qwen_hf("Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens=16)
    capability = translation_capability("as", "en")
    output = receiver.infer(capability, "মোৰ নাম ৰাম")
    assert isinstance(output, str) and output
    confidence = receiver.confidence(capability, "মোৰ নাম ৰাম", output)
    assert 0.0 < confidence <= 1.0


@REAL
def test_real_embedding_beats_lexical_on_word_order():
    """The concrete reason to upgrade similarity."""
    from asea.evaluator.similarity import LexicalSimilarity

    lexical = LexicalSimilarity()
    assert lexical.similarity("I eat rice", "rice eat I") > 0.9  # wrong, order-blind

    embedder = HFEmbeddingSimilarity()
    unrelated = embedder.similarity("I eat rice", "the server rejected the request")
    same = embedder.similarity("I eat rice", "I am eating rice")
    assert same > unrelated
