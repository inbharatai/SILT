"""Conformance: is the adapter actually universal, or just one flow with
abstraction painted on?

A "universal adapter" claim is only worth something if dissimilar sender/receiver
pairs traverse the *same* orchestration. These tests assert that property three
ways:

  1. the pipeline source contains no modality branching,
  2. four different modality pairs produce the identical audit event sequence,
  3. a brand-new modality can be added by registration alone, with no edit to
     any core module.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List

import pytest

from asea.core import pipeline as pipeline_module
from asea.core.interfaces import Distiller, Extractor
from asea.core.pipeline import Pipeline
from asea.core.plugins import PluginRegistry, default_registry
from asea.core.protocol import (
    CapabilityKey,
    Domain,
    Modality,
    PacketType,
    PromotionStatus,
)
from asea.distill.strategies import BaseDistiller
from asea.extraction.extractors import BaseExtractor
from asea.modules.mock.zoo import (
    code_cap,
    make_generic_receiver,
    make_generic_sender,
    rule_cap,
    text_cap,
    tts_cap,
)
from asea.promotion.gate import PromotionGate, PromotionPolicy


# -- 1. structural -----------------------------------------------------------


def test_pipeline_source_has_no_modality_branching():
    """If the core has to ask 'which modality is this?', it is not universal."""
    source = inspect.getsource(pipeline_module)
    offenders = [
        line.strip()
        for line in source.splitlines()
        if ("Modality." in line or ".modality ==" in line)
        and line.strip().startswith(("if ", "elif ", "match ", "case "))
    ]
    assert offenders == [], "core pipeline branches on modality: {}".format(offenders)


def test_core_modules_do_not_import_concrete_plugins():
    """Core depends on interfaces; concrete strategies are resolved at runtime."""
    for module_name in ("protocol", "interfaces", "handshake", "gap"):
        source = inspect.getsource(
            __import__("asea.core.{}".format(module_name), fromlist=["x"])
        )
        assert "from ..distill.strategies" not in source
        assert "from ..extraction.extractors" not in source


# -- 2. behavioural ----------------------------------------------------------


def _pair(cap, pairs, receiver_fallback="echo"):
    sender = make_generic_sender(
        module_id="sender-{}".format(cap.modality.value),
        capabilities=[cap],
        knowledge={cap.as_str(): dict(pairs)},
    )
    receiver = make_generic_receiver(
        module_id="receiver-{}".format(cap.modality.value),
        capabilities=[cap],
        fallback=receiver_fallback,
    )
    return sender, receiver


def _suite(cap, extraction, heldout, suite_id):
    from asea.benchmarks.harness import BenchmarkCase, BenchmarkSuite

    cases = [
        BenchmarkCase(case_id="e{}".format(i), prompt=p, expected=e, split="extraction",
                      meta={"human_verified": True})
        for i, (p, e) in enumerate(extraction)
    ] + [
        BenchmarkCase(case_id="h{}".format(i), prompt=p, expected=e, split="heldout",
                      meta={"human_verified": True})
        for i, (p, e) in enumerate(heldout)
    ]
    return BenchmarkSuite(
        suite_id=suite_id,
        task_type=cap.task_type,
        modality=cap.modality,
        domain=cap.domain,
        language=cap.language,
        cases=cases,
    )


#: Four genuinely dissimilar transfers: words, phonemes, code fragments, rules.
SCENARIOS = {
    "text": (
        text_cap("translate", "as->en", Domain.TRANSLATION),
        [("ভাত", "rice"), ("পানী", "water"), ("খাওঁ", "eat"), ("মই", "I")],
        [("মই ভাত খাওঁ", "I eat rice")],
    ),
    "speech_tts": (
        tts_cap("as-ipa"),
        [("ক", "k"), ("া", "a"), ("ম", "m")],
        [("কাম", "kam")],
    ),
    "code": (
        code_cap(),
        [("== None", "is None")],
        [("if x == None: pass", "if x is None: pass")],
    ),
    "structured": (
        rule_cap(Domain.EDUCATION, "explain"),
        [("photosynthesis", "Plants convert light into chemical energy.")],
        [("Explain photosynthesis to a beginner.",
          "Plants convert light into chemical energy.")],
    ),
}


def _run(tmp_path, name):
    cap, extraction, heldout = SCENARIOS[name]
    sender, receiver = _pair(cap, extraction)
    suite = _suite(cap, extraction, heldout, "conformance_{}".format(name))

    pipeline = Pipeline(
        workspace=tmp_path / name,
        gate=PromotionGate(PromotionPolicy(strict_no_mock=False)),
    )
    pipeline.register_module(sender)
    pipeline.register_module(receiver)
    pipeline.bind_adapter("adapter-{}".format(name), sender.module_id, receiver.module_id)
    report = pipeline.run("adapter-{}".format(name), suites=[suite])
    return pipeline, report


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_every_modality_completes_the_same_pipeline(tmp_path, name):
    pipeline, report = _run(tmp_path, name)
    events = [e["event"] for e in pipeline.audit.entries()]
    assert events == [
        "module_registered", "module_registered", "adapter_bound",
        "session_opened", "gap_negotiated", "extracted",
        "relevance_filtered", "safety_filtered", "distilled",
        "evaluated", "gate_decision", "promoted", "run_complete",
    ], "modality '{}' took a different path through the core".format(name)
    assert report.promoted, "modality '{}' produced no promotion".format(name)
    assert pipeline.audit.verify()["ok"] is True


def test_all_modalities_share_one_packet_schema(tmp_path):
    """One envelope, four payload shapes."""
    types = {}
    for name in SCENARIOS:
        pipeline, report = _run(tmp_path, name)
        packet = report.distilled[0]
        types[name] = packet.packet_type
        # Same envelope fields regardless of modality.
        assert packet.provenance.chain
        assert packet.sender_output is None
        assert packet.promotion_status == PromotionStatus.PROMOTED
        assert packet.rollback_token
    assert len(set(types.values())) > 1, "payload shapes should differ by modality"
    assert types["text"] == PacketType.GLOSSARY
    assert types["speech_tts"] == PacketType.LEXICON


# -- 3. extensibility --------------------------------------------------------


def test_new_modality_needs_no_core_edit(tmp_path):
    """Register an OCR extractor/distiller and run it. Core is untouched."""

    class OcrExtractor(BaseExtractor):
        modality = Modality.OCR

    class OcrDistiller(BaseDistiller):
        modality = Modality.OCR
        packet_type = PacketType.CORRECTION_PAIR

        def build_payload(self, group) -> Dict[str, Any]:
            return {
                "pairs": [
                    {"observed": p.notes.get("prompt"), "corrected": self.taught_value(p)}
                    for p in group
                ]
            }

    plugins = default_registry()
    plugins.register_extractor(OcrExtractor())
    plugins.register_distiller(OcrDistiller())
    assert plugins.modalities()["ocr"] == {
        "extractor": True, "distiller": True, "metric": False
    }

    cap = CapabilityKey(
        task_type="correct_ocr", modality=Modality.OCR, domain=Domain.GENERAL, language="en"
    )
    pairs = [("rn", "m"), ("cl", "d"), ("0", "O")]
    sender, receiver = _pair(cap, pairs)
    suite = _suite(cap, pairs, [("rn", "m")], "conformance_ocr")

    pipeline = Pipeline(
        workspace=tmp_path / "ocr",
        plugins=plugins,
        gate=PromotionGate(PromotionPolicy(strict_no_mock=False)),
    )
    pipeline.register_module(sender)
    pipeline.register_module(receiver)
    pipeline.bind_adapter("ocr", sender.module_id, receiver.module_id)
    report = pipeline.run("ocr", suites=[suite])

    assert report.distilled
    assert report.distilled[0].packet_type == PacketType.CORRECTION_PAIR


def test_missing_plugin_is_reported_not_crashed(tmp_path):
    """An unregistered modality must degrade cleanly and leave an audit note."""
    empty = PluginRegistry()
    cap = text_cap("translate", "as->en", Domain.TRANSLATION)
    sender, receiver = _pair(cap, [("ভাত", "rice"), ("পানী", "water")])
    suite = _suite(cap, [("ভাত", "rice"), ("পানী", "water")], [("ভাত", "rice")], "s")

    pipeline = Pipeline(workspace=tmp_path / "empty", plugins=empty)
    pipeline.register_module(sender)
    pipeline.register_module(receiver)
    pipeline.bind_adapter("a", sender.module_id, receiver.module_id)
    report = pipeline.run("a", suites=[suite])

    assert report.distilled == []
    assert any(e["event"] == "plugin_missing" for e in pipeline.audit.entries())


def test_pipeline_uses_one_similarity_backend_everywhere(tmp_path):
    """A run must not filter with one metric and score with another.

    This was a real bug: injecting an embedding backend reached the harness but
    not the relevance filter, so a run reported as semantically evaluated was
    still filtered by the lexical proxy.
    """
    from asea.core.interfaces import SimilarityBackend

    class Marker(SimilarityBackend):
        def similarity(self, a, b):
            return 1.0

        @property
        def is_semantic(self):
            return True

    marker = Marker()
    pipeline = Pipeline(workspace=tmp_path / "sim", similarity=marker)
    assert pipeline.harness.similarity is marker
    assert pipeline.relevance.similarity is marker
    assert pipeline.evaluator.harness.similarity is marker
    assert pipeline.gap_engine.harness.similarity is marker
