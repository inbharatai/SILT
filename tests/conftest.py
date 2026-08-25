"""Shared fixtures. Everything here is deterministic and offline."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from asea.benchmarks.harness import BenchmarkSuite, load_suite  # noqa: E402
from asea.core.protocol import (  # noqa: E402
    CapabilityKey,
    Domain,
    Gap,
    Modality,
    OriginKind,
    PromotionStatus,
    Provenance,
    SkillPacket,
)

DATA = ROOT / "data" / "benchmarks"


@pytest.fixture
def data_dir() -> Path:
    return DATA


@pytest.fixture
def as_en_suite() -> BenchmarkSuite:
    return load_suite(DATA / "assamese_english.json")


@pytest.fixture
def hi_en_suite() -> BenchmarkSuite:
    return load_suite(DATA / "hindi_english.json")


@pytest.fixture
def g2p_suite() -> BenchmarkSuite:
    return load_suite(DATA / "tts_pronunciation_as.json")


@pytest.fixture
def code_suite() -> BenchmarkSuite:
    return load_suite(DATA / "coding_bugfix.json")


@pytest.fixture
def medical_suite() -> BenchmarkSuite:
    return load_suite(DATA / "medical_triage.json")


@pytest.fixture
def capability() -> CapabilityKey:
    return CapabilityKey(
        task_type="translate",
        modality=Modality.TEXT,
        domain=Domain.TRANSLATION,
        language="as->en",
    )


@pytest.fixture
def clean_provenance() -> Provenance:
    return Provenance(
        origin_kind=OriginKind.CURATED_CORPUS,
        chain=["trusted-source"],
        synthetic_depth=0,
        is_mock=False,
        source_reference="unit-test",
    )


def make_packet(capability, provenance, **overrides) -> SkillPacket:
    """Build a packet that is one field away from promotable."""
    kwargs = dict(
        task_type=capability.task_type,
        source_module="trusted-source",
        target_module="learner",
        sender_capability=capability,
        modality=capability.modality,
        language=capability.language,
        domain=capability.domain,
        provenance=provenance,
        promotion_status=PromotionStatus.DISTILLED,
    )
    kwargs.update(overrides)
    return SkillPacket(**kwargs)


@pytest.fixture
def packet_factory():
    return make_packet
