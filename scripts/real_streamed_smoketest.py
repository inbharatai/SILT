"""Real-AI smoke test for the streamed deep-apply backend (SmolLM2-135M, CPU).

Run:  ASEA_RUN_REAL=1 python scripts/real_streamed_smoketest.py

Prints honest, measured figures (trainable params, training loss, parity,
Gate 2 verdict, audit integrity, rollback) for the integration report. A
REJECTED Gate 2 verdict on a 2-step CPU streamed LoRA run is the system
working -- this asserts the MECHANISM, not a promotion. No weights are
published; everything stays local (patent pending, India).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# src on path (no installed package needed)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asea.audit.logger import AuditLog
from asea.benchmarks.harness import BenchmarkCase, BenchmarkHarness, BenchmarkSuite
from asea.core.protocol import (
    Domain, EvaluationScores, LearningLevel, Modality, OriginKind,
    PacketType, PromotionStatus, Provenance, SkillPacket,
)
from asea.deepapply import DeepApplyConfig, DeepApplyRunner
from asea.deepapply.store import APPROVED
from asea.memory.store import MemoryStore

MODEL = "HuggingFaceTB/SmolLM2-135M"


def _cap(domain, language):
    from asea.core.protocol import CapabilityKey
    return CapabilityKey(task_type="translate", modality=Modality.TEXT,
                         domain=domain, language=language)


def _case(cid, p, e, split):
    return BenchmarkCase(case_id=cid, prompt=p, expected=e, split=split)


def _suite(sid, domain, language, cases):
    return BenchmarkSuite(suite_id=sid, task_type="translate", modality=Modality.TEXT,
                          domain=domain, language=language, cases=cases)


def _scores():
    return EvaluationScores(schema_compliance=1.0, semantic_similarity=0.9,
                            task_success=0.9, language_preservation=1.0,
                            hallucination_risk=0.05, aggregate=0.9,
                            baseline_score=0.30, candidate_score=0.80,
                            regression_detected=False, case_count=3,
                            case_regression_count=0)


def main():
    if not os.environ.get("ASEA_RUN_REAL"):
        print("set ASEA_RUN_REAL=1 to run (downloads SmolLM2-135M, trains on CPU)")
        sys.exit(2)
    import torch, transformers, peft  # noqa: F401
    from asea.modules.real import HFCausalConnector, translation_capability

    tmp = Path(tempfile.mkdtemp(prefix="silt_real_"))
    mem = MemoryStore(tmp / "memory")
    audit = AuditLog(tmp / "audit" / "audit.jsonl")
    harness = BenchmarkHarness()
    runner = DeepApplyRunner(mem, tmp / "adapters", audit, harness)

    entries = {"entries": [
        {"source": "ভাত", "target": "rice"},
        {"source": "পানী", "target": "water"},
        {"source": "ঘৰ", "target": "home"},
    ]}
    pkt = SkillPacket(
        packet_id="real-p1", task_type="translate", source_module="sender-real",
        target_module="smollm2-receiver",
        sender_capability=_cap(Domain.TRANSLATION, "as->en"),
        modality=Modality.TEXT, language="as->en", domain=Domain.TRANSLATION,
        packet_type=PacketType.GLOSSARY, distilled_skill=entries,
        confidence_score=0.9, evaluator_score=0.9, safety_score=1.0,
        scores=_scores(),
        provenance=Provenance(origin_kind=OriginKind.CURATED_CORPUS, chain=["sender-real"],
                              synthetic_depth=0, is_mock=False),
        learning_level=LearningLevel.L3_SKILL_PACKET,
        promotion_status=PromotionStatus.PROMOTED, rollback_token="snap-real",
    )
    mem.approve(pkt)

    recv = HFCausalConnector(
        model_id=MODEL,
        capabilities=[translation_capability("as", "en"), translation_capability("hi", "en")],
        module_id="smollm2-receiver", max_new_tokens=24,
        max_learning_level=LearningLevel.L4_PEFT_CANDIDATE,
    )
    target = _suite("as_en_real", Domain.TRANSLATION, "as->en", [
        _case("rh1", "ভাত", "rice", "heldout"),
        _case("rh2", "পানী", "water", "heldout"),
    ])
    regression = [_suite("hi_en_real", Domain.GENERAL, "hi->en", [
        _case("rr1", "নদী", "river", "regression"),
    ])]
    cfg = DeepApplyConfig(backend="streamed", max_steps=2, max_steps_cap=8,
                          lora_rank=4, lora_alpha=8, seed=0)
    report = runner.run(recv, ["real-p1"], cfg, target, regression_suites=regression)
    ad = report.adapter

    print("=" * 64)
    print("SILT streamed backend -- REAL-AI smoke test (SmolLM2-135M, CPU)")
    print("=" * 64)
    print(f"backend             : {ad.backend} {ad.backend_version}")
    print(f"storage_tier        : {ad.storage_tier}")
    print(f"config_fingerprint  : {ad.config_fingerprint}")
    print(f"parity_verified     : {ad.parity_verified}")
    print(f"parity_report_hash  : {ad.parity_report_hash[:16]}...")
    print(f"trainable_params    : {ad.trainable_param_count}")
    print(f"training_loss       : {ad.training_loss:.6f}  (finite={bool(torch.isfinite(torch.tensor(ad.training_loss)))})")
    print(f"adapter_artifact_ref: {ad.adapter_artifact_ref}")
    print(f"is_mock             : {ad.provenance.is_mock}")
    print(f"Gate 2 verdict      : {report.decision.status.value}")
    print(f"needs_human         : {report.decision.needs_human}")
    print(f"approved count      : {runner.store.count(APPROVED)}")
    events = [e["event"] for e in audit.entries()]
    print(f"audit events        : {len(events)}  (stream_backend_selected={'yes' if 'stream_backend_selected' in events else 'no'}, parity_check={'yes' if 'parity_check' in events else 'no'})")
    print(f"audit chain ok      : {audit.verify()['ok']}")
    if report.decision.status == PromotionStatus.PROMOTED:
        runner.rollback_adapter(ad.adapter_id, report.rollback_token)
        print(f"rollback            : restored prior approved state (token={report.rollback_token}), approved now {runner.store.count(APPROVED)}")
    print("=" * 64)
    print("A REJECTED/PENDING verdict on a 2-step CPU streamed LoRA run is the")
    print("system working -- this proves the MECHANISM, not a promotion.")
    print("=" * 64)


if __name__ == "__main__":
    main()