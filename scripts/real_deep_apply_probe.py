"""One-off probe: run the real deep-apply on SmolLM2-135M and print the report.

Not a test. Run with: ASEA_RUN_REAL=1 python scripts/real_deep_apply_probe.py
Prints the honest numbers for the final report. Patent pending (India).
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from asea.benchmarks.harness import BenchmarkCase, BenchmarkSuite, BenchmarkHarness
from asea.core.protocol import (
    Domain, EvaluationScores, LearningLevel, Modality, OriginKind,
    PacketType, PromotionStatus, Provenance, SkillPacket,
)
from asea.deepapply import DeepApplyConfig, DeepApplyRunner
from asea.memory.store import MemoryStore
from asea.audit.logger import AuditLog
from asea.modules.real import HFCausalConnector, translation_capability

MODEL = "HuggingFaceTB/SmolLM2-135M"


def _cap(domain, lang):
    from asea.core.protocol import CapabilityKey
    return CapabilityKey(task_type="translate", modality=Modality.TEXT,
                         domain=domain, language=lang)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="deepapply_real_"))
    mem = MemoryStore(tmp / "memory")
    audit = AuditLog(tmp / "audit" / "audit.jsonl")
    harness = BenchmarkHarness()
    runner = DeepApplyRunner(mem, tmp / "adapters", audit, harness)

    cap = _cap(Domain.TRANSLATION, "as->en")
    pkt = SkillPacket(
        packet_id="real-p1", task_type="translate", source_module="sender-real",
        target_module="smollm2-receiver", sender_capability=cap, modality=Modality.TEXT,
        language="as->en", domain=Domain.TRANSLATION, packet_type=PacketType.GLOSSARY,
        distilled_skill={"entries": [
            {"source": "ভাত", "target": "rice"},
            {"source": "পানী", "target": "water"},
            {"source": "ঘৰ", "target": "home"},
            {"source": "মাছ", "target": "fish"},
        ]},
        confidence_score=0.9, evaluator_score=0.9, safety_score=1.0,
        scores=EvaluationScores(
            schema_compliance=1.0, semantic_similarity=0.9, task_success=0.9,
            language_preservation=1.0, hallucination_risk=0.05, aggregate=0.9,
            baseline_score=0.3, candidate_score=0.8, regression_detected=False,
            case_count=2, case_regression_count=0,
        ),
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
    target = BenchmarkSuite(
        suite_id="as_en_real", task_type="translate", modality=Modality.TEXT,
        domain=Domain.TRANSLATION, language="as->en",
        cases=[
            BenchmarkCase(case_id="rh1", prompt="ভাত", expected="rice", split="heldout"),
            BenchmarkCase(case_id="rh2", prompt="পানী", expected="water", split="heldout"),
        ],
    )
    control = BenchmarkSuite(
        suite_id="hi_en_real", task_type="translate", modality=Modality.TEXT,
        domain=Domain.GENERAL, language="hi->en",
        cases=[BenchmarkCase(case_id="rr1", prompt="नदी", expected="river", split="regression")],
    )

    cfg = DeepApplyConfig(backend="standard", max_steps=3, max_steps_cap=8,
                          lora_rank=4, lora_alpha=8, seed=0)
    report = runner.run(recv, ["real-p1"], cfg, target, regression_suites=[control])
    d = report.to_dict()
    print("=== REAL DEEP-APPLY REPORT (SmolLM2-135M, 3 LoRA steps, CPU) ===")
    print(json.dumps(d, indent=2, default=str))
    print("=== audit verify ===")
    print(json.dumps(audit.verify(), indent=2))
    print("=== store stats ===")
    print(json.dumps(runner.store.stats(), indent=2))


if __name__ == "__main__":
    if not os.environ.get("ASEA_RUN_REAL"):
        raise SystemExit("set ASEA_RUN_REAL=1 to run the real LoRA train")
    main()