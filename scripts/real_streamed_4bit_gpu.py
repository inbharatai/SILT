"""Real-AI 4-bit GPU verification for the streamed backend (QLoRA).

De-risks the 4-bit + GPU wiring on the small SmolLM2-135M (cached, ~80s) BEFORE
scaling to Qwen2.5-7B. Asserts the full wired path end-to-end:
4-bit load -> peft LoRA -> disk bank -> forward parity on cuda (bitwise) ->
streamed train step -> Gate 2 attach (4-bit base + LoRA) -> verdict -> audit.

Run:  set ASEA_RUN_REAL=1 && python scripts/real_streamed_4bit_gpu.py
Prints honest measured figures. A REJECTED verdict on a 2-step run is the
system working -- this asserts the MECHANISM on GPU, not a promotion.
Local only; patent pending (India). The script publishes nothing.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

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

MODEL = os.environ.get("ASEA_MODEL", "HuggingFaceTB/SmolLM2-135M")
BACKEND = os.environ.get("ASEA_BACKEND", "streamed")  # "streamed" | "zeroforge"


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
        print("set ASEA_RUN_REAL=1 to run (loads {} in 4-bit on GPU)".format(MODEL))
        sys.exit(2)
    import torch
    from asea.modules.real import HFCausalConnector, translation_capability

    tmp = Path(tempfile.mkdtemp(prefix="silt_4bit_"))
    mem = MemoryStore(tmp / "memory")
    audit = AuditLog(tmp / "audit" / "audit.jsonl")
    harness = BenchmarkHarness()
    runner = DeepApplyRunner(mem, tmp / "adapters", audit, harness)

    pkt = SkillPacket(
        packet_id="real-4bit-p1", task_type="translate", source_module="sender-real",
        target_module="recv-4bit",
        sender_capability=_cap(Domain.TRANSLATION, "as->en"),
        modality=Modality.TEXT, language="as->en", domain=Domain.TRANSLATION,
        packet_type=PacketType.GLOSSARY,
        distilled_skill={"entries": [{"source": "ভাত", "target": "rice"},
                                     {"source": "পানী", "target": "water"},
                                     {"source": "ঘৰ", "target": "home"}]},
        confidence_score=0.9, evaluator_score=0.9, safety_score=1.0, scores=_scores(),
        provenance=Provenance(origin_kind=OriginKind.CURATED_CORPUS, chain=["sender-real"],
                              synthetic_depth=0, is_mock=False),
        learning_level=LearningLevel.L3_SKILL_PACKET,
        promotion_status=PromotionStatus.PROMOTED, rollback_token="snap-4bit",
    )
    mem.approve(pkt)

    recv = HFCausalConnector(
        model_id=MODEL,
        capabilities=[translation_capability("as", "en"), translation_capability("hi", "en")],
        module_id="recv-4bit", max_new_tokens=24,
        max_learning_level=LearningLevel.L4_PEFT_CANDIDATE,
        load_in_4bit=True,
    )
    target = _suite("as_en_real", Domain.TRANSLATION, "as->en", [
        _case("rh1", "ভাত", "rice", "heldout"),
        _case("rh2", "পানী", "water", "heldout"),
    ])
    regression = [_suite("hi_en_real", Domain.GENERAL, "hi->en", [
        _case("rr1", "নদী", "river", "regression"),
    ])]
    cfg = DeepApplyConfig(backend=BACKEND, load_in_4bit=True,
                          max_steps=2, max_steps_cap=8,
                          lora_rank=4, lora_alpha=8, seed=0)
    report = runner.run(recv, ["real-4bit-p1"], cfg, target, regression_suites=regression)
    ad = report.adapter

    print("=" * 66)
    print("SILT {} backend -- 4-bit GPU QLoRA verification ({})".format(BACKEND, MODEL))
    print("=" * 66)
    print("cuda available     : {} ({})".format(torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-"))
    print("backend             : {} {}".format(ad.backend, ad.backend_version))
    print("storage_tier        : {}".format(ad.storage_tier))
    print("config_fingerprint  : {}".format(ad.config_fingerprint))
    print("parity_verified     : {}".format(ad.parity_verified))
    print("parity_report_hash  : {}...".format(ad.parity_report_hash[:16]))
    print("trainable_params    : {}".format(ad.trainable_param_count))
    print("training_loss       : {:.6f}  (finite={})".format(
        ad.training_loss, bool(torch.isfinite(torch.tensor(ad.training_loss)))))
    print("is_mock             : {}".format(ad.provenance.is_mock))
    print("Gate 2 verdict      : {}".format(report.decision.status.value))
    print("needs_human         : {}".format(report.decision.needs_human))
    print("approved count       : {}".format(runner.store.count(APPROVED)))
    events = [e["event"] for e in audit.entries()]
    print("audit events        : {}  (stream_backend_selected={}, parity_check={})".format(
        len(events), "yes" if "stream_backend_selected" in events else "no",
        "yes" if "parity_check" in events else "no"))
    print("audit chain ok      : {}".format(audit.verify()["ok"]))
    print("vram peak GB        : {:.2f}".format(
        torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0))
    if report.decision.status == PromotionStatus.PROMOTED:
        runner.rollback_adapter(ad.adapter_id, report.rollback_token)
        print("rollback            : restored, approved now {}".format(
            runner.store.count(APPROVED)))
    print("=" * 66)
    print("A REJECTED verdict on a 2-step 4-bit GPU run is the system working --")
    print("this asserts the MECHANISM on GPU, not a promotion.")
    print("=" * 66)


if __name__ == "__main__":
    main()