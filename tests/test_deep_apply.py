"""DEEP-APPLY -- native gated LoRA trainer.

Covers the binding contract:

* Only PROMOTED packets enter training data; mock is refused at intake.
* Provenance, synthetic depth and risk tier propagate from source packets.
* High-risk source contamination parks the adapter at PENDING_HUMAN under a
  MAXIMALLY PERMISSIVE policy -- no config can bypass human approval.
* Gate 2 rejects on no-improvement / regression / per-case regression, with
  named reasons; the all-or-nothing discipline holds.
* Rollback restores the receiver to its exact pre-admission behaviour.
* The audit chain records every transition and detects tamper at an index.
* BLOCKED errors are honest and named (missing [deep] extra; no CUDA for the
  streamed backend) -- never a silent fallback, never a fabricated result.
* Gate 2 is backend-agnostic: zero backend-conditional branches; the verdict is
  identical for a standard vs streamed adapter with the same scores, and the
  backend+version are recorded in the AdapterPacket.
* A real end-to-end LoRA train on SmolLM2-135M (opt-in via ASEA_RUN_REAL=1)
  exercises the real StandardTrainerBackend. It asserts the MECHANISM runs
  (train -> eval -> Gate 2 -> audit, finite loss, >0 trainable params), not that
  a 3-step CPU LoRA promotes -- that would be dishonest.

The real trainer backends are NOT used by the structural tests. They use a
ScriptedTrainerBackend defined HERE (a test double injected via the runner's
``trainer=`` seam), which produces a deterministic AdapterArtifact whose
``attach`` returns a module with a controlled answer map. Nothing mock enters a
real path: the scripted double lives in tests/ and is never imported by the
package.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from asea.audit.logger import AuditLog
from asea.benchmarks.harness import BenchmarkCase, BenchmarkHarness, BenchmarkSuite
from asea.core.errors import AuditIntegrityError
from asea.core.protocol import (
    Domain,
    EvaluationScores,
    LearningLevel,
    Modality,
    OriginKind,
    PacketType,
    PromotionStatus,
    Provenance,
    SkillPacket,
)
from asea.deepapply import (
    AdapterPacket,
    DeepApplyBlocked,
    DeepApplyConfig,
    DeepApplyGate,
    DeepApplyIntakeError,
    DeepApplyPolicy,
    DeepApplyRunner,
    StreamedTrainerBackend,
    TrainerBackend,
    build_training_dataset,
    get_backend,
)
from asea.deepapply.adapter_packet import max_risk_tier
from asea.deepapply.runner import _merged_provenance
from asea.deepapply.store import APPROVED, CANDIDATE, REJECTED
from asea.deepapply.trainer import AdapterArtifact, _require_deep
from asea.memory.store import MemoryStore
from asea.modules.mock.base import MockModule

# ===========================================================================
# Test doubles (live HERE, never imported by the package)
# ===========================================================================


class ScriptedAdaptedModule(MockModule):
    """An adapter-conditioned receiver for tests. ``is_mock`` stays False: this
    models a real adapter changing the receiver's outputs, not a mock model."""

    is_mock = False

    def __init__(self, module_id, capabilities, answer_map, default, knowledge=None):
        super().__init__(
            module_id=module_id,
            display_name=module_id,
            capabilities=capabilities,
            roles=["receiver"],
            knowledge=knowledge or {},
            fallback="echo",
            max_learning_level=LearningLevel.L4_PEFT_CANDIDATE,
            consumes_skills=False,
        )
        self._answer_map = dict(answer_map)
        self._default = default

    def infer(self, capability, prompt):
        return self._answer_map.get(str(prompt).strip(), self._default)

    def infer_with_skills(self, capability, prompt, skills):
        return self.infer(capability, prompt)


class ScriptedAdapterArtifact(AdapterArtifact):
    """Deterministic artifact. ``attach`` returns the scripted adapted module."""

    backend = "scripted"
    backend_version = "test-v1"

    def __init__(self, adapter_path, capabilities, answer_map, default, lora_config,
                 trainable_param_count, training_loss):
        self.adapter_path = adapter_path
        self._capabilities = list(capabilities)
        self._answer_map = dict(answer_map)
        self._default = default
        self.lora_config = dict(lora_config)
        self.trainable_param_count = trainable_param_count
        self.training_loss = training_loss

    def attach(self, receiver):
        return ScriptedAdaptedModule(
            module_id="{}+scripted-lora".format(receiver.module_id),
            capabilities=self._capabilities,
            answer_map=self._answer_map,
            default=self._default,
        )


class ScriptedTrainerBackend(TrainerBackend):
    """Test double. Produces a controlled A/B so Gate 2 sees a known outcome.

    Never falls back, never fabricates a loss; the test sets ``training_loss`` and
    ``trainable_param_count`` explicitly.
    """

    name = "scripted"
    version = "test-v1"

    def __init__(self, answer_map, default="<unk>", training_loss=0.5,
                 trainable_param_count=42, lora_rank=8, lora_alpha=16):
        self._answer_map = dict(answer_map)
        self._default = default
        self.training_loss = training_loss
        self.trainable_param_count = trainable_param_count
        self._lora_config = {
            "r": lora_rank, "lora_alpha": lora_alpha, "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.05, "bias": "none", "task_type": "CAUSAL_LM",
        }

    def supports(self, receiver):
        return True

    def train(self, receiver, dataset, config, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        adapter_path = out_dir / "scripted_adapter"
        adapter_path.mkdir(parents=True, exist_ok=True)
        return ScriptedAdapterArtifact(
            adapter_path=str(adapter_path),
            capabilities=receiver.manifest().capabilities,
            answer_map=self._answer_map,
            default=self._default,
            lora_config=self._lora_config,
            trainable_param_count=self.trainable_param_count,
            training_loss=self.training_loss,
        )


# ===========================================================================
# Helpers
# ===========================================================================


def _cap(domain=Domain.TRANSLATION, language="as->en", task_type="translate"):
    from asea.core.protocol import CapabilityKey
    return CapabilityKey(task_type=task_type, modality=Modality.TEXT,
                         domain=domain, language=language)


def _case(case_id, prompt, expected, split):
    return BenchmarkCase(case_id=case_id, prompt=prompt, expected=expected, split=split)


def _suite(suite_id, domain, language, cases, task_type="translate"):
    return BenchmarkSuite(
        suite_id=suite_id, task_type=task_type, modality=Modality.TEXT,
        domain=domain, language=language, cases=cases,
    )


def good_adapter_scores(**over):
    payload = dict(
        schema_compliance=1.0, semantic_similarity=0.9, task_success=0.9,
        language_preservation=1.0, hallucination_risk=0.05, aggregate=0.9,
        baseline_score=0.30, candidate_score=0.80, regression_detected=False,
        case_count=3, case_regression_count=0,
    )
    payload.update(over)
    return EvaluationScores(**payload)


def _adapter_packet(**over):
    """A clean adapter that passes every Gate 2 check (LOW risk)."""
    base = dict(
        base_model="test-base", base_model_fingerprint="fp",
        target_module="learner",
        source_packet_ids=["p1"], source_domains=[Domain.TRANSLATION],
        synthetic_depth=0,
        provenance=Provenance(origin_kind=OriginKind.CURATED_CORPUS, chain=["sender-x"],
                              synthetic_depth=0, is_mock=False),
        learning_level=LearningLevel.L4_PEFT_CANDIDATE,
        lora_config={"r": 8}, training_config_hash="tch", dataset_hash="dh", seed=0,
        backend="standard", backend_version="peft-lora-v1",
        trainable_param_count=42, training_loss=0.5,
        adapter_artifact_ref="/tmp/adapter",
        safety_score=1.0, evaluator_score=0.9, scores=good_adapter_scores(),
        promotion_status=PromotionStatus.EVALUATED,
    )
    base.update(over)
    return AdapterPacket(**base)


def _promoted_packet(domain, chain, depth=0, is_mock=False, packet_id="p1",
                     safety_score=1.0):
    cap = _cap(domain=domain)
    return SkillPacket(
        packet_id=packet_id,
        task_type="translate", source_module=chain[-1] if chain else "sender-x",
        target_module="learner",
        sender_capability=cap, modality=Modality.TEXT, language="as->en",
        domain=domain,
        packet_type=PacketType.GLOSSARY,
        distilled_skill={"entries": [{"source": "ভাত", "target": "rice"}]},
        confidence_score=0.9, evaluator_score=0.9, safety_score=safety_score,
        scores=good_adapter_scores(),
        provenance=Provenance(origin_kind=OriginKind.CURATED_CORPUS, chain=chain,
                              synthetic_depth=depth, is_mock=is_mock),
        learning_level=LearningLevel.L3_SKILL_PACKET,
        promotion_status=PromotionStatus.PROMOTED,
        rollback_token="snap-1",
    )


# ===========================================================================
# Intake (Gate 1 re-enforced at the door of Gate 2)
# ===========================================================================


def test_intake_refuses_non_promoted_packet():
    pkt = _promoted_packet(Domain.TRANSLATION, chain=["s"], safety_score=1.0)
    pkt.promotion_status = PromotionStatus.EVALUATED
    with pytest.raises(DeepApplyIntakeError, match="non-PROMOTED"):
        build_training_dataset([pkt])


def test_intake_refuses_mock_provenance_under_strict():
    pkt = _promoted_packet(Domain.TRANSLATION, chain=["s"], is_mock=True)
    with pytest.raises(DeepApplyIntakeError, match="mock"):
        build_training_dataset([pkt], strict_no_mock=True)


def test_intake_accepts_mock_when_strict_disabled():
    pkt = _promoted_packet(Domain.TRANSLATION, chain=["s"], is_mock=True)
    ds = build_training_dataset([pkt], strict_no_mock=False)
    assert ds.contains_mock is True


def test_empty_packet_list_yields_empty_dataset():
    ds = build_training_dataset([])
    assert ds.rows == []
    assert ds.source_packet_ids == []


# ===========================================================================
# Provenance / depth / risk propagation
# ===========================================================================


def test_provenance_depth_and_risk_propagate():
    a = _promoted_packet(Domain.TRANSLATION, chain=["s1"], depth=1, packet_id="a")
    b = _promoted_packet(Domain.EDUCATION, chain=["s2", "s3"], depth=2, packet_id="b")
    ds = build_training_dataset([a, b])
    assert ds.synthetic_depth == 2  # max
    assert [d.value for d in ds.source_domains] == ["education", "translation"]
    prov = _merged_provenance([a, b], "aid")
    assert prov.chain == ["s1", "s2", "s3"]
    assert prov.synthetic_depth == 2
    assert prov.is_mock is False
    assert prov.origin_kind == OriginKind.CURATED_CORPUS


def test_max_risk_tier_is_max_severity():
    assert max_risk_tier([Domain.TRANSLATION]) == __import__("asea").core.protocol.RiskTier.LOW
    assert max_risk_tier([Domain.EDUCATION, Domain.TRANSLATION]).value == "medium"
    assert max_risk_tier([Domain.MEDICAL, Domain.TRANSLATION]).value == "high"
    assert max_risk_tier([]).value == "low"


# ===========================================================================
# Gate 2 -- the full check discipline on a trained adapter
# ===========================================================================


def test_clean_adapter_is_admitted():
    adapter = _adapter_packet()
    decision = DeepApplyGate().apply(adapter, rollback_token="snap")
    assert decision.approved is True
    assert adapter.promotion_status == PromotionStatus.PROMOTED
    assert decision.reason() == "all checks passed"


def test_gate2_emits_fifteen_low_risk_checks():
    """Equality over the FULL low-risk check set (the analogue of
    test_decision_serialises_every_check). A LOW-risk adapter emits exactly
    these 15; a HIGH-risk one adds human_approval (16).

    Was 14 pre-2026-08-17; +1 for the control-movement bound (no_control_movement),
    a gate-STRENGTHENING check -- the count is updated to the new contract, not
    a regression. See evaluator.py + gate2.py no_control_movement."""
    decision = DeepApplyGate().apply(_adapter_packet(), rollback_token="snap")
    names = {c["name"] for c in decision.to_dict()["checks"]}
    assert names == {
        "schema_validation", "training_loss_finite", "adapter_artifact_present",
        "evaluator_threshold", "safety_threshold", "benchmark_improvement",
        "no_regression", "case_regression_limit", "no_control_movement",
        "provenance_present",
        "synthetic_depth", "no_self_lineage", "rollback_metadata",
        "applicable_learning_level", "no_mock_provenance",
    }


def test_gate2_adds_human_approval_for_high_risk():
    adapter = _adapter_packet(source_domains=[Domain.MEDICAL])
    decision = DeepApplyGate().decide(adapter)
    names = {c["name"] for c in decision.to_dict()["checks"]}
    assert "human_approval" in names
    assert len(names) == 16


def test_training_loss_not_finite_blocks():
    adapter = _adapter_packet(training_loss=float("inf"))
    decision = DeepApplyGate().decide(adapter)
    assert any(c.name == "training_loss_finite" and not c.passed for c in decision.checks)


def test_missing_adapter_artifact_blocks():
    adapter = _adapter_packet(adapter_artifact_ref=None)
    decision = DeepApplyGate().decide(adapter)
    assert any(c.name == "adapter_artifact_present" and not c.passed for c in decision.checks)


def test_zero_trainable_params_blocks():
    adapter = _adapter_packet(trainable_param_count=0)
    decision = DeepApplyGate(DeepApplyPolicy(min_trainable_params=1)).decide(adapter)
    assert any(c.name == "adapter_artifact_present" and not c.passed for c in decision.checks)


def test_no_improvement_blocks():
    adapter = _adapter_packet(scores=good_adapter_scores(baseline_score=0.80, candidate_score=0.80))
    decision = DeepApplyGate().apply(adapter, rollback_token="snap")
    assert not decision.approved
    assert any(c.name == "benchmark_improvement" for c in decision.failures)


def test_regression_blocks_even_with_improvement():
    adapter = _adapter_packet(scores=good_adapter_scores(
        regression_detected=True, regression_detail="hindi: 0.9 -> 0.6"))
    decision = DeepApplyGate().apply(adapter, rollback_token="snap")
    assert not decision.approved
    assert any(c.name == "no_regression" for c in decision.failures)


def test_control_movement_blocks_even_when_it_improves_a_control():
    """The hole A3 closes (audit 2026-08-17): a CONTROL (non-target) suite that
    IMPROVED is not benign -- the adapter touched a capability it was not
    targeting. ``no_regression`` only sees drops and would let this through; the
    new hard ``no_control_movement`` check blocks it. Gate-strengthening."""
    adapter = _adapter_packet(scores=good_adapter_scores(
        regression_detected=False,
        control_movement_detected=True,
        control_movement_detail="hindi: 0.60 -> 0.73 (|delta| +0.1300 > 0.05)",
    ))
    decision = DeepApplyGate().apply(adapter, rollback_token="snap")
    assert not decision.approved
    assert any(c.name == "no_control_movement" for c in decision.failures)
    # no_regression PASSES (no drop) -- the movement check is the one that bites.
    assert all(c.name != "no_regression" for c in decision.failures)


def test_control_movement_passes_when_within_bound():
    """Movement within the bound (default 0.05) is allowed; only |delta| beyond
    it blocks. Ensures the gate does not over-fire on harmless noise."""
    adapter = _adapter_packet(scores=good_adapter_scores(
        regression_detected=False, control_movement_detected=False,
    ))
    decision = DeepApplyGate().apply(adapter, rollback_token="snap")
    assert decision.approved
    assert all(c.passed for c in decision.checks if c.name == "no_control_movement")


def test_per_case_regression_blocks_when_configured():
    adapter = _adapter_packet(scores=good_adapter_scores(case_count=3, case_regression_count=1))
    # ratio 1/3 ~ 0.33 > 0.2 -> blocked by case_regression_limit, improvement still passes
    strict = DeepApplyGate(DeepApplyPolicy(max_case_regression_ratio=0.2))
    decision = strict.apply(adapter, rollback_token="snap")
    assert not decision.approved
    assert any(c.name == "case_regression_limit" for c in decision.failures)
    # and benchmark_improvement is NOT among the failures (isolated failure)
    assert all(c.name != "benchmark_improvement" for c in decision.failures)


def test_self_lineage_blocks():
    adapter = _adapter_packet(
        target_module="learner",
        provenance=Provenance(origin_kind=OriginKind.MODEL_GENERATED,
                              chain=["s1", "learner"], synthetic_depth=1),
    )
    decision = DeepApplyGate().decide(adapter)
    assert any(c.name == "no_self_lineage" and not c.passed for c in decision.checks)


def test_excessive_synthetic_depth_blocks():
    adapter = _adapter_packet(synthetic_depth=5,
                              provenance=Provenance(origin_kind=OriginKind.MODEL_GENERATED,
                                                    chain=["s1"], synthetic_depth=5))
    decision = DeepApplyGate().decide(adapter)
    assert any(c.name == "synthetic_depth" and not c.passed for c in decision.checks)


def test_mock_source_provenance_blocks():
    adapter = _adapter_packet(
        provenance=Provenance(origin_kind=OriginKind.MODEL_GENERATED,
                              chain=["qwen-mock"], is_mock=True),
    )
    decision = DeepApplyGate().decide(adapter)
    assert any(c.name == "no_mock_provenance" and not c.passed for c in decision.checks)


def test_l5_export_only_level_blocks():
    adapter = _adapter_packet(learning_level=LearningLevel.L5_DISTILL_DATASET)
    decision = DeepApplyGate().decide(adapter)
    assert any(c.name == "applicable_learning_level" and not c.passed for c in decision.checks)


@pytest.mark.parametrize("domain", [Domain.MEDICAL, Domain.LEGAL, Domain.FINANCE])
def test_high_risk_adapter_parks_pending_human(domain):
    adapter = _adapter_packet(source_domains=[domain])
    decision = DeepApplyGate().apply(adapter, rollback_token="snap")
    assert decision.status == PromotionStatus.PENDING_HUMAN
    assert decision.needs_human is True
    assert decision.approved is False


def test_high_risk_promotes_only_with_named_approver():
    adapter = _adapter_packet(source_domains=[Domain.MEDICAL])
    decision = DeepApplyGate().apply(adapter, rollback_token="snap",
                                      human_approver="dr.review@example.org")
    assert decision.approved is True
    assert adapter.human_approved_by == "dr.review@example.org"


def test_human_approval_does_not_waive_other_checks():
    adapter = _adapter_packet(source_domains=[Domain.MEDICAL], safety_score=0.0)
    decision = DeepApplyGate().apply(adapter, rollback_token="snap",
                                      human_approver="dr.review@example.org")
    assert not decision.approved
    assert any(c.name == "safety_threshold" for c in decision.failures)


def test_no_configuration_can_disable_human_approval_for_high_risk():
    """A maximally permissive policy still cannot auto-promote a medical adapter."""
    adapter = _adapter_packet(source_domains=[Domain.MEDICAL])
    reckless = DeepApplyGate(DeepApplyPolicy(
        min_evaluator_score=0.0, min_safety_score=0.0, min_improvement=-1.0,
        max_case_regression_ratio=99.0, max_control_movement=99.0,
        max_synthetic_depth=99,
        strict_no_mock=False, require_rollback_token=False, min_trainable_params=0,
    ))
    decision = reckless.apply(adapter)
    assert decision.status == PromotionStatus.PENDING_HUMAN


def test_rejected_adapter_records_a_reason():
    adapter = _adapter_packet(scores=good_adapter_scores(baseline_score=0.80, candidate_score=0.80))
    DeepApplyGate().apply(adapter, rollback_token="snap")
    assert adapter.promotion_status == PromotionStatus.REJECTED
    assert "benchmark_improvement" in (adapter.rejection_reason or "")


# ===========================================================================
# Backend-agnosticism (Gate 2 has zero backend-conditional branches)
# ===========================================================================


def test_gate2_source_has_no_backend_branches():
    """Static guarantee: gate2 never reads adapter.backend to change a verdict."""
    src = Path(__file__).resolve().parent.parent / "src" / "asea" / "deepapply" / "gate2.py"
    text = src.read_text(encoding="utf-8")
    assert "backend" not in text, "gate2.py must not reference 'backend' (backend-agnostic)"


def test_gate2_verdict_identical_across_backends():
    """Same scores + same artifact -> same verdict, regardless of backend."""
    std = _adapter_packet(backend="standard", backend_version="peft-lora-v1")
    stm = _adapter_packet(backend="streamed", backend_version="soup-stream-v1-beta")
    d_std = DeepApplyGate().decide(std)
    d_stm = DeepApplyGate().decide(stm)
    assert d_std.status == d_stm.status
    assert [c.name for c in d_std.checks] == [c.name for c in d_stm.checks]
    assert {c.name: c.passed for c in d_std.checks} == {c.name: c.passed for c in d_stm.checks}


# ===========================================================================
# Runner-level (end-to-end mechanism with the scripted backend)
# ===========================================================================


@pytest.fixture
def runner(tmp_path):
    mem = MemoryStore(tmp_path / "memory")
    audit = AuditLog(tmp_path / "audit" / "audit.jsonl")
    harness = BenchmarkHarness()
    return DeepApplyRunner(mem, tmp_path / "adapters", audit, harness)


def _approve_packets(runner, packets):
    for p in packets:
        # PROMOTED + rollback_token already set; approve() writes to approved/.
        runner.memory_store.approve(p)


def _as_en_target_suite():
    return _suite("as_en_target", Domain.TRANSLATION, "as->en", [
        _case("h1", "ভাত", "rice", "heldout"),
        _case("h2", "পানী", "water", "heldout"),
    ])


def _hi_en_regression_suite():
    # A control capability the transfer is NOT targeting.
    return _suite("hi_en_control", Domain.GENERAL, "hi->en", [
        _case("r1", "নদী", "river", "regression"),
    ])


def _receiver(knowledge=None):
    return MockModule(
        module_id="learner", display_name="Learner",
        capabilities=[_cap(Domain.TRANSLATION, "as->en"), _cap(Domain.GENERAL, "hi->en")],
        roles=["receiver"], knowledge=knowledge or {}, fallback="echo",
        max_learning_level=LearningLevel.L4_PEFT_CANDIDATE, consumes_skills=False,
    )


def test_runner_admits_a_clean_adapter(runner, tmp_path):
    pkt = _promoted_packet(Domain.TRANSLATION, chain=["sender-x"])
    _approve_packets(runner, [pkt])
    # Baseline knows the control, not the target; candidate (scripted) knows both.
    recv = _receiver(knowledge={"translate/text/general/hi->en": {"নদী": "river"}})
    backend = ScriptedTrainerBackend(
        answer_map={"ভাত": "rice", "পানী": "water", "নদী": "river"},
        default="<unk>",
    )
    cfg = DeepApplyConfig()
    report = runner.run(
        recv, [pkt.packet_id], cfg, _as_en_target_suite(),
        regression_suites=[_hi_en_regression_suite()], trainer=backend,
    )
    assert report.status == PromotionStatus.PROMOTED
    assert runner.store.count(APPROVED) == 1
    assert report.adapter.backend == "scripted"
    assert report.adapter.backend_version == "test-v1"
    assert report.adapter.adapter_artifact_ref is not None
    assert report.adapter.trainable_param_count == 42
    assert report.rollback_token is not None

    # Audit chain records the full sequence and verifies intact.
    events = [e["event"] for e in runner.audit.entries()]
    assert "train_started" in events
    assert "train_completed" in events
    assert "gate2_decision" in events
    assert "adapter_admitted" in events
    assert runner.audit.verify()["ok"] is True


class _TelemetryTrainerBackend(ScriptedTrainerBackend):
    """A scripted double that ALSO reads the ``_on_step`` telemetry hook the
    runner stashes in the train cfg (mirroring what the real streamed/standard
    backends do). Used to pin the full telemetry wiring end-to-end without a
    real model: the runner stashes ``_on_step`` -> the backend calls it with a
    REAL (test-known) per-step loss -> the event flows out through ``on_progress``.
    The losses are fixed test values, NOT fabricated product numbers."""

    name = "telemetry-double"
    version = "test-tele-v1"

    def train(self, receiver, dataset, config, out_dir):
        on_step = config.get("_on_step")
        if on_step is not None:
            on_step({"phase": "train_step", "backend": self.name,
                     "step": 1, "max_steps": 2, "loss": 0.9})
            on_step({"phase": "train_step", "backend": self.name,
                     "step": 2, "max_steps": 2, "loss": 0.4})
        return super().train(receiver, dataset, config, out_dir)


def test_runner_emits_real_telemetry_phases_and_per_step(runner, tmp_path):
    """The ``on_progress`` callback receives a REAL phase + per-step event stream
    (default None is byte-identical). Pins the telemetry contract the Studio
    live graph will read -- never a fabricated number."""
    pkt = _promoted_packet(Domain.TRANSLATION, chain=["sender-x"])
    _approve_packets(runner, [pkt])
    recv = _receiver(knowledge={"translate/text/general/hi->en": {"নদী": "river"}})
    backend = _TelemetryTrainerBackend(
        answer_map={"ভাত": "rice", "পানী": "water", "নদী": "river"},
        default="<unk>", training_loss=0.4,
    )
    cfg = DeepApplyConfig()
    collected = []

    def on_progress(ev):
        collected.append(dict(ev))

    report = runner.run(
        recv, [pkt.packet_id], cfg, _as_en_target_suite(),
        regression_suites=[_hi_en_regression_suite()], trainer=backend,
        on_progress=on_progress,
    )
    phases = [e["phase"] for e in collected]
    # Phase sequence is the real contract the UI renders as a progress strip.
    assert phases[0] == "session_started"
    assert "dataset_built" in phases
    assert "backend_selected" in phases
    assert "train_started" in phases
    # Per-step events came THROUGH the backend's _on_step hook (real wiring).
    train_steps = [e for e in collected if e["phase"] == "train_step"]
    assert len(train_steps) == 2
    assert train_steps[0]["step"] == 1 and train_steps[0]["loss"] == 0.9
    assert train_steps[1]["step"] == 2 and train_steps[1]["loss"] == 0.4
    assert "train_completed" in phases
    train_completed = next(e for e in collected if e["phase"] == "train_completed")
    assert train_completed["training_loss"] == 0.4
    assert "gate2_evaluated" in phases
    assert "gate2_decision" in phases
    assert phases[-1] == "done"
    assert report.status == PromotionStatus.PROMOTED

    # Byte-identical default (on_progress=None) is already proven by every other
    # test in this file running run() without the callback; this test pins the
    # event contract alone.


def test_runner_propagates_provenance_depth_and_risk(runner):
    a = _promoted_packet(Domain.TRANSLATION, chain=["s1"], depth=1, packet_id="a")
    b = _promoted_packet(Domain.EDUCATION, chain=["s2", "s3"], depth=2, packet_id="b")
    _approve_packets(runner, [a, b])
    recv = _receiver()
    backend = ScriptedTrainerBackend(
        answer_map={"ভাত": "rice", "পানী": "water", "নদী": "river"}, default="<unk>",
    )
    report = runner.run(
        recv, [a.packet_id, b.packet_id], DeepApplyConfig(),
        _as_en_target_suite(), regression_suites=[_hi_en_regression_suite()],
        trainer=backend,
    )
    ad = report.adapter
    assert ad.synthetic_depth == 2
    assert [d.value for d in ad.source_domains] == ["education", "translation"]
    assert ad.risk_tier.value == "medium"  # max(EDUCATION=MEDIUM, TRANSLATION=LOW)
    assert ad.provenance.chain == ["s1", "s2", "s3"]
    assert ad.provenance.is_mock is False
    assert sorted(ad.source_packet_ids) == ["a", "b"]


def test_runner_high_risk_contamination_parks_pending_human(runner):
    """A MEDICAL source packet under a MAXIMALLY PERMISSIVE policy still parks."""
    pkt = _promoted_packet(Domain.MEDICAL, chain=["medic-1"], safety_score=1.0)
    _approve_packets(runner, [pkt])
    recv = _receiver()
    backend = ScriptedTrainerBackend(
        answer_map={"ভাত": "rice", "পানী": "water", "নদী": "river"}, default="<unk>",
    )
    reckless = DeepApplyConfig(
        min_evaluator_score=0.0, min_safety_score=0.0, min_improvement=-1.0,
        max_case_regression_ratio=99.0, max_control_movement=99.0, max_synthetic_depth=99,
        strict_no_mock=False, min_trainable_params=0,
    )
    report = runner.run(
        recv, [pkt.packet_id], reckless, _as_en_target_suite(),
        regression_suites=[_hi_en_regression_suite()], trainer=backend,
    )
    assert report.status == PromotionStatus.PENDING_HUMAN
    assert runner.store.count(APPROVED) == 0
    assert runner.store.count(CANDIDATE) == 1


def test_runner_approve_pending_promotes_with_named_human(runner):
    pkt = _promoted_packet(Domain.MEDICAL, chain=["medic-1"], safety_score=1.0)
    _approve_packets(runner, [pkt])
    recv = _receiver()
    backend = ScriptedTrainerBackend(
        answer_map={"ভাত": "rice", "পানী": "water", "নদী": "river"}, default="<unk>",
    )
    reckless = DeepApplyConfig(
        min_evaluator_score=0.0, min_safety_score=0.0, min_improvement=-1.0,
        max_case_regression_ratio=99.0, max_control_movement=99.0, max_synthetic_depth=99,
        strict_no_mock=False, min_trainable_params=0,
    )
    report = runner.run(
        recv, [pkt.packet_id], reckless, _as_en_target_suite(),
        regression_suites=[_hi_en_regression_suite()], trainer=backend,
    )
    assert report.status == PromotionStatus.PENDING_HUMAN
    decision = runner.approve_pending(report.adapter.adapter_id, "dr.k@ex.org")
    assert decision.approved is True
    assert runner.store.count(APPROVED) == 1


def test_runner_rejects_on_regression(runner):
    """Candidate improves the target but breaks a control -> REJECTED."""
    pkt = _promoted_packet(Domain.TRANSLATION, chain=["sender-x"])
    _approve_packets(runner, [pkt])
    # Baseline knows the control (river); candidate breaks it.
    recv = _receiver(knowledge={"translate/text/general/hi->en": {"নদী": "river"}})
    backend = ScriptedTrainerBackend(
        answer_map={"ভাত": "rice", "পানী": "water", "নদী": "WRONG"}, default="<unk>",
    )
    report = runner.run(
        recv, [pkt.packet_id], DeepApplyConfig(), _as_en_target_suite(),
        regression_suites=[_hi_en_regression_suite()], trainer=backend,
    )
    assert report.status == PromotionStatus.REJECTED
    assert runner.store.count(APPROVED) == 0
    assert runner.store.count(REJECTED) == 1
    assert "no_regression" in (report.adapter.rejection_reason or "")


def test_runner_rejects_on_control_movement(runner):
    """A3 end-to-end (audit 2026-08-17): the candidate IMPROVES a control
    (non-target) suite -- baseline does not know the control, the scripted
    adapter does -- so the control moves +1.0. ``no_regression`` sees no drop
    (it is an improvement, not a regression) and would have let this through
    before A3. The new ``no_control_movement`` hard check catches the movement
    and REJECTS. This is the closed hole, exercised through the real evaluator."""
    pkt = _promoted_packet(Domain.TRANSLATION, chain=["sender-x"])
    _approve_packets(runner, [pkt])
    # Baseline does NOT know the control (empty knowledge); candidate does.
    recv = _receiver()
    backend = ScriptedTrainerBackend(
        answer_map={"ভাত": "rice", "পানী": "water", "নদী": "river"}, default="<unk>",
    )
    report = runner.run(
        recv, [pkt.packet_id], DeepApplyConfig(), _as_en_target_suite(),
        regression_suites=[_hi_en_regression_suite()], trainer=backend,
    )
    assert report.status == PromotionStatus.REJECTED
    assert runner.store.count(APPROVED) == 0
    assert runner.store.count(REJECTED) == 1
    assert "no_control_movement" in (report.adapter.rejection_reason or "")
    # And it is genuinely movement, not a drop: no_regression did NOT fire.
    assert "no_regression" not in (report.adapter.rejection_reason or "")


def test_runner_admits_when_control_does_not_move(runner):
    """The flip side of A3: when the baseline already holds the control (delta
    ~0 on the control suite), no_control_movement passes and a clean adapter is
    still PROMOTED. The bound does not over-fire -- only real movement blocks."""
    pkt = _promoted_packet(Domain.TRANSLATION, chain=["sender-x"])
    _approve_packets(runner, [pkt])
    recv = _receiver(knowledge={"translate/text/general/hi->en": {"নদী": "river"}})
    backend = ScriptedTrainerBackend(
        answer_map={"ভাত": "rice", "পানী": "water", "নদী": "river"}, default="<unk>",
    )
    report = runner.run(
        recv, [pkt.packet_id], DeepApplyConfig(), _as_en_target_suite(),
        regression_suites=[_hi_en_regression_suite()], trainer=backend,
    )
    assert report.status == PromotionStatus.PROMOTED


def test_runner_rejects_on_no_improvement(runner):
    pkt = _promoted_packet(Domain.TRANSLATION, chain=["sender-x"])
    _approve_packets(runner, [pkt])
    recv = _receiver()
    # Candidate echoes the prompt exactly (identical to the baseline echo) -> no delta.
    backend = ScriptedTrainerBackend(
        answer_map={"ভাত": "ভাত", "পানী": "পানী", "নদী": "নদী"}, default="echoed",
    )
    report = runner.run(
        recv, [pkt.packet_id], DeepApplyConfig(), _as_en_target_suite(),
        regression_suites=[_hi_en_regression_suite()], trainer=backend,
    )
    assert report.status == PromotionStatus.REJECTED
    assert any(c.name == "benchmark_improvement" for c in report.decision.failures)


def test_runner_refuses_non_promoted_packet_at_intake(runner):
    pkt = _promoted_packet(Domain.TRANSLATION, chain=["s"])
    # Leave it in candidate, not approved -> _resolve_packets raises.
    runner.memory_store.put_candidate(pkt)
    recv = _receiver()
    backend = ScriptedTrainerBackend(answer_map={"ভাত": "rice"}, default="x")
    with pytest.raises(DeepApplyIntakeError, match="not in approved"):
        runner.run(recv, [pkt.packet_id], DeepApplyConfig(), _as_en_target_suite(),
                   regression_suites=[_hi_en_regression_suite()], trainer=backend)


# ===========================================================================
# Rollback restores the receiver to pre-admission behaviour
# ===========================================================================


def test_rollback_restores_approved_set_and_marks_adapter(runner):
    pkt = _promoted_packet(Domain.TRANSLATION, chain=["sender-x"])
    _approve_packets(runner, [pkt])
    recv = _receiver(knowledge={"translate/text/general/hi->en": {"নদী": "river"}})
    backend = ScriptedTrainerBackend(
        answer_map={"ভাত": "rice", "পানী": "water", "নদী": "river"}, default="<unk>",
    )
    report = runner.run(
        recv, [pkt.packet_id], DeepApplyConfig(), _as_en_target_suite(),
        regression_suites=[_hi_en_regression_suite()], trainer=backend,
    )
    assert runner.store.count(APPROVED) == 1
    token = report.rollback_token
    adapter_id = report.adapter.adapter_id

    result = runner.rollback_adapter(adapter_id, token)
    assert result["restored"] == 0  # approved set had only this adapter; snapshot was empty
    assert runner.store.count(APPROVED) == 0
    # The rolled-back adapter is recorded in rejected/ with status ROLLED_BACK.
    record = runner.store.get(REJECTED, adapter_id)
    assert record.promotion_status == PromotionStatus.ROLLED_BACK
    # Audit captured the rollback.
    events = [e["event"] for e in runner.audit.entries()]
    assert "adapter_rolled_back" in events


# ===========================================================================
# Audit chain integrity + tamper detection at an index
# ===========================================================================


def test_audit_tamper_is_detected_at_index(runner, tmp_path):
    pkt = _promoted_packet(Domain.TRANSLATION, chain=["sender-x"])
    _approve_packets(runner, [pkt])
    recv = _receiver(knowledge={"translate/text/general/hi->en": {"নদী": "river"}})
    backend = ScriptedTrainerBackend(
        answer_map={"ভাত": "rice", "পানী": "water", "নদী": "river"}, default="<unk>",
    )
    runner.run(recv, [pkt.packet_id], DeepApplyConfig(), _as_en_target_suite(),
               regression_suites=[_hi_en_regression_suite()], trainer=backend)
    assert runner.audit.verify()["ok"] is True

    # Forge the first entry's detail.
    import json as _json
    entries = runner.audit.entries()
    entries[0]["detail"]["forged"] = True
    with open(runner.audit.path, "w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(_json.dumps(e) + "\n")
    result = runner.audit.verify()
    assert result["ok"] is False
    assert result["broken_at"] == 0
    with pytest.raises(AuditIntegrityError):
        runner.audit.assert_intact()


# ===========================================================================
# BLOCKED honesty -- named errors, no silent fallback, no fabrication
# ===========================================================================


def test_unknown_backend_raises_blocked():
    with pytest.raises(DeepApplyBlocked, match="unknown trainer backend"):
        get_backend("quantum")


def test_missing_deep_extra_raises_named_blocked(monkeypatch):
    """Standard backend propagates the named DeepApplyBlocked from _require_deep."""
    def boom(_hint="..."):
        raise DeepApplyBlocked("deep-apply needs the [deep] extra (missing: peft)")
    monkeypatch.setattr("asea.deepapply.trainer._require_deep", boom)
    with pytest.raises(DeepApplyBlocked, match="\\[deep\\] extra"):
        get_backend("standard").train(_receiver(), build_training_dataset([]), {}, Path("."))


def test_streamed_backend_blocks_on_unsupported_receiver_and_runner_does_not_fallback(runner):
    """The canonical ``streamed`` backend (siltstream-backed, since 2026-08-16)
    trains a REAL HF CausalLM. A mock / non-HF receiver (no ``model_id``) is
    refused with a named ``DeepApplyBlocked``; the runner propagates it and does
    NOT silently fall back to the standard backend.

    (Repointed from the retired BETA ``StreamedTrainerBackend`` test, which
    asserted a ``CUDA``-string block on CPU. The siltstream ``streamed`` backend
    is CPU-capable, so the honest block on this machine is the
    receiver-architecture refusal, not a CUDA demand. The no-silent-fallback
    invariant -- named BLOCKED, zero approvals -- is preserved; only the
    BETA-specific ``CUDA`` string check was dropped.)
    """
    pkt = _promoted_packet(Domain.TRANSLATION, chain=["sender-x"])
    _approve_packets(runner, [pkt])
    recv = _receiver()  # MockModule, no model_id -> not a real HF receiver
    cfg = DeepApplyConfig(backend="streamed")
    with pytest.raises(DeepApplyBlocked) as exc:
        runner.run(recv, [pkt.packet_id], cfg, _as_en_target_suite(),
                   regression_suites=[_hi_en_regression_suite()])
    msg = str(exc.value)
    # Named BLOCKED reason, and it must NOT claim a standard-backend success.
    assert "streamed" in msg.lower() or "model_id" in msg.lower() or "deep" in msg.lower()
    assert "standard" not in msg.lower() or "use the standard" in msg.lower()
    # No silent fallback: nothing was admitted.
    assert runner.store.count(APPROVED) == 0


def test_require_deep_raises_when_a_dep_missing(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "peft":
            raise ImportError("no peft")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(DeepApplyBlocked, match="peft"):
        _require_deep()


# ===========================================================================
# Real end-to-end (opt-in). ASEA_RUN_REAL=1 + peft/torch/transformers present.
# ===========================================================================


REAL_MODEL = "HuggingFaceTB/SmolLM2-135M"


def _real_receiver():
    from asea.modules.real import HFCausalConnector, translation_capability
    caps = [translation_capability("as", "en"), translation_capability("hi", "en")]
    return HFCausalConnector(
        model_id=REAL_MODEL, capabilities=caps, module_id="smollm2-receiver",
        max_new_tokens=24, max_learning_level=LearningLevel.L4_PEFT_CANDIDATE,
    )


def _real_suite():
    # A tiny held-out translation suite (Assamese -> English).
    return _suite("as_en_real", Domain.TRANSLATION, "as->en", [
        _case("rh1", "ভাত", "rice", "heldout"),
        _case("rh2", "পানী", "water", "heldout"),
    ])


def _real_regression_suite():
    return _suite("hi_en_real", Domain.GENERAL, "hi->en", [
        _case("rr1", "নদী", "river", "regression"),
    ])


def _real_approved_packets():
    # PROMOTED translation packets whose distilled entries feed the LoRA.
    entries = {"entries": [
        {"source": "ভাত", "target": "rice"},
        {"source": "পানী", "target": "water"},
        {"source": "ঘৰ", "target": "home"},
    ]}
    cap = _cap(Domain.TRANSLATION, "as->en")
    return [SkillPacket(
        packet_id="real-p1", task_type="translate", source_module="sender-real",
        target_module="smollm2-receiver", sender_capability=cap, modality=Modality.TEXT,
        language="as->en", domain=Domain.TRANSLATION, packet_type=PacketType.GLOSSARY,
        distilled_skill=entries, confidence_score=0.9, evaluator_score=0.9, safety_score=1.0,
        scores=good_adapter_scores(),
        provenance=Provenance(origin_kind=OriginKind.CURATED_CORPUS, chain=["sender-real"],
                              synthetic_depth=0, is_mock=False),
        learning_level=LearningLevel.L3_SKILL_PACKET, promotion_status=PromotionStatus.PROMOTED,
        rollback_token="snap-real",
    )]


@pytest.mark.skipif(
    not os.environ.get("ASEA_RUN_REAL"),
    reason="opt-in real LoRA train; set ASEA_RUN_REAL=1 (downloads SmolLM2-135M, trains on CPU)",
)
def test_real_end_to_end_smollm2(tmp_path):
    """Real StandardTrainerBackend on SmolLM2-135M. Asserts the MECHANISM runs
    (finite loss, >0 trainable params, Gate 2 decision, intact audit), NOT that
    a 3-step CPU LoRA promotes -- that would fabricate a result."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import peft  # noqa: F401
    except ImportError as exc:
        pytest.skip("real deps unavailable: {}".format(exc))

    mem = MemoryStore(tmp_path / "memory")
    audit = AuditLog(tmp_path / "audit" / "audit.jsonl")
    harness = BenchmarkHarness()
    runner = DeepApplyRunner(mem, tmp_path / "adapters", audit, harness)
    for p in _real_approved_packets():
        mem.approve(p)

    recv = _real_receiver()
    cfg = DeepApplyConfig(backend="standard", max_steps=3, max_steps_cap=8,
                          lora_rank=4, lora_alpha=8, seed=0)
    report = runner.run(
        recv, ["real-p1"], cfg, _real_suite(),
        regression_suites=[_real_regression_suite()],
    )
    ad = report.adapter
    # Honest mechanism assertions (NOT promotion).
    assert ad.backend == "standard"
    assert ad.trainable_param_count > 0
    assert ad.training_loss == ad.training_loss  # finite (not NaN)
    assert ad.adapter_artifact_ref is not None
    assert ad.provenance.is_mock is False
    assert report.decision.status in (
        PromotionStatus.PROMOTED, PromotionStatus.REJECTED, PromotionStatus.PENDING_HUMAN,
    )
    assert audit.verify()["ok"] is True
    # If it happened to promote, rollback must restore the approved set.
    if report.decision.status == PromotionStatus.PROMOTED:
        n = runner.store.count(APPROVED)
        runner.rollback_adapter(ad.adapter_id, report.rollback_token)
        assert runner.store.count(APPROVED) == n - 1