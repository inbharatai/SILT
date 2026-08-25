"""Deep-apply runner -- the orchestrator that wires both gates together.

Flow::

    approved packets (Gate 1 already passed)
        -> build_training_dataset     (intake re-enforces Gate 1: PROMOTED only, no mock)
        -> trainer.train               (real LoRA; BLOCKED if deps/hardware missing)
        -> AdapterPacket               (provenance + depth + risk propagated from sources)
        -> DeepApplyEvaluator         (held-out A/B + regression sweep, reusing the harness)
        -> DeepApplyGate (Gate 2)      (full check discipline; PENDING_HUMAN if high-risk)
        -> AdapterStore.approve        (only PROMOTED; separated directories)
        -> AuditLog                    (every transition, same chain as packets)
        -> (rollback restores the approved-adapter set on demand)

Binding rules (verbatim from the build spec, enforced here):

* No gate is weakened. Gate 1 is re-enforced at intake (only PROMOTED packets
  enter training data); Gate 2 is the full check discipline on the *trained*
  adapter. Both gates are independent; neither can be skipped or relaxed by
  configuration.
* Double-gating is never collapsed. A packet passing Gate 1 says nothing about
  the adapter passing Gate 2.
* High-risk domains stay human-gated at BOTH gates. If ANY source packet is
  medical/legal/finance, the adapter parks at PENDING_HUMAN regardless of scores.
  No config bypass.
* Honest hardware. The trainer raises named DeepApplyBlocked errors; the runner
  propagates them. Nothing mock enters a real path; nothing is fabricated.
* Rollback covers weights. Admission snapshots the approved-adapter set; rollback
  restores it. Adapters are removable by construction and are never merged into
  base weights in v1.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..audit.logger import AuditLog
from ..benchmarks.harness import BenchmarkHarness, BenchmarkSuite
from ..core.interfaces import ModuleAdapter
from ..core.protocol import (
    Domain,
    LearningLevel,
    OriginKind,
    PromotionStatus,
    Provenance,
    SkillPacket,
)
from ..memory.store import APPROVED as PKT_APPROVED, MemoryStore
from .adapter_packet import AdapterPacket
from .dataset import TrainingDataset, build_training_dataset
from .errors import DeepApplyBlocked, DeepApplyIntakeError
from .evaluator import DeepApplyEvaluationReport, DeepApplyEvaluator
from .gate2 import DeepApplyGate, DeepApplyPolicy
from .store import AdapterRollbackLayer, AdapterStore, APPROVED, CANDIDATE, REJECTED
from .trainer import AdapterArtifact, TrainerBackend, get_backend, _fingerprint

# Origin-kind "synthetic-ness" ranking for provenance propagation: a training
# set built from any model-generated source is itself model-generated.
_ORIGIN_RANK = {
    OriginKind.HUMAN_VERIFIED: 0,
    OriginKind.CURATED_CORPUS: 1,
    OriginKind.MODEL_GENERATED: 2,
}


def _merged_provenance(packets: List[SkillPacket], adapter_id: str) -> Provenance:
    """Build the adapter's provenance from its source packets.

    * chain = ordered union of source chains (the receiver must not appear in it;
      ``no_self_lineage`` catches it if it does).
    * synthetic_depth = max over sources (training on depth-N data yields
      depth-N knowledge, not depth-N+1).
    * is_mock = OR over sources (one mock source taints the adapter).
    * origin_kind = the most synthetic source origin.
    """
    chain: List[str] = []
    seen = set()
    for p in packets:
        for m in p.provenance.chain:
            if m not in seen:
                seen.add(m)
                chain.append(m)
    depth = max((p.provenance.synthetic_depth for p in packets), default=0)
    is_mock = any(p.provenance.is_mock for p in packets)
    origin = max(
        (p.provenance.origin_kind for p in packets),
        key=lambda o: _ORIGIN_RANK.get(o, 0),
        default=OriginKind.HUMAN_VERIFIED,
    )
    return Provenance(
        origin_kind=origin,
        chain=chain,
        synthetic_depth=depth,
        is_mock=is_mock,
        source_reference="deep_apply/{}".format(adapter_id),
    )


@dataclass
class DeepApplyConfig:
    """Hyper-parameters + policy overrides for one deep-apply run.

    Defaults are the honest small-model-CPU path: a tiny LoRA, a handful of
    steps, deterministic seed. No field here can weaken either gate.
    """

    backend: str = "standard"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    learning_rate: float = 1e-4
    max_steps: int = 16
    max_steps_cap: int = 64
    epochs: int = 1
    seed: int = 0
    max_new_tokens: int = 48
    cpu_param_ceiling: int = 1_500_000_000
    strict_no_mock: bool = True
    regression_tolerance: float = 0.02
    # Gate 2 policy overrides (thresholds). None => use the DeepApplyPolicy default.
    min_evaluator_score: Optional[float] = None
    min_safety_score: Optional[float] = None
    min_improvement: Optional[float] = None
    max_case_regression_ratio: Optional[float] = None
    max_synthetic_depth: Optional[int] = None
    # Control-movement bound override (audit 2026-08-17). None => use the
    # DeepApplyPolicy default (0.05). A maximally-permissive config sets this
    # large (e.g. 99.0) to disable the no_control_movement check, mirroring how
    # max_case_regression_ratio=99.0 disables the case-regression check.
    max_control_movement: Optional[float] = None
    min_trainable_params: int = 1
    # Heavy-model / GPU knobs (default off = the honest small-model-CPU path).
    # ``load_in_4bit`` enables QLoRA: a 4-bit nf4 base resident on the GPU so a
    # 7B model fits in 8 GB VRAM (the resident parity pass needs the whole base
    # resident, which is why 4-bit is required for a heavy model). GPU-only --
    # the streamed backend refuses a 4-bit request without CUDA rather than
    # silently fall back. ``compute_device`` overrides the auto-detected device
    # ("cpu"/"cuda"); None = auto. ``storage_tier`` is the disk-bank tier.
    load_in_4bit: bool = False
    compute_device: Optional[str] = None
    storage_tier: str = "disk"

    def to_train_dict(self) -> Dict[str, Any]:
        """The dict the TrainerBackend reads (hyper-params only, no policy)."""
        return {
            "backend": self.backend,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": list(self.target_modules),
            "learning_rate": self.learning_rate,
            "max_steps": self.max_steps,
            "max_steps_cap": self.max_steps_cap,
            "epochs": self.epochs,
            "seed": self.seed,
            "max_new_tokens": self.max_new_tokens,
            "cpu_param_ceiling": self.cpu_param_ceiling,
            "load_in_4bit": self.load_in_4bit,
            "compute_device": self.compute_device,
            "storage_tier": self.storage_tier,
        }

    def training_config_hash(self) -> str:
        blob = repr(sorted(asdict(self).items()))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def build_policy(self) -> DeepApplyPolicy:
        kw: Dict[str, Any] = {}
        if self.min_evaluator_score is not None:
            kw["min_evaluator_score"] = self.min_evaluator_score
        if self.min_safety_score is not None:
            kw["min_safety_score"] = self.min_safety_score
        if self.min_improvement is not None:
            kw["min_improvement"] = self.min_improvement
        if self.max_case_regression_ratio is not None:
            kw["max_case_regression_ratio"] = self.max_case_regression_ratio
        if self.max_control_movement is not None:
            kw["max_control_movement"] = self.max_control_movement
        if self.max_synthetic_depth is not None:
            kw["max_synthetic_depth"] = self.max_synthetic_depth
        kw["min_trainable_params"] = self.min_trainable_params
        kw["strict_no_mock"] = self.strict_no_mock
        return DeepApplyPolicy(**kw)


class DeepApplyReport:
    """Everything one deep-apply run produced, for the final report."""

    def __init__(
        self,
        adapter: AdapterPacket,
        dataset: TrainingDataset,
        evaluation: DeepApplyEvaluationReport,
        decision,
        rollback_token: Optional[str],
        backend: str,
        session_id: str,
    ) -> None:
        self.adapter = adapter
        self.dataset = dataset
        self.evaluation = evaluation
        self.decision = decision
        self.rollback_token = rollback_token
        self.backend = backend
        self.session_id = session_id

    @property
    def status(self) -> PromotionStatus:
        return self.adapter.promotion_status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "adapter_id": self.adapter.adapter_id,
            "session_id": self.session_id,
            "backend": self.backend,
            "status": self.status.value,
            "base_model": self.adapter.base_model,
            "target_module": self.adapter.target_module,
            "source_packet_ids": self.adapter.source_packet_ids,
            "source_domains": [d.value for d in self.adapter.source_domains],
            "synthetic_depth": self.adapter.synthetic_depth,
            "risk_tier": self.adapter.risk_tier.value,
            "trainable_param_count": self.adapter.trainable_param_count,
            "training_loss": self.adapter.training_loss,
            "dataset_hash": self.adapter.dataset_hash,
            "dataset_rows": self.dataset.manifest["row_count"],
            "evaluator_score": self.adapter.evaluator_score,
            "improvement": (self.adapter.scores.improvement if self.adapter.scores else None),
            "case_regressions": (
                self.adapter.scores.case_regression_count if self.adapter.scores else 0
            ),
            "case_count": self.adapter.scores.case_count if self.adapter.scores else 0,
            "rollback_token": self.rollback_token,
            # parity_verified is surfaced on the train_completed / gate2_decision
            # SSE events (live). Mirror it into the persisted report so a Studio
            # client re-opening a finished run (Past runs -> watchTrain, where the
            # SSE queue has already drained) can reconstruct the parity line
            # honestly instead of showing a stale "—".
            "parity_verified": bool(getattr(self.adapter, "parity_verified", False)),
            "gate2": self.decision.to_dict(),
        }


class DeepApplyRunner:
    """Orchestrates build -> train -> evaluate -> Gate 2 -> admit/reject/pending."""

    def __init__(
        self,
        memory_store: MemoryStore,
        adapter_root: Path,
        audit: AuditLog,
        harness: Optional[BenchmarkHarness] = None,
        policy: Optional[DeepApplyPolicy] = None,
        regression_tolerance: float = 0.02,
    ) -> None:
        self.memory_store = memory_store
        self.adapter_root = Path(adapter_root)
        self.adapter_root.mkdir(parents=True, exist_ok=True)
        self.audit = audit
        self.harness = harness or BenchmarkHarness()
        self.store = AdapterStore(self.adapter_root)
        self.rollback = AdapterRollbackLayer(self.store)
        self.regression_tolerance = regression_tolerance
        # Default policy; per-run config can override via build_policy().
        self.policy = policy or DeepApplyPolicy()
        # Per-run Gate 2 policies, keyed by adapter_id. approve_pending MUST
        # re-run Gate 2 under the SAME thresholds the run used (a high-risk
        # adapter parked at PENDING_HUMAN under strict thresholds must not be
        # re-admitted under relaxed defaults). Falls back to self.policy if the
        # run was on a different runner instance.
        self._run_policies: Dict[str, DeepApplyPolicy] = {}

    # -- intake ------------------------------------------------------------

    def _resolve_packets(self, packet_ids: List[str]) -> List[SkillPacket]:
        packets: List[SkillPacket] = []
        for pid in packet_ids:
            try:
                pkt = self.memory_store.get(PKT_APPROVED, pid)
            except Exception as exc:
                raise DeepApplyIntakeError(
                    "packet {} is not in approved/ (Gate 1 not passed): {}".format(pid, exc)
                ) from exc
            packets.append(pkt)
        return packets

    # -- main --------------------------------------------------------------

    def run(
        self,
        receiver: ModuleAdapter,
        packet_ids: List[str],
        config: DeepApplyConfig,
        target_suite: BenchmarkSuite,
        regression_suites: Optional[List[BenchmarkSuite]] = None,
        human_approver: Optional[str] = None,
        actor: str = "deep_apply",
        trainer: Optional[TrainerBackend] = None,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> DeepApplyReport:
        session_id = "da-{}".format(uuid.uuid4().hex[:12])

        # Telemetry channel (optional, default None -> byte-identical to every
        # existing caller/test). Emits REAL phase + per-step events the Studio
        # live graph reads: dataset_built, backend_selected, train_started,
        # train_step (per-step loss, streamed/standard only), train_completed,
        # gate2_evaluated, gate2_decision, done. Zeroforge's per-step loop is in
        # vendored siltstream so it emits phase-level only (no faked per-step).
        # Telemetry must NEVER break a run: a buggy observer is swallowed.
        def _emit(event: Dict[str, Any]) -> None:
            if on_progress is not None:
                try:
                    on_progress(event)
                except Exception:
                    pass

        _emit({"phase": "session_started", "session_id": session_id,
               "adapter_id": None, "backend": config.backend})

        # 1. intake -- re-enforce Gate 1: only PROMOTED packets, no mock.
        packets = self._resolve_packets(packet_ids)
        dataset = build_training_dataset(packets, strict_no_mock=config.strict_no_mock)
        if not dataset.rows:
            raise DeepApplyBlocked(
                "training set built from {} packet(s) yielded 0 rows; cannot train an "
                "adapter on nothing".format(len(packets))
            )
        _emit({"phase": "dataset_built", "session_id": session_id,
               "adapter_id": None, "rows": dataset.manifest["row_count"],
               "source_packet_ids": dataset.source_packet_ids})

        # 2. pick the trainer backend. ``trainer`` injection is the seam the test
        #    double (ScriptedTrainerBackend) uses; real runs leave it None and get
        #    the configured real backend. The runner NEVER silently falls back to
        #    a different backend -- a BLOCKED error propagates.
        backend_obj = trainer or get_backend(config.backend)
        backend_name = backend_obj.name

        adapter_id = str(uuid.uuid4())
        out_dir = self.adapter_root / CANDIDATE / adapter_id / "artifact"
        out_dir.mkdir(parents=True, exist_ok=True)

        # stream_backend_selected: record WHICH backend (and what it claims it
        # can run) before training, in the same hash chain. The siltstream
        # backends expose capabilities(); scripted doubles do not.
        caps = None
        cap_attr = getattr(backend_obj, "capabilities", None)
        if callable(cap_attr):
            try:
                caps = cap_attr()
            except Exception:
                caps = None
        self.audit.append(
            "stream_backend_selected",
            actor=actor,
            session_id=session_id,
            packet_id=adapter_id,
            detail={
                "backend": backend_name,
                "version": getattr(backend_obj, "version", ""),
                "capabilities": caps,
                "base_model": getattr(receiver, "model_id", None) or receiver.module_id,
            },
        )

        _emit({"phase": "backend_selected", "session_id": session_id,
               "adapter_id": adapter_id, "backend": backend_name,
               "base_model": getattr(receiver, "model_id", None) or receiver.module_id})

        self.audit.append(
            "train_started",
            actor=actor,
            session_id=session_id,
            packet_id=adapter_id,
            detail={
                "backend": backend_name,
                "base_model": getattr(receiver, "model_id", None) or receiver.module_id,
                "target_module": receiver.module_id,
                "source_packet_ids": dataset.source_packet_ids,
                "dataset_hash": dataset.dataset_hash,
                "dataset_rows": dataset.manifest["row_count"],
            },
        )

        # 3. train -- real weights. May raise DeepApplyBlocked (named reason).
        #    The _audit context lets the streamed/zeroforge backends write
        #    parity_check events into THIS run's hash chain (even on failure).
        train_cfg = config.to_train_dict()
        train_cfg["_audit"] = {
            "log": self.audit,
            "session_id": session_id,
            "adapter_id": adapter_id,
            "actor": actor,
        }
        # Thread the per-step telemetry hook to the backend (stashed in the cfg
        # like ``_audit`` -> signature unchanged). Each backend calls it with a
        # ``train_step`` event carrying the REAL loss for that step; zeroforge
        # has no Python per-step loop so it simply never calls it (honest).
        def _on_step(ev: Dict[str, Any]) -> None:
            _emit({"phase": "train_step", "session_id": session_id,
                   "adapter_id": adapter_id, "backend": backend_name, **ev})
        train_cfg["_on_step"] = _on_step
        _emit({"phase": "train_started", "session_id": session_id,
               "adapter_id": adapter_id, "backend": backend_name,
               "max_steps": int(train_cfg.get("max_steps", 0))})
        try:
            artifact: AdapterArtifact = backend_obj.train(
                receiver, dataset, train_cfg, out_dir
            )
        except Exception as exc:
            _emit({"phase": "train_failed", "session_id": session_id,
                   "adapter_id": adapter_id, "backend": backend_name,
                   "error": type(exc).__name__, "reason": str(exc)})
            raise

        self.audit.append(
            "train_completed",
            actor=actor,
            session_id=session_id,
            packet_id=adapter_id,
            detail={
                "backend": backend_name,
                "backend_version": artifact.backend_version,
                "trainable_param_count": artifact.trainable_param_count,
                "training_loss": artifact.training_loss,
                "artifact_ref": getattr(artifact, "adapter_path", None),
            },
        )
        _emit({"phase": "train_completed", "session_id": session_id,
               "adapter_id": adapter_id, "backend": backend_name,
               "trainable_param_count": artifact.trainable_param_count,
               "training_loss": artifact.training_loss,
               "parity_verified": bool(getattr(artifact, "parity_verified", False))})

        # 4. build the AdapterPacket -- provenance/depth/risk propagated.
        base_model = getattr(receiver, "model_id", None) or receiver.module_id
        prov = _merged_provenance(packets, adapter_id)
        adapter = AdapterPacket(
            adapter_id=adapter_id,
            base_model=base_model,
            base_model_fingerprint=_fingerprint(base_model, artifact.lora_config),
            target_module=receiver.module_id,
            source_packet_ids=dataset.source_packet_ids,
            source_domains=dataset.source_domains,
            synthetic_depth=dataset.synthetic_depth,
            provenance=prov,
            learning_level=LearningLevel.L4_PEFT_CANDIDATE,
            lora_config=dict(artifact.lora_config),
            training_config_hash=config.training_config_hash(),
            dataset_hash=dataset.dataset_hash,
            seed=config.seed,
            backend=artifact.backend,
            backend_version=artifact.backend_version,
            trainable_param_count=artifact.trainable_param_count,
            training_loss=artifact.training_loss,
            adapter_artifact_ref=getattr(artifact, "adapter_path", None),
            # Streamed-backend metadata (parity is the admission bar). Standard
            # backend artifacts do not carry these -> honest defaults (empty /
            # False) flow through getattr.
            storage_tier=getattr(artifact, "storage_tier", "") or "",
            parity_verified=bool(getattr(artifact, "parity_verified", False)),
            parity_report_hash=getattr(artifact, "parity_report_hash", "") or "",
            config_fingerprint=getattr(artifact, "config_fingerprint", "") or "",
            safety_score=dataset.safety_floor,
            notes={
                "regression_suites": [s.suite_id for s in (regression_suites or [])],
                # Full parity report block (diffs, device, dtype, notes) for the
                # streamed/zeroforge backends; absent for the standard backend.
                "parity": getattr(artifact, "parity", None),
                "forward_passes": getattr(artifact, "forward_passes", None),
                "backward_passes": getattr(artifact, "backward_passes", None),
            },
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        # 5. evaluate -- held-out A/B + regression sweep (reuses the harness).
        #    Build the per-run policy FIRST so the evaluator and Gate 2 share
        #    ONE source of truth for the thresholds -- in particular the control
        #    movement bound (audit 2026-08-17): the evaluator computes the
        #    movement bool with run_policy.max_control_movement, and Gate 2's
        #    no_control_movement check reads that same bool.
        run_policy = config.build_policy()
        self._run_policies[adapter_id] = run_policy
        evaluator = DeepApplyEvaluator(
            self.harness, self.regression_tolerance, run_policy.max_control_movement
        )
        evaluation = evaluator.evaluate(
            adapter, artifact, receiver, target_suite, regression_suites or []
        )
        self.audit.append(
            "gate2_evaluated",
            actor=actor,
            session_id=session_id,
            packet_id=adapter_id,
            detail=evaluation.to_dict(),
        )
        _emit({"phase": "gate2_evaluated", "session_id": session_id,
               "adapter_id": adapter_id,
               "candidate_score": getattr(evaluation, "candidate_score", None),
               "baseline_score": getattr(evaluation, "baseline_score", None)})

        # 6. Gate 2 -- full check discipline on the trained adapter. Per-run
        #    config builds the policy (threshold overrides); the runner default
        #    policy is the base approve_pending uses.
        gate = DeepApplyGate(run_policy)
        rollback_token = self.rollback.snapshot(
            label="pre-admission:{}".format(adapter_id)
        )
        decision = gate.apply(
            adapter, rollback_token=rollback_token, human_approver=human_approver
        )
        self.audit.append(
            "gate2_decision",
            actor=actor,
            session_id=session_id,
            packet_id=adapter_id,
            detail=decision.to_dict(),
        )
        _emit({"phase": "gate2_decision", "session_id": session_id,
               "adapter_id": adapter_id,
               "status": decision.status.value if decision.status else None,
               "needs_human": decision.needs_human,
               "parity_verified": bool(getattr(adapter, "parity_verified", False))})

        # 7. record the outcome. The store decides nothing; it only records.
        if decision.status == PromotionStatus.PROMOTED:
            try:
                self.store.approve(adapter)
            except Exception as exc:
                # Duplicate-content guard fired (identical adapter already approved
                # for this base model). Record as rejected with the store's reason,
                # reason-before-status.
                adapter.rejection_reason = str(exc)
                adapter.promotion_status = PromotionStatus.REJECTED
                self.store.put_rejected(adapter)
                self.audit.append(
                    "adapter_duplicate_refused",
                    actor=actor,
                    session_id=session_id,
                    packet_id=adapter_id,
                    detail={"reason": str(exc)},
                )
            else:
                self.audit.append(
                    "adapter_admitted",
                    actor=human_approver or actor,
                    session_id=session_id,
                    packet_id=adapter_id,
                    detail={
                        "rollback_token": rollback_token,
                        "artifact_ref": adapter.adapter_artifact_ref,
                    },
                )
        elif decision.status == PromotionStatus.PENDING_HUMAN:
            self.store.put_candidate(adapter)
            self.audit.append(
                "adapter_pending_human",
                actor=actor,
                session_id=session_id,
                packet_id=adapter_id,
                detail={
                    "source_domains": [d.value for d in adapter.source_domains],
                    "risk_tier": adapter.risk_tier.value,
                },
            )
        else:
            self.store.put_rejected(adapter)
            self.audit.append(
                "adapter_rejected",
                actor=actor,
                session_id=session_id,
                packet_id=adapter_id,
                detail={"reason": adapter.rejection_reason},
            )

        _emit({"phase": "done", "session_id": session_id,
               "adapter_id": adapter_id, "backend": backend_name,
               "status": decision.status.value if decision.status else None,
               "parity_verified": bool(getattr(adapter, "parity_verified", False)),
               "training_loss": artifact.training_loss,
               "trainable_param_count": artifact.trainable_param_count})
        return DeepApplyReport(
            adapter=adapter,
            dataset=dataset,
            evaluation=evaluation,
            decision=decision,
            rollback_token=rollback_token,
            backend=backend_name,
            session_id=session_id,
        )

    # -- human-in-the-loop -------------------------------------------------

    def approve_pending(self, adapter_id: str, approver: str, actor: str = "deep_apply"):
        """Re-run Gate 2 with a named human approver for a parked adapter.

        Human approval satisfies exactly one check; it does not waive safety,
        regression, provenance or depth. The gate is re-run in full.
        """
        adapter = self.store.get(CANDIDATE, adapter_id)
        rollback_token = self.rollback.snapshot(label="pre-approval:{}".format(adapter_id))
        # Re-run Gate 2 under the SAME policy the run used (captured at run
        # time), not the default -- a high-risk adapter parked at PENDING_HUMAN
        # under strict thresholds must not be re-admitted under relaxed ones.
        policy = self._run_policies.get(adapter_id, self.policy)
        decision = DeepApplyGate(policy).apply(
            adapter, rollback_token=rollback_token, human_approver=approver
        )
        self.audit.append(
            "gate2_human_decision",
            actor=approver,
            packet_id=adapter_id,
            detail=decision.to_dict(),
        )
        if decision.approved:
            try:
                self.store.approve(adapter)
            except Exception as exc:
                # Mirror run()'s duplicate-content handling: an identical adapter
                # already approved -> record rejected with the store's reason
                # rather than propagating RollbackError and leaving the
                # candidate half-flipped with no audit event.
                adapter.rejection_reason = str(exc)
                adapter.promotion_status = PromotionStatus.REJECTED
                self.store.put_rejected(adapter)
                self.audit.append(
                    "adapter_duplicate_refused",
                    actor=approver,
                    packet_id=adapter_id,
                    detail={"reason": str(exc)},
                )
            else:
                self.audit.append(
                    "adapter_admitted",
                    actor=approver,
                    packet_id=adapter_id,
                    detail={"rollback_token": rollback_token},
                )
        else:
            self.store.put_rejected(adapter)
        return decision

    # -- rollback ---------------------------------------------------------

    def rollback_adapter(self, adapter_id: str, token: str, actor: str = "operator") -> Dict[str, Any]:
        """Detach an admitted adapter and restore the prior approved-adapter set.

        The receiver returns to its exact pre-admission behaviour because
        adapters are removable by construction and are never merged into base
        weights in v1.
        """
        # Capture the adapter record before the restore clears the approved dir.
        adapter: Optional[AdapterPacket] = None
        for bucket in (APPROVED, CANDIDATE, REJECTED):
            try:
                adapter = self.store.get(bucket, adapter_id)
                break
            except Exception:
                continue
        result = self.rollback.rollback(token)
        if adapter is not None:
            adapter.promotion_status = PromotionStatus.ROLLED_BACK
            self.store.put(adapter, REJECTED)  # audit record, status=ROLLED_BACK
        self.audit.append(
            "adapter_rolled_back",
            actor=actor,
            packet_id=adapter_id,
            detail={"token": token, **result},
        )
        return result


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def deep_apply(
    receiver: ModuleAdapter,
    packet_ids: List[str],
    config: DeepApplyConfig,
    target_suite: BenchmarkSuite,
    regression_suites: Optional[List[BenchmarkSuite]] = None,
    *,
    workspace: Path,
    human_approver: Optional[str] = None,
    trainer: Optional[TrainerBackend] = None,
) -> DeepApplyReport:
    """One-shot entry point. Builds default stores under ``workspace``."""
    workspace = Path(workspace)
    memory_store = MemoryStore(workspace / "memory")
    audit = AuditLog(workspace / "audit" / "audit.jsonl")
    harness = BenchmarkHarness()
    runner = DeepApplyRunner(memory_store, workspace / "adapters", audit, harness)
    return runner.run(
        receiver,
        packet_ids,
        config,
        target_suite,
        regression_suites,
        human_approver=human_approver,
        trainer=trainer,
    )


def from_pipeline(pipeline, adapter_root: Optional[Path] = None) -> DeepApplyRunner:
    """Build a runner that shares a pipeline's memory store, audit and harness.

    The adapter store is separate from the packet memory store (different
    directory, different artefact kind) but lives under the same workspace.
    """
    root = Path(adapter_root) if adapter_root else (pipeline.workspace / "adapters")
    return DeepApplyRunner(
        memory_store=pipeline.store,
        adapter_root=root,
        audit=pipeline.audit,
        harness=pipeline.harness,
    )