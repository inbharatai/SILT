"""The universal pipeline.

This module is the point of the whole project. Note what it does NOT contain:
there is no ``if modality == ...`` anywhere below. Text, TTS, code and
structured-rule transfers traverse byte-identical orchestration; only the
plugins resolved out of the PluginRegistry differ. tests/test_conformance.py
asserts this property rather than trusting the comment.

Stage order, and why each exists:

    handshake        two modules agree they have something to say
    gap negotiation  refuse to act without measured evidence of a deficiency
    extraction       probe the sender on the diagnostic split only
    relevance filter discard signals that cannot help
    safety filter    discard signals that could harm
    distillation     compress into an inspectable payload; drop raw output
    evaluation       held-out A/B plus a regression sweep
    snapshot         make the next step reversible before taking it
    promotion gate   all-or-nothing rule check
    store + audit    separate approved from candidate; record everything
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..audit.logger import AuditLog
from ..benchmarks.cache import TeacherScoreCache
from ..benchmarks.harness import BenchmarkHarness, BenchmarkSuite
from ..evaluator.evaluator import Evaluator
from ..filters.relevance import RelevanceFilter
from ..filters.safety import SafetyFilter
from ..memory.store import MemoryStore, RollbackLayer
from ..promotion.gate import PromotionGate
from ..registry.registries import (
    AdapterRegistry,
    ModuleRegistry,
    ReceiverRegistry,
    SenderRegistry,
)
from .errors import PluginNotFound, RollbackError
from .gap import GapEngine
from .handshake import Handshake, Session
from .interfaces import ModuleAdapter, SimilarityBackend
from .plugins import PluginRegistry, default_registry
from .protocol import LearningLevel, PromotionStatus, SkillPacket


class RunReport:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.negotiation: Dict[str, Any] = {}
        self.extracted = 0
        self.dropped_relevance: List[Dict[str, str]] = []
        self.dropped_safety: List[Dict[str, str]] = []
        self.distilled: List[SkillPacket] = []
        self.evaluations: List[Dict[str, Any]] = []
        self.decisions: List[Dict[str, Any]] = []
        self.promoted: List[str] = []
        self.pending_human: List[str] = []
        self.rejected: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session": self.session.to_dict(),
            "negotiation": self.negotiation,
            "counts": {
                "extracted": self.extracted,
                "dropped_relevance": len(self.dropped_relevance),
                "dropped_safety": len(self.dropped_safety),
                "distilled": len(self.distilled),
                "promoted": len(self.promoted),
                "pending_human": len(self.pending_human),
                "rejected": len(self.rejected),
            },
            "dropped_relevance": self.dropped_relevance,
            "dropped_safety": self.dropped_safety,
            "evaluations": self.evaluations,
            "decisions": self.decisions,
            "promoted": self.promoted,
            "pending_human": self.pending_human,
            "rejected": self.rejected,
            "mock_warning": (
                "This run involved at least one MOCK module; results describe "
                "pipeline behaviour, not model capability."
                if self.session.involves_mock
                else None
            ),
        }


class Pipeline:
    def __init__(
        self,
        workspace: Path,
        plugins: Optional[PluginRegistry] = None,
        harness: Optional[BenchmarkHarness] = None,
        gap_engine: Optional[GapEngine] = None,
        relevance: Optional[RelevanceFilter] = None,
        safety: Optional[SafetyFilter] = None,
        evaluator: Optional[Evaluator] = None,
        gate: Optional[PromotionGate] = None,
        similarity: Optional[SimilarityBackend] = None,
        teacher_cache: Optional[TeacherScoreCache] = None,
    ) -> None:
        self.workspace = Path(workspace)
        self.modules = ModuleRegistry()
        self.senders = SenderRegistry(self.modules)
        self.receivers = ReceiverRegistry(self.modules)
        self.adapters = AdapterRegistry(self.modules)

        self.plugins = plugins or default_registry()
        self.harness = harness or BenchmarkHarness(
            plugins=self.plugins, similarity=similarity
        )
        # When a caller supplies a gap_engine we use it verbatim (its own
        # teacher_cache, if any, wins); otherwise we construct one and pass the
        # caller's teacher_cache through so the pipeline can opt into sender
        # extraction caching without constructing the engine themselves.
        self.gap_engine = gap_engine or GapEngine(
            harness=self.harness, teacher_cache=teacher_cache
        )
        # One similarity backend for the whole pipeline. Letting the harness and
        # the relevance filter use different backends silently invalidates a run:
        # you would filter with a lexical proxy while scoring with embeddings and
        # report the result as semantically evaluated. Derived from the harness so
        # they cannot drift apart.
        self.relevance = relevance or RelevanceFilter(similarity=self.harness.similarity)
        self.safety = safety or SafetyFilter()
        self.evaluator = evaluator or Evaluator(harness=self.harness)
        self.gate = gate or PromotionGate()

        self.store = MemoryStore(self.workspace / "memory")
        self.rollback = RollbackLayer(self.store)
        self.audit = AuditLog(self.workspace / "audit" / "audit.jsonl")
        self.handshake = Handshake()

    # -- setup ------------------------------------------------------------

    def register_module(self, module: ModuleAdapter, replace: bool = False) -> ModuleAdapter:
        self.modules.register_module(module, replace=replace)
        self.audit.append(
            "module_registered",
            actor="system",
            detail={
                "module_id": module.module_id,
                "roles": module.manifest().roles,
                "is_mock": module.is_mock,
                "capabilities": [c.as_str() for c in module.manifest().capabilities],
            },
        )
        return module

    def bind_adapter(
        self, adapter_id: str, sender_id: str, receiver_id: str, description: str = ""
    ):
        binding = self.adapters.bind(adapter_id, sender_id, receiver_id, description)
        self.audit.append("adapter_bound", actor="system", detail=binding.to_dict())
        return binding

    # -- main flow --------------------------------------------------------

    def run(
        self,
        adapter_id: str,
        suites: List[BenchmarkSuite],
        requested_level: LearningLevel = LearningLevel.L3_SKILL_PACKET,
        human_approver: Optional[str] = None,
        actor: str = "pipeline",
    ) -> RunReport:
        binding = self.adapters.get(adapter_id)
        sender = self.modules.get(binding.sender_id)
        receiver = self.modules.get(binding.receiver_id)

        session = self.handshake.open(adapter_id, sender, receiver, requested_level)
        self.audit.append(
            "session_opened", actor=actor, session_id=session.session_id,
            detail=session.to_dict(),
        )
        report = RunReport(session)

        # 1. what does the receiver actually lack? Measure ONCE: negotiate_with_gaps
        # scores both modules on the extraction split and returns the loggable
        # report dict plus the actionable Gap objects. A second measure() call
        # here would re-run the entire sender+receiver evaluation across all
        # suites -- wasted GPU time for real models and identical results under
        # do_sample=False.
        report.negotiation, gaps = self.gap_engine.negotiate_with_gaps(
            sender, receiver, suites
        )
        self.audit.append(
            "gap_negotiated", actor=actor, session_id=session.session_id,
            detail=report.negotiation,
        )
        if not gaps:
            self.audit.append(
                "run_complete", actor=actor, session_id=session.session_id,
                detail={"result": "no actionable gap; nothing transferred"},
            )
            return report

        suite_by_capability = {s.capability().as_str(): s for s in suites}

        for gap in gaps:
            cap_str = gap.capability.as_str()
            suite = suite_by_capability[cap_str]
            try:
                extractor = self.plugins.extractor(gap.capability.modality)
                distiller = self.plugins.distiller(gap.capability.modality)
            except PluginNotFound as exc:
                self.audit.append(
                    "plugin_missing", actor=actor, session_id=session.session_id,
                    detail={"capability": cap_str, "error": str(exc)},
                )
                continue

            # 2. extraction -- diagnostic split only
            probes = [
                {
                    "case_id": c.case_id,
                    "prompt": c.prompt,
                    "expected": c.expected,
                    "meta": c.meta,
                    "source_reference": "{}#{}".format(suite.suite_id, c.case_id),
                }
                for c in suite.split("extraction")
            ]
            raw = extractor.extract(sender, receiver, gap, probes)
            report.extracted += len(raw)
            for p in raw:
                p.learning_level = session.negotiated_level
            self.audit.append(
                "extracted", actor=actor, session_id=session.session_id,
                detail={"capability": cap_str, "count": len(raw)},
            )

            # 3. relevance
            kept, dropped = self.relevance.apply(raw, receiver)
            for p in dropped:
                self.store.put_rejected(p)
                report.dropped_relevance.append(
                    {"packet_id": p.packet_id, "reason": p.rejection_reason or ""}
                )
            self.audit.append(
                "relevance_filtered", actor=actor, session_id=session.session_id,
                detail={"kept": len(kept), "dropped": len(dropped)},
            )

            # 4. safety
            kept, unsafe = self.safety.apply(kept)
            for p in unsafe:
                self.store.put_rejected(p)
                report.dropped_safety.append(
                    {"packet_id": p.packet_id, "reason": p.rejection_reason or ""}
                )
            self.audit.append(
                "safety_filtered", actor=actor, session_id=session.session_id,
                detail={"kept": len(kept), "dropped": len(unsafe)},
            )
            if not kept:
                continue

            for p in kept:
                self.store.put_candidate(p)

            # 5. distillation
            distilled = distiller.distill(kept)
            report.distilled.extend(distilled)
            for p in distilled:
                self.store.put_candidate(p)
                self.audit.append(
                    "distilled", actor=actor, session_id=session.session_id,
                    packet_id=p.packet_id,
                    detail={
                        "packet_type": p.packet_type.value if p.packet_type else None,
                        "members": p.notes.get("member_count"),
                        "content_hash": p.content_hash(),
                    },
                )

            # 6..9 per distilled packet
            regression_suites = [s for s in suites if s.capability().as_str() != cap_str]
            for packet in distilled:
                self._finalize(
                    packet, receiver, suite, regression_suites, session,
                    report, human_approver, actor,
                )

        self.audit.append(
            "run_complete", actor=actor, session_id=session.session_id,
            detail=report.to_dict()["counts"],
        )
        return report

    # -- per-packet tail --------------------------------------------------

    def _finalize(
        self,
        packet: SkillPacket,
        receiver: ModuleAdapter,
        suite: BenchmarkSuite,
        regression_suites: List[BenchmarkSuite],
        session: Session,
        report: RunReport,
        human_approver: Optional[str],
        actor: str,
    ) -> None:
        evaluation = self.evaluator.evaluate(packet, receiver, suite, regression_suites)
        report.evaluations.append(evaluation.to_dict())
        self.audit.append(
            "evaluated", actor=actor, session_id=session.session_id,
            packet_id=packet.packet_id, detail=evaluation.to_dict(),
        )

        # Snapshot BEFORE the gate so a rollback token exists to be checked.
        token = self.rollback.snapshot(label="pre-promotion:{}".format(packet.packet_id))
        decision = self.gate.apply(
            packet, rollback_token=token, human_approver=human_approver
        )
        report.decisions.append(decision.to_dict())
        self.audit.append(
            "gate_decision", actor=actor, session_id=session.session_id,
            packet_id=packet.packet_id, detail=decision.to_dict(),
        )

        if decision.status == PromotionStatus.PROMOTED:
            try:
                self.store.approve(packet)
            except RollbackError as exc:
                # Duplicate-content guard fired: identical content is already
                # approved for this receiver (typical on a re-run). Not an
                # error -- record it as a rejection with the store's reason.
                # Reason before status: the schema forbids a REJECTED packet
                # without a rejection_reason, and validate_assignment enforces
                # it on every attribute write.
                packet.rejection_reason = str(exc)
                packet.promotion_status = PromotionStatus.REJECTED
                self.store.put_rejected(packet)
                report.rejected.append(packet.packet_id)
                self.audit.append(
                    "duplicate_refused", actor=actor, session_id=session.session_id,
                    packet_id=packet.packet_id, detail={"reason": str(exc)},
                )
                return
            report.promoted.append(packet.packet_id)
            self.audit.append(
                "promoted", actor=human_approver or actor,
                session_id=session.session_id, packet_id=packet.packet_id,
                detail={"rollback_token": token},
            )
        elif decision.status == PromotionStatus.PENDING_HUMAN:
            self.store.put_candidate(packet)
            report.pending_human.append(packet.packet_id)
            self.audit.append(
                "pending_human_approval", actor=actor, session_id=session.session_id,
                packet_id=packet.packet_id,
                detail={"domain": packet.domain.value, "risk_tier": packet.risk_tier.value},
            )
        else:
            self.store.put_rejected(packet)
            report.rejected.append(packet.packet_id)
            self.audit.append(
                "rejected", actor=actor, session_id=session.session_id,
                packet_id=packet.packet_id,
                detail={"reason": packet.rejection_reason},
            )

    # -- human-in-the-loop ------------------------------------------------

    def approve_pending(self, packet_id: str, approver: str) -> Dict[str, Any]:
        """Apply a human decision to a packet parked in PENDING_HUMAN.

        The gate is re-run in full. Human approval satisfies exactly one check;
        it does not waive safety, regression or provenance requirements.
        """
        packet = self.store.get("candidate", packet_id)
        decision = self.gate.apply(
            packet, rollback_token=packet.rollback_token, human_approver=approver
        )
        if decision.approved:
            self.store.approve(packet)
        else:
            # reject() (not put_rejected): unlink the candidate entry so the
            # packet does not linger in candidate/ with its old
            # pending_human_approval status and keep reappearing in
            # cmd_report's pending_human listing. approve() already unlinks;
            # the reject path must be symmetric.
            self.store.reject(packet)
        self.audit.append(
            "human_decision", actor=approver, packet_id=packet_id,
            detail=decision.to_dict(),
        )
        return decision.to_dict()

    def rollback_to(self, token: str, actor: str = "operator") -> Dict[str, Any]:
        result = self.rollback.rollback(token)
        self.audit.append("rollback", actor=actor, detail=result)
        return result
