"""Deep-apply job manager (Studio surface for weights-mode training).

A deep-apply run trains a LoRA adapter on an HF receiver from packets that
ALREADY passed Gate 1, then Gate 2 measures the trained adapter's outcome on
held-out data before admission. It takes minutes on real weights, so runs
execute on a worker thread while the API stays responsive.

The telemetry channel is the point of this surface: the runner's
``on_progress`` callback receives a REAL phase + per-step event stream
(``session_started -> dataset_built -> backend_selected -> train_started ->
train_step... -> train_completed -> gate2_evaluated -> gate2_decision -> done``),
and this module drains that stream into an SSE channel the Studio renders as a
live loss curve + step gauge + parity/gate2 verdict. Per-step losses come from
the backend's own loop (streamed/standard); zeroforge's per-step loop is in
vendored siltstream so it emits phase-level only -- never a fabricated per-step.

HONESTY (binding, the "no loopholes" core):

  * No gate is weakened. The runner is the SAME ``DeepApplyRunner`` the CLI/tests
    use; this module only calls ``runner.run(...)`` with a real ``DeepApplyConfig``
    and drains its telemetry. Gate 2 runs in full (held-out A/B + parity +
    safety + control-movement). High-risk source domains still park at
    PENDING_HUMAN regardless of scores.
  * Parity is the admission bar; parity unverified/failed blocks (DeepApplyBlocked
    propagates as a typed failure, never a silent pass).
  * The receiver must be an HF in-process connector (LoRA trains on weights the
    process owns). Ollama receivers hold weights externally and CANNOT be
    deep-applied -- ``backend.supports(receiver)`` is False and the endpoint
    refuses up front with a 400, not a silent fallback.
  * Telemetry carries REAL numbers (per-step loss, trainable count, parity bool,
    gate2 status) read from the runner's own events. Nothing is invented; a
    phase that did not run is simply absent from the stream.
  * The raw exception text is kept server-side (logged via traceback). The
    client learns the typed error NAME and a correlation ref, never paths or
    deep-dependency messages -- matching ``jobs.py``'s redaction policy.
  * LOCAL ONLY; patent pending (India): this surface trains and reads weights on the
    local machine; it touches no network.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..benchmarks.harness import load_all, load_suite
from ..core.pipeline import Pipeline
from ..deepapply.errors import DeepApplyBlocked
from . import catalog
from ._jsonsafe import json_safe

ROOT = Path(__file__).resolve().parent.parent.parent.parent
BENCHMARKS = ROOT / "data" / "benchmarks"

#: Backends the Studio exposes (the trainer registry's real names).
_BACKENDS = ("standard", "streamed", "zeroforge")


def _coerce_config(backend: str, overrides: Dict[str, Any]):
    """Build a DeepApplyConfig from the requested backend + UI overrides.

    EXPLICIT training-only allowlist (adversarial review 2026-08-18): the
    permissive ``asdict(cfg).keys()`` path forwarded Gate 2 threshold knobs
    (``min_evaluator_score`` / ``min_safety_score`` / ``min_improvement`` /
    ``max_case_regression_ratio`` / ``max_control_movement`` / ``max_synthetic_depth``
    / ``min_trainable_params`` / ``strict_no_mock`` / ``regression_tolerance``)
    and the hardware-honesty ceiling (``cpu_param_ceiling``), so a direct API
    client could weaken Gate 2 to pass garbage. The Studio surface may only
    shape the TRAINING (not the gating): LoRA shape, optimiser, steps, seed,
    quantization flag, device, storage tier. Anything else is dropped.
    """
    from ..deepapply.runner import DeepApplyConfig

    cfg = DeepApplyConfig(backend=backend)
    # Training-shape knobs only. None of these can weaken Gate 1, Gate 2, the
    # parity gate, the hardware-honesty ceiling, or the high-risk human rule.
    allow = {
        "lora_rank", "lora_alpha", "lora_dropout", "target_modules",
        "learning_rate", "max_steps", "max_steps_cap", "epochs", "seed",
        "load_in_4bit", "compute_device", "storage_tier",
    }
    for k, v in (overrides or {}).items():
        if k in allow and v is not None:
            setattr(cfg, k, v)
    return cfg


class DeepApplyJob:
    """One deep-apply run, executed on a worker thread with live telemetry."""

    def __init__(
        self,
        config: Dict[str, Any],
        workspace_root: Path,
        source_pipeline: Pipeline,
    ) -> None:
        self.job_id = "da-{}".format((id(self) ^ int(time.time() * 1000)) & 0xFFFFFFFF)
        self.config = config
        self.workspace = Path(workspace_root) / self.job_id
        self.source_pipeline = source_pipeline
        self.status = "queued"        # queued | running | done | failed
        self.error_type: Optional[str] = None
        self.error: Optional[str] = None   # server-side only (raw), never sent to client
        self.report: Optional[Dict[str, Any]] = None
        self.created_at = datetime.now(timezone.utc).isoformat()
        self._telemetry: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _emit(self, event: Dict[str, Any]) -> None:
        # Thread-safe handoff to the SSE drain. ``on_progress`` is called from
        # the worker thread; the SSE handler reads from this queue. Sanitize at
        # the boundary: a non-finite per-step loss / parity ratio from the
        # runner would serialize to a literal NaN token the browser JSON.parse
        # rejects (silent SSE breakage). None renders honestly as "—".
        self._telemetry.put(json_safe(event))

    def _run(self) -> None:
        try:
            self.status = "running"
            self._emit({"phase": "job_started", "job_id": self.job_id})
            report = self._execute()
            self.report = report
            self.status = "done"
            self._emit({"phase": "job_done", "job_id": self.job_id,
                        "status": report.get("status")})
        except DeepApplyBlocked as exc:
            self.error_type = "DeepApplyBlocked"
            self.error = str(exc)
            traceback.print_exc()
            self.status = "failed"
            self._emit({"phase": "job_failed", "job_id": self.job_id,
                        "error": "DeepApplyBlocked"})
        except Exception as exc:  # noqa: BLE001 -- typed name to client, raw to log
            self.error_type = type(exc).__name__
            self.error = str(exc)
            traceback.print_exc()
            self.status = "failed"
            self._emit({"phase": "job_failed", "job_id": self.job_id,
                        "error": self.error_type})

    def _execute(self) -> Dict[str, Any]:
        from ..deepapply.runner import DeepApplyRunner
        from ..deepapply.trainer import get_backend

        cfg_req = self.config
        receiver_id = cfg_req.get("receiver_id")
        if not receiver_id:
            # Fall back to the source transfer job's bound receiver.
            binding = self.source_pipeline.adapters.get(
                list(self.source_pipeline.adapters)[0]
            ) if self.source_pipeline.adapters else None
            receiver_id = getattr(binding, "receiver_id", None)
        if not receiver_id:
            raise DeepApplyBlocked(
                "no receiver_id given and the source job has no bound receiver"
            )

        backend_name = cfg_req.get("backend", "standard")
        if backend_name not in _BACKENDS:
            raise DeepApplyBlocked(
                "unknown backend '{}'; choose one of {}".format(
                    backend_name, ", ".join(_BACKENDS)
                )
            )

        suite_id = cfg_req.get("suite_id")
        if not suite_id:
            raise DeepApplyBlocked("suite_id is required (the held-out target)")

        # Build the receiver fresh for this run (deep-apply needs HF in-process
        # weights; Ollama holds weights externally and cannot be LoRA-trained).
        try:
            receiver = catalog.build(receiver_id)
        except KeyError as exc:
            raise DeepApplyBlocked(
                "receiver '{}' is not in the catalog: {}".format(receiver_id, exc)
            )

        # Preflight (cheap check FIRST): deep-apply trains a LoRA on weights
        # the process OWNS, so the receiver must be an in-process HF model. An
        # Ollama (external-weights) receiver is refused here BEFORE the heavy
        # torch/transformers/peft import below -- instant, no model load, never
        # a silent fallback mid-train. ``HFCausalConnector`` import is light.
        from ..modules.real import HFCausalConnector

        if not isinstance(receiver, HFCausalConnector):
            raise DeepApplyBlocked(
                "receiver '{}' is not an in-process HF model; deep-apply trains a "
                "LoRA on weights the process owns, so it needs an HF receiver (e.g. "
                "smollm2-360m-hf, qwen2.5-0.5b), not an external-weights (Ollama) "
                "receiver".format(receiver_id)
            )

        # Preflight (heavy check): the [deep] extra (torch/peft) must be
        # installed and the backend must run in this env. This imports the deep
        # stack, so it comes AFTER the cheap HF-type rejection above.
        try:
            backend_obj = get_backend(backend_name)
        except DeepApplyBlocked as exc:
            raise DeepApplyBlocked(
                "deep-apply backend '{}' unavailable: {}".format(backend_name, exc)
            )
        if not backend_obj.supports(receiver):
            raise DeepApplyBlocked(
                "backend '{}' cannot run here (missing the [deep] extra: "
                "pip install -e '.[deep]')".format(backend_name)
            )

        # Packets to train from: only PROMOTED, only for this receiver. The
        # runner re-enforces Gate 1 (no mock, no non-promoted) at intake.
        requested = cfg_req.get("packet_ids") or []
        approved = [
            p for p in self.source_pipeline.store.list("approved")
            if p.target_module == receiver.module_id
        ]
        if requested:
            wanted = set(requested)
            approved = [p for p in approved if p.packet_id in wanted]
            missing = wanted - {p.packet_id for p in approved}
            if missing:
                raise DeepApplyBlocked(
                    "requested packet(s) not approved for '{}': {}".format(
                        receiver.module_id, sorted(missing)
                    )
                )
        if not approved:
            raise DeepApplyBlocked(
                "no approved packets for '{}' to train from; promote skill packets "
                "via a transfer first".format(receiver.module_id)
            )
        packet_ids = [p.packet_id for p in approved]

        # Target suite + regression sweep. The regression sweep = every OTHER
        # benchmark suite, so Gate 2's no_control_movement check has real teeth
        # (a capability the run is NOT targeting must not move). Skipping it
        # would be the loophole this endpoint refuses.
        suite_path = BENCHMARKS / "{}.json".format(suite_id)
        if not suite_path.exists():
            raise DeepApplyBlocked("unknown suite '{}'".format(suite_id))
        target_suite = load_suite(suite_path)
        all_suites = load_all(BENCHMARKS)
        target_cap = target_suite.capability()
        regression_suites = [
            s for s in all_suites.values() if s.capability() != target_cap
        ]

        da_cfg = _coerce_config(backend_name, cfg_req.get("overrides") or {})

        runner = DeepApplyRunner(
            memory_store=self.source_pipeline.store,
            adapter_root=self.workspace / "deepapply",
            audit=self.source_pipeline.audit,
            harness=self.source_pipeline.harness,
            regression_tolerance=self.source_pipeline.evaluator.regression_tolerance,
        )

        report = runner.run(
            receiver, packet_ids, da_cfg, target_suite,
            regression_suites=regression_suites,
            on_progress=self._emit,
        )
        return report.to_dict()

    # -- public views -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Client-facing summary. Carries the typed error NAME only (no raw
        text); the full report when done; status always."""
        out: Dict[str, Any] = {
            "job_id": self.job_id,
            "status": self.status,
            "created_at": self.created_at,
            "config": self.config,
        }
        if self.error_type is not None:
            out["error"] = self.error_type
        if self.report is not None:
            # Backstop: sanitize the served report so a non-finite float (e.g.
            # a NaN loss or parity ratio from a collapsed run) can never 500 the
            # GET endpoint or the listing. Telemetry is already cleaned in
            # _emit; this covers the final report copy.
            out["report"] = json_safe(self.report)
        return out

    def telemetry(self):
        """Generator yielding SSE telemetry frames as the run progresses, then
        a final status frame. Real events from the runner; no fabrication."""
        while True:
            try:
                ev = self._telemetry.get(timeout=0.5)
            except queue.Empty:
                if self.status in ("done", "failed"):
                    break
                continue
            yield "event: telemetry\ndata: {}\n\n".format(
                json.dumps(ev, ensure_ascii=False, default=str)
            )
        # Drain any events queued between the last poll and the status flip.
        while True:
            try:
                ev = self._telemetry.get_nowait()
            except queue.Empty:
                break
            yield "event: telemetry\ndata: {}\n\n".format(
                json.dumps(ev, ensure_ascii=False, default=str)
            )
        yield "event: status\ndata: {}\n\n".format(
            json.dumps({"status": self.status, "errored": self.error_type is not None})
        )


class DeepApplyManager:
    """In-memory registry of deep-apply jobs (session-scoped, like JobManager)."""

    def __init__(self) -> None:
        self._jobs: Dict[str, DeepApplyJob] = {}
        self._lock = threading.Lock()

    def create(
        self,
        config: Dict[str, Any],
        workspace_root: Path,
        source_pipeline: Pipeline,
    ) -> DeepApplyJob:
        job = DeepApplyJob(config, workspace_root, source_pipeline)
        with self._lock:
            self._jobs[job.job_id] = job
        job.start()
        return job

    def get(self, job_id: str) -> DeepApplyJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError("unknown deep-apply job '{}'".format(job_id))
        return job

    def listing(self) -> List[Dict[str, Any]]:
        return [j.to_dict() for j in
                sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)]