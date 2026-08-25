"""Transfer job manager.

A transfer against real models takes minutes, so runs execute on worker
threads while the API stays responsive. Each job gets its own workspace
directory; the job's audit log is the single source of truth that the SSE
stream tails -- the Studio never invents state, it replays evidence.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..benchmarks.harness import load_suite
from ..core.gap import GapEngine
from ..core.pipeline import Pipeline
from ..core.plugins import default_registry
from ..benchmarks.harness import BenchmarkHarness
from ..evaluator.evaluator import Evaluator
from ..filters.relevance import RelevanceFilter, RelevancePolicy
from . import catalog
from ._jsonsafe import json_safe

ROOT = Path(__file__).resolve().parent.parent.parent.parent
BENCHMARKS = ROOT / "data" / "benchmarks"


def _similarity(kind: str):
    if kind == "embedding":
        from ..modules.real import best_available_similarity

        return best_available_similarity(quiet=True)
    from ..evaluator.similarity import LexicalSimilarity

    return LexicalSimilarity()


class TransferJob:
    def __init__(self, config: Dict[str, Any], workspace_root: Path) -> None:
        self.job_id = uuid.uuid4().hex[:12]
        self.config = config
        self.workspace = Path(workspace_root) / self.job_id
        self.status = "queued"        # queued | loading | running | done | failed
        self.error: Optional[str] = None
        self.report: Optional[Dict[str, Any]] = None
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.pipeline: Optional[Pipeline] = None
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self.status = "loading"
            sender = catalog.build(self.config["sender"])
            receiver = catalog.build(self.config["receiver"])

            similarity = _similarity(self.config.get("similarity", "embedding"))
            plugins = default_registry()
            harness = BenchmarkHarness(plugins=plugins, similarity=similarity)

            relevance_kwargs: Dict[str, Any] = {}
            if self.config.get("relevance_floor") is not None:
                relevance_kwargs["sender_correctness_floor"] = float(
                    self.config["relevance_floor"]
                )
            relevance = RelevanceFilter(
                RelevancePolicy(**relevance_kwargs), similarity=similarity
            )

            pipeline = Pipeline(
                workspace=self.workspace,
                plugins=plugins,
                harness=harness,
                gap_engine=GapEngine(harness=harness),
                relevance=relevance,
                evaluator=Evaluator(harness=harness),
                # DEFAULT strict gate. Nothing in the catalog is a mock, so the
                # anti-mock check never needs relaxing -- and is not relaxed.
            )
            self.pipeline = pipeline
            pipeline.register_module(sender)
            pipeline.register_module(receiver)
            adapter_id = "{}-to-{}".format(sender.module_id, receiver.module_id)
            pipeline.bind_adapter(
                adapter_id, sender.module_id, receiver.module_id,
                description=self.config.get("description", "SILT Studio transfer"),
            )

            suites = [
                load_suite(BENCHMARKS / "{}.json".format(name))
                for name in self.config["suites"]
            ]

            self.status = "running"
            report = pipeline.run(
                adapter_id,
                suites=suites,
                human_approver=self.config.get("approver") or None,
                actor="studio:{}".format(self.job_id),
            )
            self.report = report.to_dict()
            self.status = "done"
        except Exception as exc:  # pragma: no cover - surfaced to the UI
            self.error = "{}: {}".format(type(exc).__name__, exc)
            self.status = "failed"
            traceback.print_exc()

    # -- evidence access ------------------------------------------------------

    def audit_path(self) -> Path:
        return self.workspace / "audit" / "audit.jsonl"

    def events(self, after_index: int = -1) -> List[Dict[str, Any]]:
        path = self.audit_path()
        if not path.exists():
            return []
        out = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                # A torn/partial write (process crash mid-append, disk hiccup)
                # leaves an unparseable line; the audit logger acknowledges this
                # as a real scenario. Skip it rather than letting json.loads
                # raise through the generator and kill the SSE stream forever.
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("index", 0) > after_index:
                    out.append(entry)
        return out

    def to_dict(self) -> Dict[str, Any]:
        # Do NOT surface the raw error string or the absolute workspace path to
        # callers (adversarial audit 2026-08-13 #20): both are leaked
        # unauthenticated via GET /api/transfers and the listing endpoint. The
        # full error is kept on the job object for the operator's own SSE stream
        # (see stream()) and server-side logs; the API exposes only whether the
        # job errored, not why or where.
        return {
            "job_id": self.job_id,
            "status": self.status,
            "errored": self.error is not None,
            "config": self.config,
            "created_at": self.created_at,
        }


class JobManager:
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = Path(workspace_root)
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.jobs: Dict[str, TransferJob] = {}
        self._lock = threading.Lock()

    def create(self, config: Dict[str, Any]) -> TransferJob:
        job = TransferJob(config, self.workspace_root)
        with self._lock:
            self.jobs[job.job_id] = job
        job.start()
        return job

    def get(self, job_id: str) -> TransferJob:
        job = self.jobs.get(job_id)
        if job is None:
            raise KeyError("unknown job '{}'".format(job_id))
        return job

    def listing(self) -> List[Dict[str, Any]]:
        return [job.to_dict() for job in
                sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)]

    def stream(self, job_id: str, poll_seconds: float = 0.5):
        """Generator yielding SSE frames: audit events as they are appended,
        then a final status frame when the job leaves the running states."""
        job = self.get(job_id)
        last_index = -1
        while True:
            for entry in job.events(after_index=last_index):
                last_index = entry.get("index", last_index)
                # Sanitize BEFORE serialize: json.dumps defaults to
                # allow_nan=True, so a NaN-bearing audit detail (an ``evaluated``
                # event whose evaluation.to_dict() carried a non-finite score)
                # would serialize to the literal ``NaN`` token, which the
                # browser's JSON.parse rejects per the ECMAScript spec -- the
                # frame is silently dropped and the EventSource client breaks.
                # ``default=str`` does NOT rescue NaN (it only fires for
                # non-JSON types; float is recognized). json_safe turns the
                # non-finite value to None at the boundary. (adversarial review
                # 2026-08-19, loophole #3 -- the same silent-SSE breakage the
                # spring/deepapply _emit fix was written to prevent.)
                yield "event: audit\ndata: {}\n\n".format(
                    json.dumps(json_safe(entry), ensure_ascii=False, default=str)
                )
            if job.status in ("done", "failed"):
                # flush any trailing events written between poll and status flip
                for entry in job.events(after_index=last_index):
                    last_index = entry.get("index", last_index)
                    yield "event: audit\ndata: {}\n\n".format(
                        json.dumps(json_safe(entry), ensure_ascii=False, default=str)
                    )
                # Do NOT surface the raw error string here. ``to_dict`` redacts
                # it (the API exposes only whether the job errored, not why); the
                # SSE stream must match that policy -- index.html renders this
                # frame into the DOM, and ``job.error`` carries the raw exception
                # text (paths, deep-dependency messages). The full error stays
                # server-side (already logged via traceback.print_exc in _run).
                yield "event: status\ndata: {}\n\n".format(
                    json.dumps({"status": job.status, "errored": job.error is not None})
                )
                return
            time.sleep(poll_seconds)
