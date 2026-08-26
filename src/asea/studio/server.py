"""SILT Studio API server.

    PYTHONPATH=src python3 -m uvicorn asea.studio.server:app --port 8377

Design rules enforced here (see docs/silt_studio design analysis):
  * REAL connectors only -- the catalog cannot construct a mock.
  * The UI replays evidence: every live element is driven by audit events.
  * No endpoint writes to the approved store except through the gate; human
    approval requires a named approver and re-runs the full gate.
  * The playground is read-only with respect to learning.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError, field_validator

log = logging.getLogger("asea.studio")

from ..benchmarks.harness import BenchmarkCase, BenchmarkSuite, load_suite
from ..core.pipeline import Pipeline
from ..core.protocol import (
    CapabilityKey,
    Domain,
    HIGH_RISK_DOMAINS,
    Modality,
    SkillPacket,
)
from . import catalog
from .jobs import BENCHMARKS, JobManager, ROOT
from .deepapply_jobs import DeepApplyManager
from .spring_jobs import SpringManager
from ._jsonsafe import json_safe

STATIC = Path(__file__).resolve().parent / "static"
# README.md at the project root (server.py is at src/asea/studio/server.py,
# so parents[3] is the repo root). Served verbatim at request time so the
# landing's README view is a single source of truth -- it can never drift
# from or contradict the real file.
README_PATH = Path(__file__).resolve().parents[3] / "README.md"
WORKSPACES = ROOT / ".studio"

# Public origins that may bridge to this local engine via the hosted SILT Studio.
# Defaults to the production landing site; comma-separated env override available.
_PUBLIC_STUDIO_ORIGINS = [
    o.strip()
    for o in os.environ.get("SILT_STUDIO_ALLOWED_ORIGINS", "https://silt.inbharat.ai").split(",")
    if o.strip()
]

_LOOPBACK_ORIGIN_REGEX = re.compile(
    r"^https?://("
    r"localhost|"
    r"127\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
    r"\[::1\]|"
    r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
    r"172\.(1[6-9]|2[0-9]|3[0-1])\.\d{1,3}\.\d{1,3}|"
    r"192\.168\.\d{1,3}\.\d{1,3}"
    r")(?::\d+)?$"
)


class LoopbackCORSMiddleware(CORSMiddleware):
    """CORS middleware for the local SILT Studio engine.

    The engine is local-first and must never be exposed on 0.0.0.0 with a
    wildcard allow-origin. This middleware allows the configured public
    SILT landing origin(s) plus loopback / private-network origins, and answers
    the Private-Network-Access preflight header when the browser requests it
    for a allowed origin.

    Credentials are not reflected: the engine owns its own session and the bridge
    does not forward cookies.
    """

    def _is_allowed_origin(self, origin: str) -> bool:
        if not origin:
            return False
        if origin in _PUBLIC_STUDIO_ORIGINS:
            return True
        return bool(_LOOPBACK_ORIGIN_REGEX.match(origin))

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await super().__call__(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        request_origin = headers.get(b"origin", b"").decode("latin-1") or ""
        is_options = scope.get("method") == "OPTIONS"
        wants_private_network = (
            headers.get(b"access-control-request-private-network", b"").decode("latin-1").lower()
            == "true"
        )

        add_private_network_header = (
            is_options and wants_private_network and self._is_allowed_origin(request_origin)
        )

        if add_private_network_header:

            async def wrapped_send(message):
                if message["type"] == "http.response.start":
                    message_headers = list(message.get("headers") or [])
                    message_headers.append((b"access-control-allow-private-network", b"true"))
                    message["headers"] = message_headers
                await send(message)

            await super().__call__(scope, receive, wrapped_send)
            return

        await super().__call__(scope, receive, send)


app = FastAPI(
    title="SILT Studio",
    description="Skill Interchange Layer with Trust-gating -- web platform. "
                "Real connectors only; every number traces to an audit event "
                "or a benchmark case.",
    version="0.1.0",
)

# Local-only CORS: the engine binds to 127.0.0.1 by design; this middleware mirrors
# the same policy at the HTTP layer. No wildcard origins, no public 0.0.0.0 exposure,
# no reflected credentials.
app.add_middleware(
    LoopbackCORSMiddleware,
    allow_origins=_PUBLIC_STUDIO_ORIGINS,
    allow_origin_regex=_LOOPBACK_ORIGIN_REGEX.pattern,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "X-SILT-Bridge-Origin"],
    expose_headers=["Content-Disposition"],
)

manager = JobManager(WORKSPACES)
deepapply_manager = DeepApplyManager()
spring_manager = SpringManager()


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------


class TransferRequest(BaseModel):
    sender: str
    receiver: str
    # Bounded + de-duplicated (adversarial audit 2026-08-13 #19): an unbounded
    # suites list with duplicate stems amplified into thousands of real-model
    # inference runs per request. 8 distinct suites is more than any real
    # transfer needs; duplicates are rejected explicitly below.
    suites: List[str] = Field(min_length=1, max_length=8)
    similarity: str = Field(default="embedding", pattern="^(embedding|lexical)$")
    relevance_floor: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    approver: Optional[str] = None
    description: str = Field(default="", max_length=500)


class ApprovalRequest(BaseModel):
    packet_id: str
    approver: str = Field(min_length=3)


class RollbackRequest(BaseModel):
    token: str


class PlaygroundRequest(BaseModel):
    job_id: str
    module: str
    # Bounded (adversarial audit 2026-08-13 #19): an unbounded prompt was
    # shipped straight to a real LLM backend with no length cap.
    prompt: str = Field(max_length=8000)
    capability: Dict[str, Any]
    use_skills: bool = True


class CatalogEntryRequest(BaseModel):
    """Register an Ollama model at runtime by tag (the 'load an AI model' path).

    Capabilities are derived from the chosen benchmark suite so the new module
    immediately participates in handshakes for that task. Real-only: ``build()``
    still runs its mock-refusal check and the preflight confirms the tag is
    pulled before the entry is left in the catalog.
    """
    module_id: str = Field(min_length=2, pattern=r"^[a-z0-9][a-z0-9-_]*$")
    # Pattern-constrained (adversarial audit 2026-08-13 #40): the tag is
    # reflected into the catalog, the 'requires' hint, and the Ollama model=
    # argument, so it must be a safe Ollama tag, not an arbitrary string.
    ollama_tag: str = Field(min_length=1, max_length=64,
                            pattern=r"^[a-zA-Z0-9._:/-]+$")
    role: str = Field(pattern="^(sender|receiver)$")
    suite_id: str
    description: str = Field(default="", max_length=500)
    think: Optional[bool] = None

    @field_validator("description")
    @classmethod
    def _description_must_not_contain_html_breakout(cls, v):
        """The description is reflected back through ``/api/catalog`` and
        interpolated into ``<option>`` text at several Studio sinks. Reject the
        HTML breakout chars ``<`` / ``>`` at the SOURCE (adversarial review
        2026-08-18: a crafted description could break out of the ``<select>`` and
        inject markup -- the same bug class as the compress-table ``state`` XSS,
        one surface over). Defense in depth: the UI ALSO escapes the description
        at the sink; this guards a future sink that forgets to. Legitimate
        descriptions keep ``&`` / quotes (harmless in element text once the sink
        escapes them); only the breakout chars are refused."""
        if v and ("<" in v or ">" in v):
            raise ValueError(
                "description must not contain '<' or '>' (it is reflected into "
                "the Studio UI)"
            )
        return v


class SkillTestRequest(BaseModel):
    """Read-only accuracy A/B: a receiver on a suite's held-out split, with and
    without its approved skill packet(s). No gate, no store writes."""
    job_id: str
    module: str
    suite_id: str
    packet_id: Optional[str] = None
    similarity: str = Field(default="embedding", pattern="^(embedding|lexical)$")


class DeepApplyRequest(BaseModel):
    """Start a deep-apply run: train a LoRA adapter on an HF receiver from
    packets that already passed Gate 1, then Gate 2 measures the trained
    adapter on held-out data. ``job_id`` is the source transfer job whose
    approved packets to train from. ``receiver_id`` defaults to that job's
    bound receiver but may be any HF catalog entry (deep-apply needs in-process
    weights; Ollama receivers are refused up front). No field weakens a gate;
    overrides are DeepApplyConfig's own documented knobs."""
    job_id: str
    receiver_id: Optional[str] = None
    suite_id: str
    backend: str = Field(default="standard", pattern="^(standard|streamed|zeroforge)$")
    packet_ids: Optional[List[str]] = None
    overrides: Optional[Dict[str, Any]] = None


class SpringRequest(BaseModel):
    """Start a SiltSpring compression certification: load an HF model, measure
    its full-precision reference loss on the chosen held-out suites, then
    certify each quantization state (int8/int4/int2) and stream per-state
    results + peak VRAM. ``module_id`` must be an HF catalog entry (the
    certifier loads its weights); Ollama is refused up front. Memory bound is
    surfaced honestly: certifying needs the full-precision reference resident
    (~14 GB fp16 for 7B -- does not fit 8 GB), so pick a model that fits."""
    module_id: str
    suite_ids: List[str] = Field(min_length=1, max_length=8)
    levels: Optional[List[str]] = None
    tolerance: float = Field(default=0.05, ge=0.0, le=1.0)
    device: str = Field(default="auto", pattern="^(auto|cpu|cuda)$")
    max_len: int = Field(default=96, ge=8, le=512)

    @field_validator("levels")
    @classmethod
    def _levels_must_be_known_quant_states(cls, v):
        """Constrain ``levels`` to the vendor's quantization states (adversarial
        review 2026-08-18): ``levels`` is reflected into the per-state table as
        ``state`` and back to the client, so an arbitrary string was a self-XSS
        vector AND would hand the vendor an unknown level. Only int8/int4/int2
        (the states ``certify_hf_states`` implements) are admitted."""
        if v is None:
            return v
        allowed = ("int8", "int4", "int2")
        bad = [lv for lv in v if lv not in allowed]
        if bad:
            raise ValueError(
                "levels must be one of {} (got {!r})".format(allowed, bad)
            )
        return v


class SuiteAuthorRequest(BaseModel):
    """Author a NEW benchmark suite from the Studio (the 'define a new
    capability' path). A suite is pure data -- ``BenchmarkSuite`` is a pydantic
    model loaded from JSON by ``load_suite`` (there is NO per-suite Python
    loader), so the Studio CAN author one by writing
    ``data/benchmarks/<suite_id>.json``. The filename STEM is the canonical key
    across the Studio (``_suites_by_stem``), so ``suite_id`` is pattern-locked
    to a safe filename and the JSON ``suite_id`` is forced == the stem. A
    transfer needs BOTH an ``extraction`` split (the sender extracts the skill
    from it) AND a ``heldout`` split (the evaluator scores the receiver on it),
    so both are required. A high-risk domain (medical/legal/finance) keeps the
    human-approval gate (driven by ``risk_tier_for_domain``) -- authoring only
    writes data, it never weakens a gate. LOCAL ONLY; patent pending (India)."""
    suite_id: str = Field(min_length=2, max_length=63,
                          pattern=r"^[a-z0-9][a-z0-9_-]*$")
    description: str = Field(default="", max_length=500)
    task_type: str = Field(min_length=1, max_length=64)
    modality: Modality
    domain: Domain = Domain.GENERAL
    language: Optional[str] = Field(default=None, max_length=64)
    cases: List[BenchmarkCase] = Field(min_length=2)

    @field_validator("description")
    @classmethod
    def _description_must_not_contain_html_breakout(cls, v):
        # The description is reflected into the Studio UI, so refuse the HTML
        # tag-OPENING char at the source. Only ``<`` can start a tag -- ``>``
        # alone is inert in HTML text and appears in legitimate copy (e.g.
        # "fr->en", "a => b"), so blocking it would refuse honest input. The UI
        # escapes the value regardless (esc() handles both), so this is defense
        # at the source, not the only layer.
        if v and "<" in v:
            raise ValueError("description must not contain '<' (it is reflected "
                             "into the Studio UI)")
        return v


# --------------------------------------------------------------------------
# Static UI
# --------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/logo.svg", include_in_schema=False)
def logo():
    return FileResponse(STATIC / "logo.svg", media_type="image/svg+xml")


@app.get("/favicon.svg", include_in_schema=False)
def favicon():
    return FileResponse(STATIC / "favicon.svg", media_type="image/svg+xml")


@app.get("/api/readme", include_in_schema=False)
def readme():
    """Serve the repo README.md verbatim, at request time. Read-only; the
    landing renders this so the on-page README is a single source of truth
    (never a hand-copied copy that could drift or become false)."""
    if not README_PATH.is_file():
        raise HTTPException(404, "README.md not found at repo root")
    return PlainTextResponse(
        README_PATH.read_text(encoding="utf-8"),
        media_type="text/markdown; charset=utf-8",
    )


# --------------------------------------------------------------------------
# Catalog & suites
# --------------------------------------------------------------------------


@app.get("/api/health")
def health():
    return {"ok": True, "service": "silt-studio", "mock_free": True}


@app.get("/api/catalog")
def get_catalog():
    return {"modules": catalog.listing(),
            "note": "REAL connectors only. Mocks are not reachable through the Studio."}


def _suites_by_stem():
    return {p.stem: load_suite(p) for p in sorted(BENCHMARKS.glob("*.json"))}


@app.get("/api/suites")
def get_suites():
    suites = _suites_by_stem()
    return {
        suite_id: {
            "task_type": s.task_type,
            "modality": s.modality.value,
            "domain": s.domain.value,
            "language": s.language,
            "splits": s.counts(),
            "high_risk": s.domain in HIGH_RISK_DOMAINS,
            "description": s.description[:240],
        }
        for suite_id, s in suites.items()
    }


def _load_suite_or_404(suite_id: str) -> BenchmarkSuite:
    """Load a suite by STEM (the Studio convention) or 404. Used by the
    support-check + the run-creation support asserts so an unknown suite is a
    structured 404, not a mid-run crash."""
    path = BENCHMARKS / "{}.json".format(suite_id)
    if not path.exists():
        raise HTTPException(404, "unknown suite '{}'".format(suite_id))
    return load_suite(path)


def _capability_support(module_id: str, suite: BenchmarkSuite) -> Dict[str, Any]:
    """Build the module (CHEAP -- ``manifest()`` reads declared capabilities and
    needs no weights; HF connectors load lazily on infer) and report whether it
    supports ``suite.capability()`` plus the capability strings it does support
    (for an honest rejection message). Raises 400 on an unknown module so the
    caller learns that, not a downstream KeyError."""
    try:
        module = catalog.build(module_id)
    except KeyError as exc:
        raise HTTPException(400, "unknown module '{}': {}".format(module_id, exc))
    cap = suite.capability()
    manifest = module.manifest()
    caps = sorted(manifest.capability_set())
    return {
        "id": module_id,
        "supports": bool(manifest.supports(cap)),
        "capability": cap.as_str(),
        "capabilities": caps,
    }


def _assert_support(module_id: str, suite: BenchmarkSuite, role: str) -> None:
    """Hard reject (binding -- 'no half measures'): raise 400 if ``module_id``
    does not support ``suite.capability()``. The message names the capability
    the suite needs and lists what the module actually supports, so a caller
    learns WHY the combo is refused instead of watching a job spawn and fail
    mid-run with an opaque error (or, worse, run and produce garbage scores on
    a capability the model never declared). Called at EVERY run-creation path
    (transfer / skills-test / deep-apply / SiltSpring) before any job starts."""
    info = _capability_support(module_id, suite)
    if not info["supports"]:
        raise HTTPException(
            400,
            "{} '{}' does not support suite '{}' capability '{}' (it supports: "
            "{}). Pick a suite whose capability matches this model, or load a "
            "model that declares this capability via the Skills tab.".format(
                role, module_id, suite.suite_id, info["capability"],
                ", ".join(info["capabilities"]) or "(nothing)",
            )
        )


@app.post("/api/suites")
def author_suite(request: SuiteAuthorRequest):
    """Author a new benchmark suite (define a new capability the Studio can
    target). Writes ``data/benchmarks/<suite_id>.json`` atomically; the suite is
    keyed by filename STEM (the Studio convention), so the JSON ``suite_id`` is
    forced == the stem. Requires >=1 ``extraction`` AND >=1 ``heldout`` case (a
    transfer reads both). A high-risk domain keeps the human-approval gate --
    this only writes data, it never weakens a gate. LOCAL ONLY."""
    sid = request.suite_id
    path = BENCHMARKS / "{}.json".format(sid)
    # Refuse silent overwrite of a capability definition (a suite IS a
    # capability -- overwriting one would silently change what every packet
    # derived from it means). 409, not 200-on-top-of-200.
    if path.exists():
        raise HTTPException(
            409, "suite '{}' already exists (refusing to overwrite a capability "
                 "definition)".format(sid)
        )
    splits = {c.split for c in request.cases}
    if "extraction" not in splits:
        raise HTTPException(
            400, "suite needs at least one 'extraction' case (the sender "
                 "extracts the skill from it)"
        )
    if "heldout" not in splits:
        raise HTTPException(
            400, "suite needs at least one 'heldout' case (the evaluator scores "
                 "the receiver on it)"
        )
    try:
        suite = BenchmarkSuite(
            suite_id=sid,  # forced == stem (the Studio key)
            description=request.description,
            task_type=request.task_type,
            modality=request.modality,
            domain=request.domain,
            language=request.language,
            cases=request.cases,
        )
    except ValidationError as exc:
        raise HTTPException(400, "invalid suite: {}".format(exc))
    BENCHMARKS.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(suite.model_dump(mode="json"), ensure_ascii=False, indent=2)
    # Atomic write: temp file in the same dir + os.replace, so a crash mid-write
    # cannot leave a half-written suite that load_suite would reject (mirrors
    # benchmarks/cache.py's put). The temp name is prefixed with the suite id so
    # a leftover is debuggable, and it lives in BENCHMARKS so os.replace is on
    # one filesystem (atomic).
    fd, tmp = tempfile.mkstemp(prefix=sid + ".", suffix=".tmp",
                               dir=str(BENCHMARKS))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(blob)
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    entry = {
        "task_type": suite.task_type,
        "modality": suite.modality.value,
        "domain": suite.domain.value,
        "language": suite.language,
        "splits": suite.counts(),
        "high_risk": suite.domain in HIGH_RISK_DOMAINS,
        "description": suite.description[:240],
    }
    notice = (
        "High-risk domain: every packet derived from this suite requires a named "
        "human approver and can never auto-promote, regardless of score."
        if suite.domain in HIGH_RISK_DOMAINS else None
    )
    return json_safe({"suite_id": sid, "suite": entry, "high_risk_notice": notice})


@app.get("/api/suites/{suite_id}/support")
def suite_support(suite_id: str, sender: str, receiver: str):
    """Live capability support verdict for a (sender, receiver, suite) triple --
    the UI preview of the SAME ``_assert_support`` check that hard-rejects at
    run creation. Returns whether each model supports the suite's capability
    and what each actually supports, so an unsupported combo is explained, not
    silently run. 404 if the suite is unknown."""
    suite = _load_suite_or_404(suite_id)
    s = _capability_support(sender, suite)
    r = _capability_support(receiver, suite)
    reasons = []
    if not s["supports"]:
        reasons.append(
            "sender '{}' does not support '{}' (it supports: {})".format(
                sender, s["capability"],
                ", ".join(s["capabilities"]) or "(nothing)")
        )
    if not r["supports"]:
        reasons.append(
            "receiver '{}' does not support '{}' (it supports: {})".format(
                receiver, r["capability"],
                ", ".join(r["capabilities"]) or "(nothing)")
        )
    return json_safe({
        "suite_id": suite_id,
        "suite_capability": s["capability"],
        "sender": s,
        "receiver": r,
        "ok": bool(s["supports"] and r["supports"]),
        "reasons": reasons,
    })


# --------------------------------------------------------------------------
# Transfers
# --------------------------------------------------------------------------


def _preflight_model(module_id: str) -> None:
    """Fail fast at POST time if a connector's backing model is missing.

    Only connectors that expose a cheap ``health()`` (e.g. Ollama's
    ``/api/tags``) are probed. Connectors without ``health()`` are skipped, so
    this never eagerly loads weights -- HF connectors build lazily. A missing
    model is reported as a 400 with the connector's own pull/install hint,
    rather than letting a job start and fail 30s later with an opaque error.
    """
    try:
        adapter = catalog.build(module_id)
    except Exception as exc:  # pragma: no cover - surfaced to the caller
        raise HTTPException(400, "cannot build '{}': {}".format(module_id, exc))
    health = getattr(adapter, "health", None)
    if not callable(health):
        return
    try:
        report = health()
    except Exception as exc:  # pragma: no cover - degrade gracefully
        raise HTTPException(
            400, "health check for '{}' failed: {}".format(module_id, exc)
        )
    if isinstance(report, dict) and report.get("model_present") is False:
        hint = report.get("hint") or "install its backing model"
        raise HTTPException(
            400,
            "module '{}' backing model '{}' is not installed. {}".format(
                module_id, report.get("model"), hint
            ),
        )


@app.post("/api/transfers")
def create_transfer(request: TransferRequest):
    if request.sender not in {m["id"] for m in catalog.listing()}:
        raise HTTPException(400, "unknown sender '{}'".format(request.sender))
    if request.receiver not in {m["id"] for m in catalog.listing()}:
        raise HTTPException(400, "unknown receiver '{}'".format(request.receiver))
    if request.sender == request.receiver:
        raise HTTPException(400, "sender and receiver must differ (self-transfer "
                                 "is the model-collapse loop)")
    known = {p.stem for p in BENCHMARKS.glob("*.json")}
    unknown = [s for s in request.suites if s not in known]
    if unknown:
        raise HTTPException(400, "unknown suites: {}".format(unknown))
    # Reject duplicate stems (adversarial audit 2026-08-13 #19): the real
    # amplification vector was the same valid stem repeated thousands of times,
    # each triggering real sender+receiver inference. The list is already
    # length-bounded by the schema; duplicates are refused explicitly rather
    # than silently de-duplicated.
    if len(set(request.suites)) != len(request.suites):
        raise HTTPException(400, "duplicate suites: {}".format(request.suites))
    # Hard reject with an explanation when a model does not support a suite's
    # capability (binding -- "if it doesn't support the trainer or learner AI,
    # it rejects with explanation"): BEFORE any job is spawned and BEFORE the
    # heavier _preflight_model network probe, so an unsupported combo is a 400
    # naming the capability + what each model supports, not a mid-run crash or
    # garbage scores on a capability the model never declared. Checked for BOTH
    # models against EVERY requested suite.
    for stem in request.suites:
        suite = _load_suite_or_404(stem)
        _assert_support(request.sender, suite, "sender (trainer)")
        _assert_support(request.receiver, suite, "receiver (learner)")
    # Fail fast before spawning a job: refuse transfers whose backing model
    # is not present (e.g. an Ollama tag that was never pulled). A job that
    # starts against a missing model would otherwise fail mid-run with a less
    # obvious error and nothing to show in the UI.
    _preflight_model(request.sender)
    _preflight_model(request.receiver)
    job = manager.create(request.model_dump())
    return job.to_dict()


@app.get("/api/transfers")
def list_transfers():
    return {"jobs": manager.listing()}


@app.get("/api/transfers/{job_id}")
def get_transfer(job_id: str):
    try:
        job = manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    payload = job.to_dict()
    # Sanitize the served report: a transfer evaluation can produce a
    # non-finite score (a metric plugin / embedding similarity that returns
    # nan for a degenerate case -> min(1.0, nan) propagates nan through
    # SuiteResult.score -> negotiation.receiver_score / evaluations[].scores).
    # FastAPI's JSONResponse serializes with allow_nan=False, so a single NaN
    # would 500 this endpoint (the exact failure the spring fix addressed).
    # The report's scores are a raw dict (not model_dump_json), so pydantic's
    # nan->null coercion does NOT apply here -- json_safe is the backstop.
    # (adversarial review 2026-08-19, loophole #1.)
    payload["report"] = json_safe(job.report)
    return payload


@app.get("/api/transfers/{job_id}/events")
def stream_events(job_id: str):
    try:
        manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    return StreamingResponse(
        manager.stream(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------------------------------
# Packets, approval, rollback  (all through the job's own pipeline/gate)
# --------------------------------------------------------------------------


def _pipeline_for(job_id: str) -> Pipeline:
    try:
        job = manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    if job.pipeline is None:
        raise HTTPException(409, "job has not initialised its pipeline yet")
    return job.pipeline


@app.get("/api/transfers/{job_id}/packets")
def list_packets(job_id: str):
    pipeline = _pipeline_for(job_id)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for bucket in ("approved", "candidate", "rejected"):
        out[bucket] = [
            json.loads(p.model_dump_json()) for p in pipeline.store.list(bucket)
        ]
    out["snapshots"] = pipeline.rollback.list_snapshots()
    return out


@app.post("/api/transfers/{job_id}/approve")
def approve(job_id: str, request: ApprovalRequest):
    """Named human approval. Re-runs the FULL gate -- approval satisfies exactly
    one check and cannot waive the others."""
    pipeline = _pipeline_for(job_id)
    try:
        decision = pipeline.approve_pending(request.packet_id, approver=request.approver)
    except Exception as exc:
        # Log the full detail server-side; return only a generic message + a
        # correlation ref to the caller (adversarial audit 2026-08-13 #20:
        # returning str(exc) leaked pipeline/store internals to any caller).
        ref = uuid.uuid4().hex[:8]
        log.exception("approve failed for job %s packet %s (ref %s)",
                      job_id, request.packet_id, ref)
        raise HTTPException(400, "approval failed (ref {})".format(ref))
    return decision


@app.post("/api/transfers/{job_id}/rollback")
def rollback(job_id: str, request: RollbackRequest):
    pipeline = _pipeline_for(job_id)
    try:
        return pipeline.rollback_to(request.token, actor="studio-user")
    except Exception as exc:
        ref = uuid.uuid4().hex[:8]
        log.exception("rollback failed for job %s (ref %s)", job_id, ref)
        raise HTTPException(400, "rollback failed (ref {})".format(ref))


@app.get("/api/transfers/{job_id}/audit")
def audit(job_id: str):
    pipeline = _pipeline_for(job_id)
    # Audit entries are re-parsed from on-disk JSONL with json.loads, which
    # accepts the literal ``NaN`` token (allow_nan=True on parse) and returns a
    # Python ``nan``. The log was written with ``json.dumps(..., default=str)``
    # whose ``default`` does NOT rescue NaN (it only fires for non-JSON
    # *types*, and float is a recognized type), so a NaN-bearing detail (e.g. an
    # ``evaluated`` event carrying evaluation.to_dict()) round-trips through
    # disk as ``NaN`` and comes back as a nan float -- FastAPI's allow_nan=False
    # encoder would then 500 this endpoint. json_safe is the backstop.
    # (adversarial review 2026-08-19, loophole #2.)
    return {"integrity": pipeline.audit.verify(),
            "entries": json_safe(pipeline.audit.entries())}


@app.get("/api/transfers/{job_id}/export")
def export_bundle(job_id: str, base_model: Optional[str] = None,
                   include_mock: bool = False):
    """Download the "trained model": a zip of the job's approved skill packet(s).

    Read-only with respect to the memory store -- it emits packets the gate has
    ALREADY promoted, nothing more. Refuses mock-derived and non-promoted
    packets by default (same guard as ``export_dataset``). SILT trains no
    weights; the bundle is the skill packet(s) the receiver conditions on at
    inference time, plus -- when ``base_model`` is supplied -- an L4/L5 dataset
    and a NOT_EXECUTED training job spec for an external trainer.
    """
    from tempfile import TemporaryDirectory

    from fastapi.responses import Response

    from ..distill.export import export_artifact_bundle

    pipeline = _pipeline_for(job_id)
    approved = pipeline.store.list("approved")
    audit_path = pipeline.workspace / "audit" / "audit.jsonl"
    filename = "{}_skill_bundle.zip".format(job_id)
    # Build the bundle in a temp dir, then read the bytes into memory so the
    # temp dir can be cleaned up immediately (a FileResponse would read lazily,
    # after the temp dir is gone). Bundles are small (JSON packets), so this is
    # cheap and avoids both a temp-file lifetime race and a leaked temp dir.
    with TemporaryDirectory(prefix="silt-export-") as tmp:
        zip_path = export_artifact_bundle(
            approved, Path(tmp), name="{}_skill_bundle".format(job_id),
            base_model=base_model, include_mock=include_mock,
            audit_path=audit_path, policy=pipeline.gate.policy,
        )
        data = zip_path.read_bytes()
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment: filename="{}"'.format(filename)},
    )


# --------------------------------------------------------------------------
# Deep-apply (weights-mode training) -- the Studio surface for "train an
# adapter on a low-GPU system and watch it happen". A run trains a LoRA on an
# HF receiver from packets that already passed Gate 1, then Gate 2 measures
# the trained adapter on held-out data. The telemetry endpoint streams the
# runner's REAL phase + per-step events as SSE (loss curve, step gauge, parity,
# gate2 verdict). No gate is weakened: this only calls the same
# DeepApplyRunner the CLI/tests use, and drains its on_progress stream.
# --------------------------------------------------------------------------


@app.post("/api/deepapply")
def start_deep_apply(request: DeepApplyRequest):
    """Start a deep-apply run. ``job_id`` is the source transfer job (whose
    approved packets to train from); that job must have initialised its
    pipeline. Returns the deep-apply job id immediately; poll
    ``/api/deepapply/{id}`` for status or ``/api/deepapply/{id}/telemetry`` for
    the live SSE stream. The receiver must be an HF in-process connector
    (Ollama is refused with 400 -- it cannot be LoRA-trained)."""
    pipeline = _pipeline_for(request.job_id)
    # Hard reject with an explanation before the train job spawns: the receiver
    # must support the suite's capability (a train that targets a capability the
    # model never declared would tune it on a task it cannot score). Same
    # _assert_support the transfer path uses, against the receiver + the target
    # suite, so the reject message is identical in shape.
    if request.suite_id:
        suite = _load_suite_or_404(request.suite_id)
        _assert_support(request.receiver_id, suite, "receiver (learner)")
    job = deepapply_manager.create(
        {
            "source_job_id": request.job_id,
            "receiver_id": request.receiver_id,
            "suite_id": request.suite_id,
            "backend": request.backend,
            "packet_ids": request.packet_ids,
            "overrides": request.overrides,
        },
        WORKSPACES,
        pipeline,
    )
    return {"deepapply_id": job.job_id, "status": job.status,
            "source_job_id": request.job_id}


@app.get("/api/deepapply")
def list_deep_apply():
    """List deep-apply jobs (newest first). Client-facing summaries carry the
    typed error NAME only, never raw exception text."""
    return {"jobs": deepapply_manager.listing()}


@app.get("/api/deepapply/{job_id}")
def get_deep_apply(job_id: str):
    try:
        job = deepapply_manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    return job.to_dict()


@app.get("/api/deepapply/{job_id}/telemetry")
def stream_deep_apply_telemetry(job_id: str):
    """SSE stream of a deep-apply run's REAL telemetry: phase events
    (dataset_built, backend_selected, train_started, train_completed,
    gate2_evaluated, gate2_decision, done) and per-step ``train_step`` events
    carrying the live loss (streamed/standard backends; zeroforge emits
    phase-level only -- its per-step loop is in vendored siltstream). Closes
    with a ``status`` frame. No fabricated numbers."""
    try:
        job = deepapply_manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    return StreamingResponse(
        job.telemetry(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------------------------------
# SiltSpring compression certification -- the Studio surface for "compress a
# big model to transfer skills and verify the compression does not degrade
# them". Loads an HF model, measures the full-precision reference on the chosen
# held-out suites, certifies each quantization state (int8/int4/int2), and
# streams per-state results + peak VRAM as SSE. The compression is REAL
# (vendored certify_hf_states) and measured against the same held-out suites
# the gates use -- not a fake "% smaller" number.
# --------------------------------------------------------------------------


@app.post("/api/spring")
def start_spring(request: SpringRequest):
    """Start a SiltSpring certification run. ``module_id`` must be an HF catalog
    entry (the certifier loads its weights); Ollama is refused with a typed
    error up front. Returns the spring job id immediately; poll
    ``/api/spring/{id}`` for status or ``/api/spring/{id}/telemetry`` for the
    live per-state SSE stream."""
    # Hard reject with an explanation before the compression job spawns: the
    # model must support each target suite's capability (certifying quantization
    # against a capability the model never declared would measure compression
    # degradation on a task it cannot score -- a meaningless number, and exactly
    # the "silently run garbage" this feature refuses). The spring job also
    # re-asserts HF-type internally; this is the capability-side gate, checked
    # here so the 400 carries the capability + supported-list explanation and no
    # telemetry job is started.
    for stem in request.suite_ids or []:
        suite = _load_suite_or_404(stem)
        _assert_support(request.module_id, suite, "model")
    job = spring_manager.create(
        {
            "module_id": request.module_id,
            "suite_ids": request.suite_ids,
            "levels": request.levels,
            "tolerance": request.tolerance,
            "device": request.device,
            "max_len": request.max_len,
        },
        WORKSPACES,
    )
    return {"spring_id": job.job_id, "status": job.status,
            "module_id": request.module_id}


@app.get("/api/spring")
def list_spring():
    """List SiltSpring certification jobs (newest first). Client-facing
    summaries carry the typed error NAME only, never raw exception text."""
    return {"jobs": spring_manager.listing()}


@app.get("/api/spring/{job_id}")
def get_spring(job_id: str):
    try:
        job = spring_manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    return job.to_dict()


@app.get("/api/spring/{job_id}/telemetry")
def stream_spring_telemetry(job_id: str):
    """SSE stream of a SiltSpring run's REAL per-state telemetry:
    model_loading, certify_started, one ``spring_state`` per quantization
    level (bytes_packed, per-skill loss + degradation, certified/revoked
    skills), then a closing ``status`` frame. No fabricated numbers."""
    try:
        job = spring_manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    return StreamingResponse(
        job.telemetry(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------------------------------
# Skills library, runtime model loading, test-before-download
#   All three are read-only w.r.t. the memory store and the gate. The library
#   globs approved packets across jobs; the catalog POST injects a real Ollama
#   entry; the test endpoint runs the same harness A/B the gate's evaluator
#   used, but never calls Evaluator.evaluate (which mutates the packet) and
#   never calls the gate.
# --------------------------------------------------------------------------


@app.get("/api/skills")
def list_skills():
    """Cross-job library of approved skill packets SILT has validated.

    Read-only glob over ``.studio/*/memory/approved/*.json`` -- reuses the same
    ``SkillPacket.model_validate`` path the memory store uses, writes nothing,
    and never touches a live job pipeline. Lets the dashboard list every
    successfully-tested technique and load it for a new transfer or test.
    """
    rows = []
    for path in sorted(WORKSPACES.glob("*/memory/approved/*.json")):
        try:
            pkt = SkillPacket.model_validate(json.loads(path.read_text("utf-8")))
        except Exception:
            # Skip unreadable files rather than fail the whole list -- the
            # library must stay robust to a half-written or corrupt packet.
            continue
        rows.append({
            "job_id": path.parts[-4],  # .studio/<job_id>/memory/approved/<id>.json
            "packet_id": pkt.packet_id,
            "created_at": pkt.provenance.created_at,
            "capability": pkt.sender_capability.as_str(),
            "task_type": pkt.task_type,
            "domain": pkt.domain.value,
            "language": pkt.language,
            "target_module": pkt.target_module,
            "packet_type": pkt.packet_type.value if pkt.packet_type else None,
            "learning_level": pkt.learning_level.value,
            "evaluator_score": pkt.evaluator_score,
            "promotion_status": pkt.promotion_status.value,
            "human_approved_by": pkt.human_approved_by,
            "is_mock": pkt.provenance.is_mock,
            "synthetic_depth": pkt.provenance.synthetic_depth,
            "provenance_chain": pkt.provenance.chain,
        })
    # Defense in depth (adversarial review 2026-08-19, loophole #7): a packet
    # persisted to approved/ with a non-finite evaluator_score would read back
    # as a nan float (json.loads accepts the NaN token) and 500 this listing
    # via FastAPI's allow_nan=False encoder. The gate normally rejects a
    # NaN-scored packet (nan >= min_evaluator_score is False), but a packet
    # that reached approved/ by another path would crash the whole library
    # view -- json_safe turns the non-finite value to None instead.
    return {"skills": json_safe(rows), "count": len(rows)}


@app.post("/api/catalog")
def add_catalog_entry(request: CatalogEntryRequest):
    """Load an AI model into the Studio at runtime by Ollama tag.

    Adds a real ``CATALOG`` entry whose factory is ``catalog._ollama(tag, ...)``
    with capabilities derived from the chosen suite. ``build()`` still enforces
    its mock-refusal check (which passes for OllamaConnector) and the preflight
    probes ``OllamaConnector.health()`` so a tag that was never pulled fails
    here with the ``ollama pull <tag>`` hint, not 30s into a run. Idempotent --
    re-adding a runtime id overwrites it; built-in ids are protected.
    """
    known = {p.stem for p in BENCHMARKS.glob("*.json")}
    if request.suite_id not in known:
        raise HTTPException(400, "unknown suite '{}'".format(request.suite_id))
    suite = load_suite(BENCHMARKS / "{}.json".format(request.suite_id))
    cap = suite.capability()
    factory = catalog._ollama(
        request.ollama_tag, roles=[request.role],
        capabilities=[cap], think=request.think,
    )
    # Hold the catalog lock around the builtin guard + CATALOG mutation + cache
    # invalidation (adversarial audit 2026-08-13 #39/#40): the read-check-write
    # must be atomic so a builtin entry cannot be overwritten by a concurrent
    # caller, and a concurrent GET /api/catalog / build() cannot see a torn
    # dict. The factory is built outside the lock (above) so a slow Ollama
    # probe does not block other catalog reads.
    with catalog._lock:
        if (
            request.module_id in catalog.CATALOG
            and catalog.CATALOG[request.module_id].get("builtin")
        ):
            raise HTTPException(
                400, "cannot overwrite a built-in module '{}'".format(request.module_id)
            )
        catalog.CATALOG[request.module_id] = {
            "factory": factory,
            "roles": [request.role],
            "description": request.description or "runtime-added ollama model",
            "requires": "ollama serve + ollama pull {}".format(request.ollama_tag),
        }
        catalog._cache.pop(request.module_id, None)
    # Preflight the tag now -- surfaces the pull hint if the model is absent.
    # This is DELIBERATELY after registration: a failed preflight leaves the
    # entry in CATALOG so the user can `ollama pull <tag>` and retry the
    # transfer without re-POSTing (test_catalog_post_adds_entry... asserts
    # this). The entry is not "broken" -- catalog.build() constructs the
    # OllamaConnector cheaply (no weights); the failure surfaces at run time
    # with the pull hint.
    _preflight_model(request.module_id)
    return {"module_id": request.module_id, "roles": [request.role],
            "ollama_tag": request.ollama_tag, "capability": cap.as_str(),
            "health": "ok"}


@app.post("/api/skills/test")
def test_skill(request: SkillTestRequest):
    """Test a trained model's accuracy BEFORE downloading.

    Runs the receiver on the suite's held-out split with and without its
    approved skill packet(s), using the SAME ``BenchmarkHarness.run`` path the
    gate's evaluator used -- but calls the harness directly, never
    ``Evaluator.evaluate`` (which mutates ``packet.promotion_status`` and
    ``packet.scores``) and never the gate. Read-only with respect to the memory
    store: there is no path from here into the approved set. Returns per-case
    accuracy so the user can decide whether the skill is worth downloading.
    """
    pipeline = _pipeline_for(request.job_id)
    known = {p.stem for p in BENCHMARKS.glob("*.json")}
    if request.suite_id not in known:
        raise HTTPException(400, "unknown suite '{}'".format(request.suite_id))
    try:
        module = catalog.build(request.module)
    except KeyError as exc:
        raise HTTPException(400, str(exc))
    suite = load_suite(BENCHMARKS / "{}.json".format(request.suite_id))
    # Hard reject with an explanation before running the receiver: a skill test
    # against a capability the model never declared would score it on a task it
    # cannot do, yielding a misleading "0% accuracy" that looks like a broken
    # skill rather than a capability mismatch. Same _assert_support as the other
    # run paths, so the reject message names the capability + what the model
    # supports.
    _assert_support(request.module, suite, "model")
    if not [c for c in suite.cases if c.split == "heldout"]:
        raise HTTPException(
            400, "suite '{}' has no held-out cases".format(request.suite_id)
        )

    from ..benchmarks.harness import BenchmarkHarness
    from ..core.plugins import default_registry
    from .jobs import _similarity
    harness = BenchmarkHarness(
        plugins=default_registry(), similarity=_similarity(request.similarity)
    )

    # Redacted payloads the receiver is permitted to see. Optionally one packet.
    all_skills = pipeline.store.approved_skills(module.module_id)
    if request.packet_id:
        skills = [s for s in all_skills if s.get("packet_id") == request.packet_id]
        if not skills:
            raise HTTPException(
                400,
                "no approved packet '{}' for '{}'".format(
                    request.packet_id, module.module_id
                ),
            )
    else:
        skills = all_skills

    baseline = harness.run(module, suite, split="heldout")
    candidate = harness.run(module, suite, split="heldout", skills=skills)

    # Reuse EvaluationReport.case_diff's logic (evaluator.py) over the two
    # SuiteResults, without constructing an EvaluationReport (no mutation).
    by_id = {c.case_id: c for c in candidate.case_results}
    cases = []
    for base in baseline.case_results:
        cand = by_id.get(base.case_id)
        if cand is None:
            continue
        cases.append({
            "case_id": base.case_id,
            "expected": base.expected,
            "baseline_output": base.actual,
            "candidate_output": cand.actual,
            "baseline_score": round(base.score, 4),
            "candidate_score": round(cand.score, 4),
            "delta": round(cand.score - base.score, 4),
            "regressed": cand.score < base.score,
        })
    high_risk = suite.domain in HIGH_RISK_DOMAINS
    # Sanitize: this endpoint runs the live harness and returns raw
    # round(score, 4) values. A metric plugin / embedding similarity that
    # yields nan for a degenerate case propagates through SuiteResult.score
    # into baseline/candidate/improvement and the per-case deltas; FastAPI's
    # allow_nan=False encoder would 500 the whole test. json_safe turns a
    # non-finite score to None (the UI renders "—") rather than crashing.
    # (adversarial review 2026-08-19, loophole #4.)
    return json_safe({
        "job_id": request.job_id,
        "module": module.module_id,
        "is_mock": module.is_mock,
        "suite_id": request.suite_id,
        "skills_active": len(skills),
        "high_risk": high_risk,
        "baseline": {
            "score": round(baseline.score, 4),
            "task_success": round(baseline.task_success, 4),
            "case_count": len(baseline.case_results),
        },
        "candidate": {
            "score": round(candidate.score, 4),
            "task_success": round(candidate.task_success, 4),
            "case_count": len(candidate.case_results),
        },
        "improvement": round(candidate.score - baseline.score, 4),
        "similarity_is_semantic": candidate.similarity_is_semantic,
        "cases": cases,
        "disclaimer": (
            "High-risk domain: outputs are not professional advice."
            if high_risk else None
        ),
    })


# --------------------------------------------------------------------------
# Playground -- read-only w.r.t. learning
# --------------------------------------------------------------------------


@app.post("/api/playground")
def playground(request: PlaygroundRequest):
    """Run one prompt against a module, optionally conditioned on the job's
    APPROVED packets. Uses the identical infer_with_skills path the evaluator
    measured. There is deliberately no path from here into the memory store."""
    pipeline = _pipeline_for(request.job_id)
    try:
        module = catalog.build(request.module)
    except KeyError as exc:
        raise HTTPException(400, str(exc))

    capability = CapabilityKey(
        task_type=request.capability["task_type"],
        modality=Modality(request.capability["modality"]),
        domain=Domain(request.capability.get("domain", "general")),
        language=request.capability.get("language"),
    )

    skills = (
        pipeline.store.approved_skills(module.module_id)
        if request.use_skills else []
    )
    if request.use_skills and skills:
        output = module.infer_with_skills(capability, request.prompt, skills)
    else:
        output = module.infer(capability, request.prompt)

    return {
        "module": module.module_id,
        "is_mock": module.is_mock,          # always False by catalog construction
        "skills_active": len(skills),
        "high_risk": capability.domain in HIGH_RISK_DOMAINS,
        "output": output,
        "disclaimer": (
            "High-risk domain: output is not professional advice."
            if capability.domain in HIGH_RISK_DOMAINS else None
        ),
    }
