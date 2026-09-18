"""Closed schemas for capability-build (v1).

All models are :class:`asea.compose.schema.StrictModel` (extra=forbid,
strict types, frozen, no inf/nan) -- the same closed-record discipline as
``asea.validation.schema``. Schemas carry a ``schema`` string tag:

  * ``silt.capability_spec.v1``      -- machine-readable capability contract
  * ``silt.capability_trace.v1``    -- one teacher observation (either class)
  * ``silt.capability_footprint.v1`` -- components associated+important to a
                                         capability under a defined workload
  * ``silt.capability_receipt.v1``   -- immutable signed run report

Honesty contracts encoded here (binding):

  * Spec thresholds (retention ratio, control regression) are OPERATOR
    CONFIGURATION, never guarantees; the models refuse nothing on their
    behalf beyond type/range sanity.
  * ``CapabilityTrace`` enforces evidence-class separation at schema level:
    a ``behavioural_remote`` trace must NOT carry internal (router/expert)
    observations and an ``internal_open_weight`` trace must NOT carry a
    behavioural transcript. The two classes are never mixed.
  * ``CapabilityFootprint`` REQUIRES a non-empty ``limitations`` list: a
    footprint without stated limitations is invalid on its face. Routing
    enrichment is correlation; only entries under ``causal_interventions``
    carry intervention-measured causal evidence.
  * Receipt metrics are optional; when absent at materialisation time they
    are rendered as the literal token ``NOT_MEASURED`` -- never estimated.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union  # noqa: F401 (Union kept for future typed metrics)

from pydantic import Field, model_validator

from asea.compose.schema import StrictModel

from . import (
    BLOCKED_RESOURCE,
    NOT_MEASURED,
    TRACE_CLASS_BEHAVIOURAL,
    TRACE_CLASS_INTERNAL,
)

Hash = str  # validated by pattern below

CAPABILITY_SPEC_SCHEMA = "silt.capability_spec.v1"
CAPABILITY_TRACE_SCHEMA = "silt.capability_trace.v1"
CAPABILITY_FOOTPRINT_SCHEMA = "silt.capability_footprint.v1"
CAPABILITY_RECEIPT_SCHEMA = "silt.capability_receipt.v1"

TraceClass = Literal["behavioural_remote", "internal_open_weight"]
Group = Literal["target", "control"]


# ---------------------------------------------------------------------------
# CapabilitySpec -- silt.capability_spec.v1
# ---------------------------------------------------------------------------

class TeacherRef(StrictModel):
    """Exact teacher pin. ``revision`` must identify an exact checkpoint
    (HF commit SHA, provider model revision), never a moving tag alone."""
    provider: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=256)
    revision: str = Field(min_length=1, max_length=256)
    access: TraceClass = TRACE_CLASS_BEHAVIOURAL


class HardwareBudget(StrictModel):
    max_model_storage_bytes: Optional[int] = Field(default=None, ge=0)
    max_peak_ram_bytes: Optional[int] = Field(default=None, ge=0)
    max_peak_vram_bytes: Optional[int] = Field(default=None, ge=0)


class EvaluationSplits(StrictModel):
    training: str = Field(min_length=1, max_length=256)
    development: str = Field(min_length=1, max_length=256)
    heldout: str = Field(min_length=1, max_length=256)
    final: str = Field(min_length=1, max_length=256)


class CapabilitySpec(StrictModel):
    schema_version: Literal[1] = 1
    #: Serialized under the key ``"schema"`` (alias); the Python field is
    #: ``schema_tag`` because ``schema`` shadows a StrictModel attribute.
    schema_tag: Literal[CAPABILITY_SPEC_SCHEMA] = Field(
        default=CAPABILITY_SPEC_SCHEMA, alias="schema"
    )
    capability_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]{1,127}$")
    modality: Literal["code", "text", "speech_tts", "structured"] = "code"
    description: str = Field(min_length=1, max_length=2048)
    teacher: TeacherRef
    target_metrics: List[str] = Field(min_length=1, max_length=32)
    controls: List[str] = Field(min_length=1, max_length=32)
    minimum_retention_ratio: float = Field(ge=0.0, le=1.0)
    maximum_control_regression: float = Field(ge=0.0, le=1.0)
    hardware_budget: HardwareBudget = Field(default_factory=HardwareBudget)
    evaluation: EvaluationSplits
    #: Explicit per-run operator consent that a REMOTE teacher connector may
    #: be used. False (default) means any remote connector refuses to run.
    #: Consent never carries over between runs.
    remote_connector_selected: bool = False

    @model_validator(mode="after")
    def _cross_checks(self):
        if not self.controls:
            raise ValueError("controls must not be empty")
        if self.minimum_retention_ratio <= 0:
            raise ValueError("minimum_retention_ratio must be positive")
        return self


def spec_fingerprint(spec: CapabilitySpec) -> str:
    """Canonical sha256 over the spec payload (stable identity for traces,
    footprints and receipts). Serialized with aliases so the digest covers
    the on-disk form (``schema`` key)."""
    from asea.artifacts import digest
    return digest(spec.model_dump(mode="json", by_alias=True))


# ---------------------------------------------------------------------------
# CapabilityTrace -- silt.capability_trace.v1
# ---------------------------------------------------------------------------

class TraceOutcome(StrictModel):
    success: Optional[bool] = None
    tests_passed: Optional[int] = Field(default=None, ge=0)
    tests_total: Optional[int] = Field(default=None, ge=0)
    metrics: Dict[str, Any] = Field(default_factory=dict)


class BehaviouralRecord(StrictModel):
    """What an API/cloud teacher legitimately exposes. No router, expert or
    internal-state fields exist here and can never be added for a remote
    trace."""
    prompt: str = Field(min_length=1, max_length=65536)
    response: str = Field(max_length=131072)
    tool_calls: List[str] = Field(default_factory=list, max_length=256)
    attempts: int = Field(default=0, ge=0)
    corrections: int = Field(default=0, ge=0)
    latency_ms: Optional[float] = Field(default=None, ge=0.0)
    prompt_tokens: Optional[int] = Field(default=None, ge=0)
    response_tokens: Optional[int] = Field(default=None, ge=0)


class InternalRecord(StrictModel):
    """Open-weight instrumented observations: router scores, selected and
    dispatched experts, per-expert output statistics, per layer. Only valid
    for locally controlled weights, collected via the isolated worker."""
    layers: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    activation_statistics: Dict[str, Any] = Field(default_factory=dict)


class CapabilityTrace(StrictModel):
    schema_version: Literal[1] = 1
    schema_tag: Literal[CAPABILITY_TRACE_SCHEMA] = Field(
        default=CAPABILITY_TRACE_SCHEMA, alias="schema"
    )
    capability_id: str = Field(min_length=1, max_length=128)
    #: No ':' (or other Windows-illegal filename chars): the sample id is
    #: embedded verbatim in trace artifact names ("<capability>-<sample_id>"),
    #: and a ':' would make the workspace un-check-outable on Windows
    #: (audit 2026-09-18). Must stay compatible with the dataset builder's
    #: stricter lowercase contract in asea.capability_build.dataset._ID_RE.
    sample_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    model_revision: str = Field(min_length=1, max_length=256)
    prompt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    trace_class: TraceClass
    group: Group
    outcome: TraceOutcome
    behavioural: Optional[BehaviouralRecord] = None
    internal: Optional[InternalRecord] = None

    @model_validator(mode="after")
    def _evidence_class_separation(self):
        if self.trace_class == TRACE_CLASS_BEHAVIOURAL:
            if self.internal is not None or self.behavioural is None:
                raise EvidenceClassErrorValue(
                    "behavioural_remote traces carry a behavioural record only; "
                    "internal observations are forbidden in this evidence class"
                )
        else:
            if self.behavioural is not None or self.internal is None:
                raise EvidenceClassErrorValue(
                    "internal_open_weight traces carry an internal record only; "
                    "behavioural transcripts are forbidden in this evidence class"
                )
        return self


class EvidenceClassErrorValue(ValueError):
    """Raised inside pydantic validation (pydantic wraps it); the typed
    public error lives in :mod:`asea.capability_build.errors`."""


# ---------------------------------------------------------------------------
# CapabilityFootprint -- silt.capability_footprint.v1
#
# Deliberately NOT named CapabilityDiff: asea.capability_diff means comparison
# of receiver capability between approved-set snapshots. This object means
# "computational components associated with and experimentally important to a
# defined capability under a defined workload" -- never "these neurons
# contain coding knowledge".
# ---------------------------------------------------------------------------

class FootprintEvidence(StrictModel):
    trace_class: TraceClass
    target_cases: int = Field(ge=0)
    control_cases: int = Field(ge=0)


class InterventionEntry(StrictModel):
    """One temporary causal intervention on one component. The score is
    Perf(base) - Perf(masked) per group; a large target drop with a small
    control drop is what makes a component interesting."""
    component: str = Field(min_length=1, max_length=256)
    target_drop: float
    control_drop: float
    target_cases: int = Field(ge=0)
    control_cases: int = Field(ge=0)
    restored_and_verified: bool
    #: Seeded determinism marker: identical inputs must reproduce this entry.
    seed: int


class CapabilityFootprint(StrictModel):
    schema_version: Literal[1] = 1
    schema_tag: Literal[CAPABILITY_FOOTPRINT_SCHEMA] = Field(
        default=CAPABILITY_FOOTPRINT_SCHEMA, alias="schema"
    )
    capability: str = Field(min_length=1, max_length=128)
    teacher: str = Field(min_length=1, max_length=256)
    teacher_revision: str = Field(min_length=1, max_length=256)
    spec_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence: FootprintEvidence
    #: Enrichment is CORRELATION ONLY -- usage evidence, not causal expert
    #: importance (compiler core discipline inherited verbatim).
    layers: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    experts: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    causal_interventions: Dict[str, InterventionEntry] = Field(default_factory=dict)
    confidence: Dict[str, Any] = Field(default_factory=dict)
    limitations: List[str] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def _must_state_limitations(self):
        if not self.limitations:
            raise ValueError("a footprint without limitations is invalid on its face")
        return self


# ---------------------------------------------------------------------------
# CapabilityBuildReceipt -- silt.capability_receipt.v1
# ---------------------------------------------------------------------------

#: Everything a receipt may state about a run. Statuses: ``completed`` (the
#: requested command finished), ``rejected`` (validation failure, exit 2) or
#: ``BLOCKED_RESOURCE`` (a required resource was unavailable, honestly).
ReceiptStatus = Literal["completed", "rejected", "BLOCKED_RESOURCE"]


class ResourceMeasurements(StrictModel):
    disk_bytes: Optional[int] = None
    ram_peak_bytes: Optional[int] = None
    vram_peak_bytes: Optional[int] = None
    latency_ms_p50: Optional[float] = None
    tokens_per_second: Optional[float] = None
    energy_joules: Optional[float] = None


class GateStates(StrictModel):
    gate1_status: Optional[str] = None
    gate2_status: Optional[str] = None
    siltspring_states: Dict[str, str] = Field(default_factory=dict)


class CapabilityBuildReceipt(StrictModel):
    schema_version: Literal[1] = 1
    schema_tag: Literal[CAPABILITY_RECEIPT_SCHEMA] = Field(
        default=CAPABILITY_RECEIPT_SCHEMA, alias="schema"
    )
    command: str = Field(min_length=1, max_length=64)
    status: ReceiptStatus
    capability_id: str = Field(min_length=1, max_length=128)
    spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    #: Evidence class actually obtained for this run; never a claim.
    evidence_class: TraceClass
    teacher: Dict[str, str] = Field(default_factory=dict)
    student: Dict[str, str] = Field(default_factory=dict)
    data_split_hashes: Dict[str, str] = Field(default_factory=dict)
    measurements: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    capability_retention: Optional[float] = None
    control_regressions: Dict[str, float] = Field(default_factory=dict)
    resources: ResourceMeasurements = Field(default_factory=ResourceMeasurements)
    gates: GateStates = Field(default_factory=GateStates)
    failure_history: List[str] = Field(default_factory=list, max_length=256)
    limitations: List[str] = Field(min_length=1, max_length=256)
    artifact_hashes: Dict[str, str] = Field(default_factory=dict)
    runtime_versions: Dict[str, str] = Field(default_factory=dict)
    hardware_identity: Dict[str, str] = Field(default_factory=dict)
    error: Optional[str] = None
    #: Set by the signer at materialisation time; excluded from the signed
    #: canonical bytes (LocalSigner contract).
    signature: Optional[str] = None
    signature_alg: Optional[str] = None
    key_fingerprint: Optional[str] = None
    honesty_note: Optional[str] = None

    @model_validator(mode="after")
    def _blocked_needs_error(self):
        if self.status == BLOCKED_RESOURCE and not self.error:
            raise ValueError("BLOCKED_RESOURCE requires an error naming requirement and remedy")
        return self

    @model_validator(mode="after")
    def _placeholder_spec_hash_only_on_blocked(self):
        # The CLI fills spec_sha256 with 64 zeros when a run was blocked
        # before ANY spec could be loaded. That placeholder may ride ONLY
        # on a BLOCKED_RESOURCE receipt; a completed or rejected receipt
        # carrying it would claim to describe a spec it never read
        # (audit 2026-09-18).
        if self.spec_sha256 == "0" * 64 and self.status != BLOCKED_RESOURCE:
            raise ValueError(
                "spec_sha256 is the all-zero placeholder, which only a "
                "BLOCKED_RESOURCE receipt (no spec was loadable) may carry; "
                "a %r receipt must name the real spec hash" % self.status
            )
        return self

    def materialised(self) -> Dict[str, Any]:
        """The dict that gets signed and written: optional unmeasured fields
        are rendered as the literal token ``NOT_MEASURED`` -- never estimated,
        never silently dropped. Dumped with aliases so the signed bytes carry
        the on-disk ``schema`` key."""
        data = self.model_dump(mode="json", by_alias=True)
        for key in ("capability_retention", "error"):
            if data.get(key) is None:
                data[key] = NOT_MEASURED if key == "capability_retention" else None
        resources = dict(data.get("resources") or {})
        for field_name in ("disk_bytes", "ram_peak_bytes", "vram_peak_bytes",
                           "latency_ms_p50", "tokens_per_second", "energy_joules"):
            if resources.get(field_name) is None:
                resources[field_name] = NOT_MEASURED
        data["resources"] = resources
        data.setdefault("honesty_note", None)
        return data

    @classmethod
    def dematerialise(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Reverse :meth:`materialised`: replace the literal ``NOT_MEASURED``
        tokens with ``None`` so a materialised (signed, on-disk) receipt can
        be re-validated against this strict schema. Used by
        :func:`asea.capability_build.receipt.verify_receipt` -- never to
        quietly accept a missing measurement as a number."""
        data = dict(data)
        if data.get("capability_retention") == NOT_MEASURED:
            data["capability_retention"] = None
        if isinstance(data.get("resources"), dict):
            resources = dict(data["resources"])
            for field_name in ("disk_bytes", "ram_peak_bytes", "vram_peak_bytes",
                               "latency_ms_p50", "tokens_per_second", "energy_joules"):
                if resources.get(field_name) == NOT_MEASURED:
                    resources[field_name] = None
            data["resources"] = resources
        if data.get("error") is None:
            data["error"] = None
        return data