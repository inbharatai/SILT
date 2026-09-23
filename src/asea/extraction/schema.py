"""Extraction-program schemas (brief section D).

All models are :class:`asea.compose.schema.StrictModel` (extra=forbid,
strict types, frozen, no inf/nan). Every schema carries a versioned tag
under the ``"schema"`` key and the package refuses unknown versions:

  * ``silt.extraction.spec.v1``              -- CapabilityExtractionSpec
  * ``silt.extraction.source_manifest.v1``   -- SourceCheckpointManifest
  * ``silt.extraction.cases.v1``             -- FunctionalCaseManifest
  * ``silt.extraction.routing.v1``           -- RoutingObservation
  * ``silt.extraction.activation.v1``        -- ActivationObservation
  * ``silt.extraction.intervention.v1``     -- InterventionAttempt
  * ``silt.extraction.causal_effect.v1``    -- CausalEffectEstimate
  * ``silt.extraction.graph.v1``            -- CausalComponentGraph
  * ``silt.extraction.plan.v1``             -- ExtractionPlan
  * ``silt.extraction.tensor_transform.v1``  -- TensorTransformation
  * ``silt.extraction.provenance.v1``       -- TensorProvenance
  * ``silt.extraction.compiled_config.v1``  -- CompiledCapabilityConfig
  * ``silt.extraction.compiled_manifest.v1`` -- CompiledCapabilityManifest
  * ``silt.extraction.recovery.v1``         -- RecoveryRun
  * ``silt.extraction.receipt.v1``          -- CapabilityExtractionReceipt

C2 honesty contract encoded here (binding): an internal observation
(RoutingObservation / ActivationObservation) used as CAPABILITY EVIDENCE
must be JOINED to a host-oracle functional outcome through the
:class:`FunctionalJoin` fields (capability id, case id, prompt hash,
source revision, generation-policy hash, oracle artifact hash, verdict).
Unjudged telemetry (``functional_join`` absent) may be STORED but can
never reach footprint enrichment, the causal component graph, or a
receipt: :func:`judged_only` filters it out and every evidence-bearing
builder is required to call the filter. A join whose verdict is null is
not a join.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from pydantic import Field, model_validator

from asea.compose.schema import StrictModel

Hash = str  # 64-hex sha256, pattern-validated where a field is a digest

SPEC_SCHEMA = "silt.extraction.spec.v1"
SOURCE_MANIFEST_SCHEMA = "silt.extraction.source_manifest.v1"
CASES_SCHEMA = "silt.extraction.cases.v1"
ROUTING_SCHEMA = "silt.extraction.routing.v1"
ACTIVATION_SCHEMA = "silt.extraction.activation.v1"
INTERVENTION_SCHEMA = "silt.extraction.intervention.v1"
CAUSAL_EFFECT_SCHEMA = "silt.extraction.causal_effect.v1"
GRAPH_SCHEMA = "silt.extraction.graph.v1"
PLAN_SCHEMA = "silt.extraction.plan.v1"
TENSOR_TRANSFORM_SCHEMA = "silt.extraction.tensor_transform.v1"
PROVENANCE_SCHEMA = "silt.extraction.provenance.v1"
COMPILED_CONFIG_SCHEMA = "silt.extraction.compiled_config.v1"
COMPILED_MANIFEST_SCHEMA = "silt.extraction.compiled_manifest.v1"
RECOVERY_SCHEMA = "silt.extraction.recovery.v1"
RECEIPT_SCHEMA = "silt.extraction.receipt.v1"

_HASH_PATTERN = r"^[0-9a-f]{64}$"

#: Ledger status vocabulary (owner's brief, Section A): an item holds
#: EXACTLY one of these, and a fixture result can never upgrade a
#: real-model status.
STATUS_VALUES = (
    "NOT_IMPLEMENTED",
    "IMPLEMENTED_UNTESTED",
    "FIXTURE_VERIFIED",
    "REAL_MODEL_VERIFIED",
    "BLOCKED_RESOURCE",
    "FAILED_HYPOTHESIS",
    "PARTIAL_RESULT",
    "CERTIFIED",
)

#: Tensor provenance taxonomy (owner's brief, Section P).
PROVENANCE_METHODS = (
    "COPIED_EXACT",
    "SLICED",
    "REINDEXED",
    "MERGED",
    "FACTORIZED",
    "PROJECTED",
    "RECOVERY_TRAINED",
    "NEW_METADATA",
)

#: CausalComponentGraph node classification (owner's brief, Section L).
NODE_ROLES = (
    "REQUIRED",
    "REDUNDANT_UNDER_WORKLOAD",
    "TARGET_ENRICHED",
    "SHARED_GENERAL",
    "NEGATIVE_OR_HARMFUL",
    "UNKNOWN",
    "NOT_TESTED",
)

#: Verdict of the final-split decision (owner's brief, Section T): the
#: thresholds were preregistered BEFORE the final evaluation and are
#: never weakened afterwards.
FINAL_VERDICTS = ("CERTIFIED", "FAILED_HYPOTHESIS", "PARTIAL_RESULT",
                  "NOT_MEASURED", "BLOCKED_RESOURCE")


class FunctionalJoin(StrictModel):
    """The C2 join: an internal observation bound to a host-oracle
    functional outcome. Every field the brief names is required, and
    ``verdict`` MUST be a concrete boolean -- null verdicts are rejected
    at schema level (``success = null`` is not evidence)."""

    capability_id: str = Field(min_length=1, max_length=128)
    case_id: str = Field(min_length=1, max_length=128)
    prompt_hash: str = Field(pattern=_HASH_PATTERN)
    source_revision: str = Field(min_length=1, max_length=256)
    generation_policy_hash: str = Field(pattern=_HASH_PATTERN)
    oracle_artifact_hash: str = Field(pattern=_HASH_PATTERN)
    verdict: bool


# ---------------------------------------------------------------------------
# CapabilityExtractionSpec -- silt.extraction.spec.v1
# ---------------------------------------------------------------------------

class SourceModelRef(StrictModel):
    repository: str = Field(min_length=1, max_length=256)
    model: str = Field(min_length=1, max_length=256)
    variant: str = Field(min_length=1, max_length=256)
    commit_sha: str = Field(min_length=1, max_length=64)
    license: str = Field(min_length=1, max_length=256)


class RetentionThresholds(StrictModel):
    """Preregistered decision thresholds. Recorded in the spec BEFORE any
    measurement; weakening them after seeing final results is forbidden
    by the program, and the receipt echoes the ORIGINAL values."""

    minimum_retention_ratio: float = Field(ge=0.0, le=1.0)
    maximum_stored_parameter_reduction: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _positive(self):
        if self.minimum_retention_ratio <= 0.0:
            raise ValueError("minimum_retention_ratio must be positive")
        if self.maximum_stored_parameter_reduction <= 0.0:
            raise ValueError("maximum_stored_parameter_reduction must be positive")
        return self


class CapabilityExtractionSpec(StrictModel):
    schema_version: Literal[1] = 1
    schema_tag: Literal[SPEC_SCHEMA] = Field(default=SPEC_SCHEMA, alias="schema")
    capability_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]{1,127}$")
    description: str = Field(min_length=1, max_length=4096)
    source_model: SourceModelRef
    thresholds: RetentionThresholds
    target_task_families: List[str] = Field(min_length=1, max_length=32)
    control_task_families: List[str] = Field(min_length=1, max_length=32)
    evaluation: Dict[str, str] = Field(min_length=1)
    #: The final split lives in a physically sealed artifact (C6); this is
    #: its path, which development commands must refuse to open.
    sealed_final_path: str = Field(min_length=1, max_length=2048)

    @model_validator(mode="after")
    def _cross_checks(self):
        overlap = (set(self.target_task_families)
                   & set(self.control_task_families))
        if overlap:
            raise ValueError(
                "target and control task families must be disjoint; "
                "shared: %s" % sorted(overlap))
        if "final" not in self.evaluation:
            raise ValueError("evaluation must name the 'final' split")
        return self


# ---------------------------------------------------------------------------
# SourceCheckpointManifest -- silt.extraction.source_manifest.v1 (C3)
# ---------------------------------------------------------------------------

class CheckpointFile(StrictModel):
    filename: str = Field(min_length=1, max_length=512)
    bytes: int = Field(ge=0)
    sha256: str = Field(pattern=_HASH_PATTERN)
    #: LFS/Xet object id where the platform provides one (None allowed;
    #: the sha256 is always required).
    object_id: Optional[str] = Field(default=None, max_length=256)


def source_manifest_fingerprint(files: Dict[str, CheckpointFile]) -> str:
    """The deterministic aggregate digest: sha256 over sorted
    ``<filename>:<sha256>:<bytes>`` lines. Reordering the dict cannot
    change it."""
    import hashlib

    lines = sorted(
        "%s:%s:%d" % (name, entry.sha256, entry.bytes)
        for name, entry in files.items()
    )
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


class SourceCheckpointManifest(StrictModel):
    """EXACT source-checkpoint identity (C3): every shard, tokenizer and
    processor file, config, generation config and chat template hashed;
    plus a deterministic aggregate (Merkle) fingerprint. A manifest with
    a missing REQUIRED file (config.json, generation_config.json,
    chat template, at least one weight shard) is invalid -- exact
    identity is claimed only when every required shard is verified."""

    schema_version: Literal[1] = 1
    schema_tag: Literal[SOURCE_MANIFEST_SCHEMA] = Field(
        default=SOURCE_MANIFEST_SCHEMA, alias="schema")
    repository: str = Field(min_length=1, max_length=256)
    model: str = Field(min_length=1, max_length=256)
    variant: str = Field(min_length=1, max_length=256)
    commit_sha: str = Field(min_length=1, max_length=64)
    license: str = Field(min_length=1, max_length=256)
    files: Dict[str, CheckpointFile] = Field(min_length=1)
    #: Where the chat template lives in THIS checkpoint tree: either a
    #: dedicated ``chat_template.jinja`` or the ``chat_template`` key
    #: inside ``tokenizer_config.json`` (the common HF layout). C3
    #: requires the chat-template IDENTITY, not a specific filename.
    chat_template_source: Literal["chat_template.jinja",
                                  "tokenizer_config.json"]
    #: Deterministic aggregate: sha256 over the sorted
    #: "<filename>:<sha256>:<bytes>" lines (the Merkle-style fingerprint
    #: of the whole tree).
    aggregate_sha256: str = Field(pattern=_HASH_PATTERN)

    @model_validator(mode="after")
    def _required_files(self):
        names = set(self.files)
        for required in ("config.json", "generation_config.json",
                         "tokenizer_config.json"):
            if required not in names:
                raise ValueError(
                    "exact checkpoint identity requires %r to be hashed; "
                    "a manifest without it is not an exact identity" % required)
        if not any(name.endswith(".safetensors") for name in names):
            raise ValueError(
                "exact checkpoint identity requires at least one weight "
                "shard (.safetensors) digest")
        if self.chat_template_source not in names:
            raise ValueError(
                "chat_template_source %r is not one of the hashed files"
                % self.chat_template_source)
        expected = source_manifest_fingerprint(self.files)
        if self.aggregate_sha256 != expected:
            raise ValueError(
                "aggregate_sha256 %s does not match the hashed files "
                "(expected %s); exact identity is claimed only when the "
                "aggregate IS the tree" % (self.aggregate_sha256,
                                           expected))
        return self


# ---------------------------------------------------------------------------
# FunctionalCaseManifest -- silt.extraction.cases.v1
# ---------------------------------------------------------------------------

class FunctionalCase(StrictModel):
    case_id: str = Field(min_length=1, max_length=128)
    family: str = Field(min_length=1, max_length=128)
    group: Literal["target", "control", "required", "descriptive"]
    content_sha256: str = Field(pattern=_HASH_PATTERN)
    prompt_sha256: str = Field(pattern=_HASH_PATTERN)
    license: str = Field(min_length=1, max_length=256)


class FunctionalCaseManifest(StrictModel):
    schema_version: Literal[1] = 1
    schema_tag: Literal[CASES_SCHEMA] = Field(default=CASES_SCHEMA, alias="schema")
    capability_id: str = Field(min_length=1, max_length=128)
    split: Literal["training", "development", "heldout", "final", "controls"]
    cases: List[FunctionalCase] = Field(min_length=1)
    #: Near-duplicate guard: no two cases in one split may share a prompt
    #: hash, and no case may appear in more than one split of the same
    #: capability (enforced by the builder; the manifest carries the
    #: declaration for the receipt).
    near_duplicate_free: bool

    @model_validator(mode="after")
    def _unique(self):
        prompts = [case.prompt_sha256 for case in self.cases]
        if len(set(prompts)) != len(prompts):
            raise ValueError("duplicate prompt hashes within the split")
        families = set(case.family for case in self.cases)
        if not families:
            raise ValueError("a case manifest needs at least one family")
        if not self.near_duplicate_free:
            raise ValueError(
                "near_duplicate_free must be true: manifests are produced "
                "by the builder AFTER the near-duplicate checks pass")
        return self


# ---------------------------------------------------------------------------
# Internal observations (C2: judged telemetry only reaches evidence)
# ---------------------------------------------------------------------------

class RoutingObservation(StrictModel):
    """One routing-telemetry record. ``functional_join`` is OPTIONAL for
    STORAGE (unjudged telemetry may be kept), but
    :func:`judged_only` excludes every record without a complete join
    from anything that becomes capability evidence."""

    schema_version: Literal[1] = 1
    schema_tag: Literal[ROUTING_SCHEMA] = Field(
        default=ROUTING_SCHEMA, alias="schema")
    capability_id: str = Field(min_length=1, max_length=128)
    layer_id: int = Field(ge=0)
    expert_id: int = Field(ge=0)
    dispatched_count: int = Field(ge=0)
    sigmoid_mass: float = Field(ge=0.0)
    source_revision: str = Field(min_length=1, max_length=256)
    functional_join: Optional[FunctionalJoin] = None

    @model_validator(mode="after")
    def _join_binds_same_revision(self):
        join = self.functional_join
        if join is not None and join.source_revision != self.source_revision:
            raise ValueError(
                "functional join %r was judged against source revision %r, "
                "not %r -- a join that crosses revisions is not evidence"
                % (join.case_id, join.source_revision, self.source_revision))
        return self


class ActivationObservation(StrictModel):
    schema_version: Literal[1] = 1
    schema_tag: Literal[ACTIVATION_SCHEMA] = Field(
        default=ACTIVATION_SCHEMA, alias="schema")
    capability_id: str = Field(min_length=1, max_length=128)
    layer_id: int = Field(ge=0)
    expert_id: int = Field(ge=0)
    activation_l2_mean: float = Field(ge=0.0)
    observations: int = Field(ge=1)
    source_revision: str = Field(min_length=1, max_length=256)
    functional_join: Optional[FunctionalJoin] = None

    @model_validator(mode="after")
    def _join_binds_same_revision(self):
        join = self.functional_join
        if join is not None and join.source_revision != self.source_revision:
            raise ValueError(
                "functional join %r was judged against source revision %r, "
                "not %r -- a join that crosses revisions is not evidence"
                % (join.case_id, join.source_revision, self.source_revision))
        return self


def judged_only(records: Sequence[Any]) -> List[Any]:
    """C2 enforcement: only records with a COMPLETE functional join may
    become capability evidence. Unjudged telemetry is storage, never
    enrichment; a join with a null verdict does not exist at schema level
    (FunctionalJoin.verdict is a required bool)."""
    return [record for record in records
            if getattr(record, "functional_join", None) is not None]


# ---------------------------------------------------------------------------
# InterventionAttempt -- silt.extraction.intervention.v1
# ---------------------------------------------------------------------------

class InterventionAttempt(StrictModel):
    schema_version: Literal[1] = 1
    schema_tag: Literal[INTERVENTION_SCHEMA] = Field(
        default=INTERVENTION_SCHEMA, alias="schema")
    capability_id: str = Field(min_length=1, max_length=128)
    component: str = Field(min_length=1, max_length=256)
    target_drop: float
    control_drop: float
    target_cases: int = Field(ge=1)
    control_cases: int = Field(ge=1)
    seed: int = Field(ge=0)
    restored_and_verified: bool
    source_revision: str = Field(min_length=1, max_length=256)
    functional_joins: List[FunctionalJoin] = Field(min_length=1)

    @model_validator(mode="after")
    def _verified_or_flagged(self):
        if not self.restored_and_verified:
            raise ValueError(
                "an unrestorable intervention is never a valid causal "
                "record; store it in the failed-attempt ledger (C7) instead")
        return self


# ---------------------------------------------------------------------------
# CausalEffectEstimate -- silt.extraction.causal_effect.v1
# ---------------------------------------------------------------------------

class CausalEffectEstimate(StrictModel):
    """One component's causal effect under controlled intervention, with
    its matched-control arm (Section M): the estimate is only causal when
    the intervention was real, restored and verified, AND exceeds the
    matched control selector's effect under the same protocol."""

    schema_version: Literal[1] = 1
    schema_tag: Literal[CAUSAL_EFFECT_SCHEMA] = Field(
        default=CAUSAL_EFFECT_SCHEMA, alias="schema")
    capability_id: str = Field(min_length=1, max_length=128)
    component: str = Field(min_length=1, max_length=256)
    effect: float
    control_arm_effect: float
    attempts: int = Field(ge=1)
    seed_set: List[int] = Field(min_length=1, max_length=64)
    source_revision: str = Field(min_length=1, max_length=256)


# ---------------------------------------------------------------------------
# CausalComponentGraph -- silt.extraction.graph.v1
# ---------------------------------------------------------------------------

class GraphNode(StrictModel):
    component: str = Field(min_length=1, max_length=256)
    role: Literal[tuple(NODE_ROLES)]  # type: ignore[valid-type]
    evidence_count: int = Field(ge=0)
    #: A NEGATIVE_OR_HARMFUL / REQUIRED role requires at least one
    #: verified intervention attempt; correlation-only nodes can only be
    #: TARGET_ENRICHED / SHARED_GENERAL / UNKNOWN / NOT_TESTED.
    causal_evidence: bool


class CausalComponentGraph(StrictModel):
    schema_version: Literal[1] = 1
    schema_tag: Literal[GRAPH_SCHEMA] = Field(
        default=GRAPH_SCHEMA, alias="schema")
    capability_id: str = Field(min_length=1, max_length=128)
    source_revision: str = Field(min_length=1, max_length=256)
    nodes: List[GraphNode] = Field(min_length=1)
    limitations: List[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _roles_need_causes(self):
        for node in self.nodes:
            if node.role in ("REQUIRED", "NEGATIVE_OR_HARMFUL") \
                    and not node.causal_evidence:
                raise ValueError(
                    "node %s carries role %s without causal evidence; "
                    "correlation alone can never support that role"
                    % (node.component, node.role))
        return self


# ---------------------------------------------------------------------------
# ExtractionPlan -- silt.extraction.plan.v1
# ---------------------------------------------------------------------------

class ExpertRetention(StrictModel):
    layer_id: int = Field(ge=0)
    retained_experts: List[int] = Field(min_length=0)
    selection_basis: Literal["causal", "matched_random", "frequency",
                             "dispatch", "magnitude"]


class ExtractionPlan(StrictModel):
    """The conservative first-extraction plan (Section N): tokenizer,
    embeddings, dense layers, attention and head retained; a causally
    selected routed-expert subset per layer; routers rebuilt to the
    retained set."""

    schema_version: Literal[1] = 1
    schema_tag: Literal[PLAN_SCHEMA] = Field(default=PLAN_SCHEMA, alias="schema")
    capability_id: str = Field(min_length=1, max_length=128)
    source_manifest_sha256: str = Field(pattern=_HASH_PATTERN)
    graph_sha256: str = Field(pattern=_HASH_PATTERN)
    retained_dense: bool
    retained_tokenizer: bool
    retained_attention: bool
    retained_head: bool
    expert_retention: List[ExpertRetention] = Field(min_length=1)
    selection_arm: Literal["causal", "matched_random", "frequency",
                           "dispatch", "magnitude"]

    @model_validator(mode="after")
    def _conservative_defaults(self):
        if not (self.retained_dense and self.retained_tokenizer
                and self.retained_attention and self.retained_head):
            raise ValueError(
                "the first extraction is conservative: tokenizer, "
                "embeddings, dense layers, attention and head must be "
                "retained; plans that drop them are out of scope for v1")
        return self


# ---------------------------------------------------------------------------
# TensorTransformation / TensorProvenance (P)
# ---------------------------------------------------------------------------

class TensorTransformation(StrictModel):
    source_tensor: str = Field(min_length=1, max_length=512)
    target_tensor: str = Field(min_length=1, max_length=512)
    method: Literal[tuple(PROVENANCE_METHODS)]  # type: ignore[valid-type]


class TensorProvenance(StrictModel):
    schema_version: Literal[1] = 1
    schema_tag: Literal[PROVENANCE_SCHEMA] = Field(
        default=PROVENANCE_SCHEMA, alias="schema")
    manifest_sha256: str = Field(pattern=_HASH_PATTERN)
    transformations: List[TensorTransformation] = Field(min_length=1)
    source_manifest_sha256: str = Field(pattern=_HASH_PATTERN)

    @model_validator(mode="after")
    def _every_method_recorded(self):
        methods = set(t.method for t in self.transformations)
        if "RECOVERY_TRAINED" in methods and "COPIED_EXACT" not in methods:
            raise ValueError(
                "recovery training repairs an EXTRACTED artifact; a "
                "provenance record with no copied source tensor describes "
                "a model that was not extracted from the source")
        return self


# ---------------------------------------------------------------------------
# CompiledCapabilityConfig / Manifest (N/O)
# ---------------------------------------------------------------------------

class CompiledCapabilityConfig(StrictModel):
    """The standalone model's own config: a GLM-derived architecture with
    a reduced routed-expert set per layer and REBUILT routers (the
    original router weight rows for retained experts are reindexed, never
    re-learned outside recovery training). Named the
    CompiledCapabilityModel; never 'student', 'adapter', 'small GLM' or
    'extracted intelligence'."""

    schema_version: Literal[1] = 1
    schema_tag: Literal[COMPILED_CONFIG_SCHEMA] = Field(
        default=COMPILED_CONFIG_SCHEMA, alias="schema")
    architecture: Literal["glm5_next_moe_derived"] = "glm5_next_moe_derived"
    num_hidden_layers: int = Field(ge=1)
    hidden_size: int = Field(ge=1)
    n_routed_experts_per_layer: Dict[str, int] = Field(min_length=1)
    num_experts_per_tok: int = Field(ge=1)
    first_k_dense_replace: int = Field(ge=0)
    vocab_size: int = Field(ge=1)
    torch_dtype: Literal["bfloat16"] = "bfloat16"


class CompiledCapabilityManifest(StrictModel):
    schema_version: Literal[1] = 1
    schema_tag: Literal[COMPILED_MANIFEST_SCHEMA] = Field(
        default=COMPILED_MANIFEST_SCHEMA, alias="schema")
    name: str = Field(min_length=1, max_length=512)
    capability_id: str = Field(min_length=1, max_length=128)
    config: CompiledCapabilityConfig
    plan_sha256: str = Field(pattern=_HASH_PATTERN)
    provenance_sha256: str = Field(pattern=_HASH_PATTERN)
    source_manifest_sha256: str = Field(pattern=_HASH_PATTERN)
    stored_parameters: int = Field(ge=1)
    source_stored_parameters: int = Field(ge=1)

    @property
    def stored_parameter_reduction(self) -> float:
        return 1.0 - (self.stored_parameters / self.source_stored_parameters)


# ---------------------------------------------------------------------------
# RecoveryRun -- silt.extraction.recovery.v1 (R)
# ---------------------------------------------------------------------------

class RecoveryRun(StrictModel):
    """Recovery training repairs extraction damage on the EXTRACTED
    artifact only. Initialization from Qwen, Gemma, MiniCPM or any other
    pretrained model is forbidden; a run whose base is not the extracted
    artifact is invalid. The trainer never certifies itself: the verdict
    lives in the receipt's independent evaluation, not here."""

    schema_version: Literal[1] = 1
    schema_tag: Literal[RECOVERY_SCHEMA] = Field(
        default=RECOVERY_SCHEMA, alias="schema")
    capability_id: str = Field(min_length=1, max_length=128)
    compiled_manifest_sha256: str = Field(pattern=_HASH_PATTERN)
    initialized_from: Literal["extracted_artifact"] = "extracted_artifact"
    steps: int = Field(ge=1)
    trainable_parameters: int = Field(ge=1)
    final_loss: float = Field(ge=0.0)
    data_boundary: Literal["training_split_only"] = "training_split_only"


# ---------------------------------------------------------------------------
# CapabilityExtractionReceipt -- silt.extraction.receipt.v1 (V)
# ---------------------------------------------------------------------------

class CapabilityExtractionReceipt(StrictModel):
    """Immutable end-to-end record. Unmeasured metrics are the literal
    ``NOT_MEASURED`` string, never an estimate. The preregistered
    thresholds are echoed VERBATIM from the spec; a receipt whose
    thresholds differ from the spec it pins is invalid."""

    schema_version: Literal[1] = 1
    schema_tag: Literal[RECEIPT_SCHEMA] = Field(
        default=RECEIPT_SCHEMA, alias="schema")
    capability_id: str = Field(min_length=1, max_length=128)
    command: str = Field(min_length=1, max_length=256)
    spec_sha256: str = Field(pattern=_HASH_PATTERN)
    source_manifest_sha256: str = Field(pattern=_HASH_PATTERN)
    preregistered_thresholds: RetentionThresholds
    retained_ratio: Union[Literal["NOT_MEASURED"], float]
    stored_parameter_reduction: Union[Literal["NOT_MEASURED"], float]
    control_regressions: Dict[str, Union[Literal["NOT_MEASURED"], float]] = Field(
        default_factory=dict)
    final_verdict: Literal[tuple(FINAL_VERDICTS)]  # type: ignore[valid-type]
    final_split_evaluated_exactly_once: bool
    failure_history: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _verdict_discipline(self):
        if self.final_verdict in ("CERTIFIED", "FAILED_HYPOTHESIS",
                                  "PARTIAL_RESULT") \
                and not self.final_split_evaluated_exactly_once:
            raise ValueError(
                "a %s verdict requires the sealed final split to have "
                "been evaluated exactly once" % self.final_verdict)
        if self.final_verdict == "CERTIFIED":
            if self.retained_ratio == "NOT_MEASURED" \
                    or self.stored_parameter_reduction == "NOT_MEASURED":
                raise ValueError(
                    "CERTIFIED requires measured retention and reduction")
            if self.retained_ratio < self.preregistered_thresholds.minimum_retention_ratio:
                raise ValueError(
                    "retained ratio %r below the preregistered threshold "
                    "%r -- the verdict cannot be CERTIFIED"
                    % (self.retained_ratio,
                       self.preregistered_thresholds.minimum_retention_ratio))
            if self.stored_parameter_reduction < self.preregistered_thresholds.maximum_stored_parameter_reduction:
                raise ValueError(
                    "parameter reduction %r below the preregistered "
                    "threshold %r -- the verdict cannot be CERTIFIED"
                    % (self.stored_parameter_reduction,
                       self.preregistered_thresholds.maximum_stored_parameter_reduction))
        return self