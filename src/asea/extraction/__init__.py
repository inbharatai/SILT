"""SILT GLM-5.3-Flash capability extraction and model conversion
(master brief 2026-09-23): identify the computational structure a
defined capability requires, extract it into a standalone
CompiledCapabilityModel, repair extraction damage where necessary, and
prove retention on a sealed final split through untouched functional
evaluation.

Core imports are pydantic-only (the CI sanity job requires bare
``import asea`` to work without optional extras); anything that touches
model loading lives in the isolated worker, never here.

Binding honesty rules carried by this package:

* frequency is correlation; only controlled interventions support
  causal language;
* fixture tests are FIXTURE_VERIFIED, never real-model evidence;
* unjudged telemetry is storage, never capability evidence (C2);
* the sealed final split is opened exactly once (C6);
* failed attempts are recorded immutably and stay visible (C7);
* if hardware is insufficient the stage reports BLOCKED_RESOURCE, and
  no fixture or estimate is ever substituted for it;
* the recovery trainer never certifies itself.
"""

from asea.extraction.errors import (
    EXIT_BLOCKED,
    EXIT_EXECUTION_FAILURE,
    EXIT_INVALID_EVIDENCE,
    EXIT_REFUSED,
    EXIT_SUCCESS,
    BlockedResource,
    ExtractionError,
    ExecutionFailure,
    InvalidEvidence,
    NotHonestYet,
    Refused,
)
from asea.extraction.schema import (
    CASES_SCHEMA,
    COMPILED_CONFIG_SCHEMA,
    COMPILED_MANIFEST_SCHEMA,
    GRAPH_SCHEMA,
    PLAN_SCHEMA,
    PROVENANCE_SCHEMA,
    RECEIPT_SCHEMA,
    RECOVERY_SCHEMA,
    ROUTING_SCHEMA,
    SOURCE_MANIFEST_SCHEMA,
    SPEC_SCHEMA,
    ActivationObservation,
    CapabilityExtractionReceipt,
    CapabilityExtractionSpec,
    CausalComponentGraph,
    CausalEffectEstimate,
    CheckpointFile,
    CompiledCapabilityConfig,
    CompiledCapabilityManifest,
    ExtractionPlan,
    FunctionalCase,
    FunctionalCaseManifest,
    FunctionalJoin,
    InterventionAttempt,
    RetentionThresholds,
    RoutingObservation,
    SourceCheckpointManifest,
    TensorProvenance,
    TensorTransformation,
    judged_only,
    source_manifest_fingerprint,
)
from asea.extraction.sealed import (
    DEVELOPMENT,
    FINAL_EVALUATION,
    SealedSplit,
)
from asea.extraction.ledger import (
    FailedAttemptLedger,
)

__all__ = [
    "ActivationObservation",
    "BlockedResource",
    "CASES_SCHEMA",
    "EXIT_BLOCKED",
    "EXIT_EXECUTION_FAILURE",
    "EXIT_INVALID_EVIDENCE",
    "EXIT_REFUSED",
    "EXIT_SUCCESS",
    "CapabilityExtractionReceipt",
    "CapabilityExtractionSpec",
    "CausalComponentGraph",
    "CausalEffectEstimate",
    "CheckpointFile",
    "CompiledCapabilityConfig",
    "CompiledCapabilityManifest",
    "DEVELOPMENT",
    "ExtractionError",
    "ExecutionFailure",
    "ExtractionPlan",
    "FailedAttemptLedger",
    "FINAL_EVALUATION",
    "FunctionalCase",
    "FunctionalCaseManifest",
    "FunctionalJoin",
    "InterventionAttempt",
    "InvalidEvidence",
    "NotHonestYet",
    "RECEIPT_SCHEMA",
    "RECOVERY_SCHEMA",
    "Refused",
    "RetentionThresholds",
    "RoutingObservation",
    "SealedSplit",
    "SourceCheckpointManifest",
    "TensorProvenance",
    "TensorTransformation",
    "judged_only",
    "source_manifest_fingerprint",
]