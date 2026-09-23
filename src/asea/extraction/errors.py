"""Typed extraction errors with the CLI exit-code contract (brief C5).

Exit codes (binding, tested against ACTUAL subprocess exit status):

* ``0`` -- successful operation
* ``2`` -- typed refusal / rejected request (the operator asked for
  something the program refuses on principle)
* ``3`` -- invalid evidence or schema (the operator supplied an
  artifact that fails validation)
* ``4`` -- resource blocked (the operation is legitimate but this host
  cannot perform it; the exact blocker and remedy are mandatory)
* ``5`` -- experimental execution failure (the operation started and
  failed; the failed-attempt ledger (C7) must carry the record)

Every error carries ``exit_code`` so the CLI maps it without string
matching, and a one-line ``remedy`` naming what the operator can do.
Errors are pydantic-free plain exceptions so the package core stays
importable with pydantic-only (and error classes with nothing at all).
"""

from __future__ import annotations

EXIT_SUCCESS = 0
EXIT_REFUSED = 2
EXIT_INVALID_EVIDENCE = 3
EXIT_BLOCKED = 4
EXIT_EXECUTION_FAILURE = 5

EXIT_NAMES = {
    EXIT_SUCCESS: "success",
    EXIT_REFUSED: "refused",
    EXIT_INVALID_EVIDENCE: "invalid_evidence",
    EXIT_BLOCKED: "blocked",
    EXIT_EXECUTION_FAILURE: "execution_failure",
}


class ExtractionError(Exception):
    """Base class for extraction-program errors. ``exit_code`` is the
    documented CLI exit status for this failure kind."""

    exit_code = 5

    def __init__(self, message: str, remedy: str):
        self.remedy = remedy
        super().__init__(message)

    def as_dict(self):
        return {
            "error": type(self).__name__,
            "exit_code": self.exit_code,
            "exit_name": EXIT_NAMES.get(self.exit_code, "unknown"),
            "message": str(self),
            "remedy": self.remedy,
        }


class Refused(ExtractionError):
    """Typed refusal (exit 2): the request itself violates a binding
    program rule -- opening the sealed final split from a development
    command, weakening preregistered thresholds, certifying without
    the sealed final evaluation, initializing recovery from a foreign
    pretrained model."""

    exit_code = EXIT_REFUSED


class InvalidEvidence(ExtractionError):
    """Invalid evidence or schema (exit 3): a spec, manifest, trace,
    graph, receipt or other artifact that fails validation. Includes
    unjudged telemetry offered as capability evidence (C2)."""

    exit_code = EXIT_INVALID_EVIDENCE


class BlockedResource(ExtractionError):
    """Resource blocked (exit 4): legitimate operation, insufficient
    host. NEVER substituted with an estimate or a fixture result; the
    message states the requirement, the remedy states how to unblock.
    This is the expected, honest outcome for real-model GLM stages on
    the current 8 GB host."""

    exit_code = EXIT_BLOCKED

    def __init__(self, requirement: str, remedy: str):
        self.requirement = requirement
        super().__init__("blocked: %s" % requirement, remedy)


class ExecutionFailure(ExtractionError):
    """Experimental execution failure (exit 5): the run started and
    failed. The failed-attempt ledger (C7) must hold the immutable
    record of the attempt before this error surfaces."""

    exit_code = EXIT_EXECUTION_FAILURE


class NotHonestYet(Refused):
    """A command whose real-model implementation does not exist yet.
    Honest NOT_IMPLEMENTED refusal (exit 2): the CLI reports the stage
    as NOT_IMPLEMENTED with the status vocabulary instead of faking a
    result. This is deliberately a *refusal*, not exit 5, because the
    program has not attempted the operation at all."""