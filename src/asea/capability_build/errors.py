"""Typed errors for capability-build.

Distinct classes so tests can assert on failure *kind*, matching the
pattern in :mod:`asea.deepapply.errors`. All subclass :class:`AseaError`
so they ride the existing hierarchy without modifying the core module.

Honesty contract (binding, shared with :class:`asea.deepapply.errors.DeepApplyBlocked`):

  * ``BlockedResource`` always names the requirement AND the remedy.
    Never a silent fallback, never a fabricated result.
  * ``RemoteConsentRequired`` is raised BEFORE any network activity when a
    remote (cloud/API) teacher was requested without explicit per-run
    operator consent. No silent remote fallback exists anywhere in this
    package.
  * ``EvidenceClassError`` is raised when behavioural (remote) evidence and
    internal (open-weight) evidence would be mixed in one artifact. The two
    classes are never interchangeable.
"""

from __future__ import annotations

from ..core.errors import AseaError


class CapabilityBuildError(AseaError):
    """Base for everything capability-build raises."""


class BlockedResource(CapabilityBuildError):
    """A required resource is unavailable: insufficient hardware for an
    open-weight teacher, a missing isolated worker runtime, or an absent
    platform capability.

    Always carries ``requirement`` and ``remedy`` attributes naming the
    blocker and the exact action that would unblock it. The run is reported
    as ``BLOCKED_RESOURCE``; it is never silently skipped and never
    fabricated.
    """

    def __init__(self, requirement: str, remedy: str) -> None:
        self.requirement = requirement
        self.remedy = remedy
        super().__init__(
            "blocked: %s. Remedy: %s" % (requirement, remedy)
        )


class RemoteConsentRequired(CapabilityBuildError):
    """A remote (cloud/API) teacher connector was requested without the
    explicit per-run operator consent flag.

    Remedy: pass ``--allow-remote`` on the CLI or set
    ``remote_connector_selected: true`` in the capability spec for THIS run.
    Approval in a previous run never carries over.
    """


class EvidenceClassError(CapabilityBuildError):
    """Behavioural (remote) and internal (open-weight) evidence classes were
    mixed in one trace, footprint or receipt. Never allowed."""


class WorkerUnavailable(CapabilityBuildError):
    """The isolated GLM worker could not start: the container runtime is
    absent, the worker image was not built, or the worker exited before
    completing the protocol handshake."""


class WorkerCrashed(CapabilityBuildError):
    """The isolated GLM worker died mid-request. Partial validated output
    is preserved and reported; a trace is never fabricated for the lost
    portion."""


class WorkerTimeout(WorkerCrashed):
    """The worker exceeded the request deadline and was killed. The
    deadline (and the worker's stderr tail) ride the message; whatever
    the worker had not produced by the deadline is reported as lost, not
    fabricated."""


class SpecInvalid(CapabilityBuildError):
    """A capability spec failed schema or cross-field validation."""


class FootprintInvalid(CapabilityBuildError):
    """A capability footprint failed validation (wrong teacher revision,
    missing controls, or empty evidence)."""


class InterventionInvalid(CapabilityBuildError):
    """A causal intervention was malformed: masking a component that does
    not exist, restoring into a state that does not verify, or an
    intervention that left the teacher modified."""