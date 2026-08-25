"""Typed errors. Distinct classes so tests can assert on failure *kind*."""


class AseaError(Exception):
    """Base for everything this package raises."""


class RegistrationError(AseaError):
    pass


class HandshakeError(AseaError):
    pass


class ExtractionError(AseaError):
    pass


class DistillationError(AseaError):
    pass


class EvaluationError(AseaError):
    pass


class PromotionBlocked(AseaError):
    """Raised when code attempts a promotion the gate forbids."""


class PluginNotFound(AseaError):
    pass


class RollbackError(AseaError):
    pass


class AuditIntegrityError(AseaError):
    pass


class BatchedInferenceError(AseaError):
    """A batched forward pass failed (typically CUDA OOM on a bounded GPU).

    Carries the module, the attempted batch size and the original exception so
    the failure is always typed -- a batched ``infer_batch`` never degrades
    silently to a smaller batch or to single-case inference. The harness
    intentionally does NOT auto-recover: an unasked-for fallback would hide an
    OOM and let a run finish with different (loop-infer) outputs than the batched
    path would have produced, which is exactly the silent-degradation the typed
    errors rule exists to prevent. The operator sets a smaller
    ``max_batch_size`` (or the module a smaller ``preferred_batch_size``) and
    re-runs.
    """

    def __init__(self, module_id: str, batch_size: int, original: BaseException) -> None:
        self.module_id = module_id
        self.batch_size = batch_size
        self.original = original
        super().__init__(
            "batched inference failed for module '{}' at batch_size {}: {}: {}".format(
                module_id, batch_size, type(original).__name__, original
            )
        )


class InferenceCountMismatchError(AseaError):
    """A batched ``infer_batch`` / ``infer_with_skills_batch`` returned the
    wrong number of outputs.

    Returning fewer/more outputs than prompts would misalign outputs to cases
    (output ``i`` must answer ``prompts[i]``), which is silent corruption, not a
    perf shortfall. The harness asserts the count and raises this rather than
    zipping/truncating. Carries the module id, expected and got counts.
    """

    def __init__(self, module_id: str, expected: int, got: int) -> None:
        self.module_id = module_id
        self.expected = expected
        self.got = got
        super().__init__(
            "module '{}' returned {} outputs for {} prompts (batched inference "
            "must preserve order and count)".format(module_id, got, expected)
        )


class CacheCorruptionError(AseaError):
    """A cache entry on disk is present but unreadable.

    Distinct from a clean MISS (key absent -> the cache silently returns None
    and the caller re-runs). A corrupt entry (truncated/invalid JSON, e.g. from
    a crashed write) is a real failure mode with a typed error (audit
    2026-08-17, F2) -- the cache does NOT self-heal by silently discarding it and
    recomputing, because that would hide a disk/serialisation problem. The
    operator deletes the offending file and re-runs. ``put`` writes atomically
    (temp file + ``os.replace``) so a crash mid-write never produces a
    half-written file in the first place.
    """

    def __init__(self, path, original: BaseException) -> None:
        self.path = path
        self.original = original
        super().__init__(
            "corrupt teacher-cache entry at {}: {}: {}".format(
                path, type(original).__name__, original
            )
        )


# -- Capability Diff (B1a, audit 2026-08-17) ---------------------------------
#
# The diff compares a receiver's capability under two approved-set snapshots
# and emits a locally HMAC-signed report. Every failure mode is typed: a
# missing/corrupt snapshot, a missing signing key, and a tampered report each
# raise a distinct error -- nothing silently degrades to an empty diff, an
# unsigned report, or a "valid" verdict.


class DiffError(AseaError):
    """Base for capability-diff failures."""


class SnapshotNotFoundError(DiffError):
    """A diff was asked to load a snapshot token that does not exist (or whose
    path escapes the snapshots directory -- the same containment guard
    :class:`~asea.memory.store.RollbackLayer.rollback` uses). Distinct from an
    EMPTY snapshot (a token whose approved set happens to be empty, which is a
    legitimate, honest delta of zero, not an error)."""

    def __init__(self, token: str) -> None:
        self.token = token
        super().__init__(
            "snapshot token '{}' not found (or escapes the snapshots directory)".format(
                token
            )
        )


# -- Local HMAC signing (shared by B1a capability diff and B3 erasure cert) --
#
# Signing failures are general (any locally HMAC-signed report can have a
# missing key or a tampered payload), so they share a base distinct from
# DiffError -- the erasure certificate reuses them without being a diff.

class SigningError(AseaError):
    """Base for local-HMAC signing failures shared by every signed report
    (capability diff, erasure certificate)."""


class SigningKeyError(SigningError):
    """The local HMAC signing key is missing or unreadable. ``verify`` raises
    this (NOT a silent pass) when it cannot find the key that signed a report --
    a report whose key has vanished cannot be verified, and saying "valid"
    anyway would be a silent forgery. The key is generated on first use and is
    NEVER uploaded (local only; patent pending India)."""

    def __init__(self, path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__("local signing key at {} unusable: {}".format(path, reason))


class SignatureMismatchError(SigningError):
    """A report's HMAC does not match its contents -- the report was edited
    after signing, or signed under a different key. Distinct from a missing key
    (:class:`SigningKeyError`): here the key is present and the math was done,
    and the report is tampered. ``verify`` raises this rather than returning a
    bare ``valid=False`` so a caller cannot mistake a forgery for a fresh,
    unsigned report."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__("signed-report signature mismatch: {}".format(reason))


# -- SPRT early-stop (B2, audit 2026-08-17) ----------------------------------


class SprtError(AseaError):
    """Base for SPRT failures."""


class SprtConfigError(SprtError):
    """An invalid SPRT configuration: ``p0`` must exceed ``p1`` (the
    'unacceptable' packet must regress more often than the 'acceptable' one, or
    the test is inverted and would early-reject GOOD packets), and all rates
    must be in (0, 1). Raised at construction, not silently coerced -- a
    misconfigured SPRT that quietly swapped its hypotheses would early-reject
    every good packet, which is the worst possible silent failure for a gate."""


# -- Verified unlearning (B3, audit 2026-08-17) -------------------------------


class UnlearningError(AseaError):
    """Base for verified-unlearning failures. The erasure certificate's HONESTY
    BOUNDARY is that it verifies SKILL-LAYER unlearning (the packet is gone from
    the approved set the receiver reads and the measured capability reverted to
    baseline) -- NOT weight-level forgetting. A receiver connector with its own
    internal state may retain capability independently; that is out of scope and
    never claimed. Errors here are about malformed verification inputs, not
    about 'did it unlearn' (that is a certificate verdict, not an error)."""
