"""SiltStream named errors.

Every failure mode raises a typed, named error. Nothing returns a fake
success, and nothing silently falls back to a different execution mode.
"""

from __future__ import annotations


class SiltStreamError(RuntimeError):
    """Base class for all SiltStream errors."""


class UnsupportedModelError(SiltStreamError):
    """The model architecture cannot be streamed. Names the reason."""


class StorageError(SiltStreamError):
    """The layer bank could not store or fetch a layer."""


class ParityError(SiltStreamError):
    """Streamed execution diverged from resident execution beyond tolerance."""


class BackendUnavailableError(SiltStreamError):
    """The requested compute device / feature is not available on this host."""
