"""Typed errors for deep-apply.

Distinct classes so tests can assert on failure *kind*, matching the pattern in
asea.core.errors. These subclass :class:`AseaError` so they ride the same
hierarchy without modifying the core module.
"""

from __future__ import annotations

from ..core.errors import AseaError


class DeepApplyError(AseaError):
    """Base for everything deep-apply raises."""


class DeepApplyIntakeError(DeepApplyError):
    """A packet was refused at training-data intake (not PROMOTED, or mock)."""


class DeepApplyBlocked(DeepApplyError):
    """Deep-apply cannot run here: missing [deep] extra, no CUDA for a big
    model, or an unsupported architecture for the requested backend.

    Always names the requirement and the remedy. Never a silent fallback,
    never a fabricated result.
    """


class AdapterNotPromoted(DeepApplyError):
    """An attempt to admit an adapter that did not pass Gate 2."""